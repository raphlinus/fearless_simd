// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::Ident;

use crate::ops::{load_interleaved_arg_ty, store_interleaved_arg_ty, valid_reinterpret};
use crate::{
    arch::{Arch, wasm::Wasm},
    generic::{generic_combine, generic_op, generic_split},
    ops::{OpSig, TyFlavor, ops_for_type},
    types::{SIMD_TYPES, ScalarType, VecType, type_imports},
};

#[derive(Clone, Copy)]
pub enum Level {
    WasmSimd128,
}

impl Level {
    fn name(self) -> &'static str {
        match self {
            Level::WasmSimd128 => "WasmSimd128",
        }
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

fn mk_simd_impl(level: Level) -> TokenStream {
    let level_tok = level.token();
    let mut methods = vec![];

    for vec_ty in SIMD_TYPES {
        let scalar_bits = vec_ty.scalar_bits;
        let ty_name = vec_ty.rust_name();
        let ty = vec_ty.rust();

        for (method, sig) in ops_for_type(vec_ty, true) {
            let b1 = vec_ty.n_bits() > 128 && !matches!(method, "split" | "narrow")
                || vec_ty.n_bits() > 256;
            let b2 = !matches!(method, "load_interleaved_128")
                && !matches!(method, "store_interleaved_128");

            if b1 && b2 {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }

            let method_name = format!("{method}_{ty_name}");
            let method_ident = Ident::new(&method_name, Span::call_site());
            let ret_ty = sig.ret_ty(vec_ty, TyFlavor::SimdTrait);
            let m = match sig {
                OpSig::Splat => {
                    let scalar = vec_ty.scalar.rust(scalar_bits);
                    let expr = Wasm.expr(method, vec_ty, &[quote! { val }]);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, val: #scalar) -> #ty<Self> {
                            #expr.simd_into(self)
                        }
                    }
                }
                OpSig::Unary => {
                    let args = [quote! { a.into() }];
                    let expr = if matches!(method, "fract") {
                        assert_eq!(ty_name, "f32x4", "only support fract_f32x4");
                        quote! {
                            self.sub_f32x4(a, self.trunc_f32x4(a))
                        }
                    } else {
                        let expr = Wasm.expr(method, vec_ty, &args);
                        quote! { #expr.simd_into(self) }
                    };
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            #expr
                        }
                    }
                }
                OpSig::Binary if method == "copysign" => {
                    assert_eq!(ty_name, "f32x4", "only support copysign_f32x4");
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            let sign_mask = f32x4_splat(-0.0_f32);
                            let sign_bits = v128_and(b.into(), sign_mask.into());
                            let magnitude = v128_andnot(a.into(), sign_mask.into());
                            v128_or(magnitude, sign_bits).simd_into(self)
                        }
                    }
                }
                OpSig::Binary => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    match method {
                        "mul" if vec_ty.scalar_bits == 8 && vec_ty.len == 16 => {
                            let (extmul_low, extmul_high) = match vec_ty.scalar {
                                ScalarType::Unsigned => (
                                    quote! { u16x8_extmul_low_u8x16 },
                                    quote! { u16x8_extmul_high_u8x16 },
                                ),
                                ScalarType::Int => (
                                    quote! { i16x8_extmul_low_i8x16 },
                                    quote! { i16x8_extmul_high_i8x16 },
                                ),
                                _ => unreachable!(),
                            };

                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                    let low = #extmul_low(a.into(), b.into());
                                    let high = #extmul_high(a.into(), b.into());
                                    u8x16_shuffle::<0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30>(low, high).simd_into(self)
                                }
                            }
                        }
                        "max_precise" | "min_precise" => {
                            // For `max_precise` and `min_precise` the arguments are switched such
                            // that `max(NaN, x)` and `min(NaN, x)` result in `x`. This matches
                            // `_mm_max_ps` and `_mm_min_ps` semantics on x86.
                            let swapped_args = [quote! { b.into() }, quote! { a.into() }];
                            let expr: TokenStream = Wasm.expr(method, vec_ty, &swapped_args);
                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                    #expr.simd_into(self)
                                }
                            }
                        }
                        _ => {
                            let expr = Wasm.expr(method, vec_ty, &args);
                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                    #expr.simd_into(self)
                                }
                            }
                        }
                    }
                }
                OpSig::Ternary => {
                    if matches!(method, "madd" | "msub") {
                        let first_ident = {
                            let str = if method == "madd" {
                                "add_f32x4"
                            } else {
                                "sub_f32x4"
                            };

                            Ident::new(str, Span::call_site())
                        };

                        assert_eq!(
                            vec_ty,
                            &VecType {
                                scalar: ScalarType::Float,
                                scalar_bits: 32,
                                len: 4,
                            }
                        );
                        // TODO: `relaxed-simd` has madd.
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                                self.#first_ident(a, self.mul_f32x4(b, c))
                            }
                        }
                    } else {
                        unimplemented!()
                    }
                }
                OpSig::Compare => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    let expr = Wasm.expr(method, vec_ty, &args);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            #expr.simd_into(self)
                        }
                    }
                }
                OpSig::Select => {
                    let mask_ty = vec_ty.mask_ty().rust();
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                            v128_bitselect(b.into(), c.into(), a.into()).simd_into(self)
                        }
                    }
                }
                OpSig::Combine => generic_combine(vec_ty),
                OpSig::Split => generic_split(vec_ty),
                OpSig::Zip(is_low) => {
                    let (indices, shuffle_fn) = match vec_ty.scalar_bits {
                        8 => {
                            let indices = if is_low {
                                quote! { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 }
                            } else {
                                quote! { 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 }
                            };
                            (indices, quote! { u8x16_shuffle })
                        }
                        16 => {
                            let indices = if is_low {
                                quote! { 0, 8, 1, 9, 2, 10, 3, 11 }
                            } else {
                                quote! { 4, 12, 5, 13, 6, 14, 7, 15 }
                            };
                            (indices, quote! { u16x8_shuffle })
                        }
                        32 => {
                            let indices = if is_low {
                                quote! { 0, 4, 1, 5 }
                            } else {
                                quote! { 2, 6, 3, 7 }
                            };
                            (indices, quote! { u32x4_shuffle })
                        }
                        _ => panic!("unsupported scalar_bits for zip operation"),
                    };

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            #shuffle_fn::<#indices>(a.into(), b.into()).simd_into(self)
                        }
                    }
                }
                OpSig::Shift => {
                    let prefix = match vec_ty.scalar {
                        ScalarType::Int => "i",
                        ScalarType::Unsigned => "u",
                        _ => unimplemented!(),
                    };
                    let shift_name = format!("{prefix}{}x{}_shr", vec_ty.scalar_bits, vec_ty.len);
                    let shift_fn = Ident::new(&shift_name, Span::call_site());

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, shift: u32) -> #ret_ty {
                            #shift_fn(a.into(), shift).simd_into(self)
                        }
                    }
                }
                OpSig::Reinterpret(scalar, scalar_bits) => {
                    // The underlying data for WASM SIMD is a v128, so a reinterpret is just that, a
                    // reinterpretation of the v128.
                    assert!(valid_reinterpret(vec_ty, scalar, scalar_bits));

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            <v128>::from(a).simd_into(self)
                        }
                    }
                }
                OpSig::Cvt(scalar, scalar_bits) => {
                    let conversion_fn =
                        match (scalar, scalar_bits, vec_ty.scalar, vec_ty.scalar_bits) {
                            (ScalarType::Unsigned, 32, ScalarType::Float, 32) => {
                                quote! { u32x4_trunc_sat_f32x4 }
                            }
                            (ScalarType::Float, 32, ScalarType::Unsigned, 32) => {
                                quote! { f32x4_convert_u32x4 }
                            }
                            _ => unimplemented!(),
                        };

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            #conversion_fn(a.into()).simd_into(self)
                        }
                    }
                }
                OpSig::WidenNarrow(to_ty) => {
                    match method {
                        "widen" => {
                            assert_eq!(vec_ty.rust_name(), "u8x16");
                            assert_eq!(to_ty.rust_name(), "u16x16");
                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                    let low = u16x8_extend_low_u8x16(a.into());
                                    let high = u16x8_extend_high_u8x16(a.into());
                                    self.combine_u16x8(low.simd_into(self), high.simd_into(self))
                                }
                            }
                        }
                        "narrow" => {
                            assert_eq!(vec_ty.rust_name(), "u16x16");
                            assert_eq!(to_ty.rust_name(), "u8x16");
                            // WASM SIMD only has saturating narrowing instructions, so we emulate
                            // truncated narrowing by masking out the
                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                    let mask = u16x8_splat(0xFF);
                                    let (low, high) = self.split_u16x16(a);
                                    let low_masked = v128_and(low.into(), mask);
                                    let high_masked = v128_and(high.into(), mask);
                                    let result = u8x16_narrow_i16x8(low_masked, high_masked);
                                    result.simd_into(self)
                                }
                            }
                        }
                        _ => unimplemented!(),
                    }
                    // let to_ty = &VecType::new(scalar, scalar_bits, vec_ty.len);
                }
                OpSig::LoadInterleaved(block_size, count) => {
                    assert_eq!(count, 4, "only count of 4 is crrently supported");
                    let arg = load_interleaved_arg_ty(block_size, count, vec_ty);
                    let elems_per_vec = block_size as usize / vec_ty.scalar_bits;

                    // For WASM we need to simulate interleaving with shuffle, and we only have
                    // access to 2, 4 and 16 lanes. So, for 64 u8's, we need to split and recombine
                    // the vectors.
                    let (lower_indices, upper_indices, shuffle_fn) = match vec_ty.scalar_bits {
                        8 => (
                            quote! { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 },
                            quote! { 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 },
                            quote! { u8x16_shuffle },
                        ),
                        16 => (
                            quote! { 0, 8, 1, 9, 2, 10, 3, 11 },
                            quote! { 4, 12, 5, 13, 6, 14, 7, 15 },
                            quote! { u16x8_shuffle },
                        ),
                        32 => (
                            quote! { 0, 4, 1, 5 },
                            quote! { 2, 6, 3, 7 },
                            quote! { u32x4_shuffle },
                        ),
                        _ => panic!("unsupported scalar_bits"),
                    };

                    let combine_method_name =
                        |scalar: ScalarType, scalar_bits: usize, lane_count: usize| -> Ident {
                            let scalar = match scalar {
                                ScalarType::Float => 'f',
                                ScalarType::Unsigned => 'u',
                                _ => unimplemented!(),
                            };
                            format_ident!("combine_{scalar}{scalar_bits}x{lane_count}")
                        };

                    let combine_method =
                        combine_method_name(vec_ty.scalar, vec_ty.scalar_bits, elems_per_vec);
                    let combine_method_2x =
                        combine_method_name(vec_ty.scalar, vec_ty.scalar_bits, elems_per_vec * 2);

                    let combine_code = quote! {
                        let combined_lower = self.#combine_method(out0.simd_into(self), out1.simd_into(self));
                        let combined_upper = self.#combine_method(out2.simd_into(self), out3.simd_into(self));
                        self.#combine_method_2x(combined_lower, combined_upper)
                    };

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, #arg) -> #ret_ty {
                                let v0: v128 = unsafe { v128_load(src[0 * #elems_per_vec..].as_ptr() as *const v128) };
                                let v1: v128 = unsafe { v128_load(src[1 * #elems_per_vec..].as_ptr() as *const v128) };
                                let v2: v128 = unsafe { v128_load(src[2 * #elems_per_vec..].as_ptr() as *const v128) };
                                let v3: v128 = unsafe { v128_load(src[3 * #elems_per_vec..].as_ptr() as *const v128) };

                                // InterleaveLowerLanes(v0, v2) and InterleaveLowerLanes(v1, v3)
                                let v02_lower = #shuffle_fn::<#lower_indices>(v0, v2);
                                let v13_lower = #shuffle_fn::<#lower_indices>(v1, v3);

                                // InterleaveUpperLanes(v0, v2) and InterleaveUpperLanes(v1, v3)
                                let v02_upper = #shuffle_fn::<#upper_indices>(v0, v2);
                                let v13_upper = #shuffle_fn::<#upper_indices>(v1, v3);

                                // Interleave lower and upper to get final result
                                let out0 = #shuffle_fn::<#lower_indices>(v02_lower, v13_lower);
                                let out1 = #shuffle_fn::<#upper_indices>(v02_lower, v13_lower);
                                let out2 = #shuffle_fn::<#lower_indices>(v02_upper, v13_upper);
                                let out3 = #shuffle_fn::<#upper_indices>(v02_upper, v13_upper);

                                #combine_code
                        }
                    }
                }
                OpSig::StoreInterleaved(block_size, count) => {
                    assert_eq!(count, 4, "only count of 4 is currently supported");
                    let arg = store_interleaved_arg_ty(block_size, count, vec_ty);
                    let elems_per_vec = block_size as usize / vec_ty.scalar_bits;

                    let (lower_indices, upper_indices, shuffle_fn) = match vec_ty.scalar_bits {
                        8 => (
                            quote! { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 },
                            quote! { 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 },
                            quote! { u8x16_shuffle },
                        ),
                        16 => (
                            quote! { 0, 8, 1, 9, 2, 10, 3, 11 },
                            quote! { 4, 12, 5, 13, 6, 14, 7, 15 },
                            quote! { u16x8_shuffle },
                        ),
                        32 => (
                            quote! { 0, 4, 1, 5 },
                            quote! { 2, 6, 3, 7 },
                            quote! { u32x4_shuffle },
                        ),
                        _ => panic!("unsupported scalar_bits"),
                    };

                    let split_method_name =
                        |scalar: ScalarType, scalar_bits: usize, lane_count: usize| -> Ident {
                            let scalar = match scalar {
                                ScalarType::Float => 'f',
                                ScalarType::Unsigned => 'u',
                                _ => unimplemented!(),
                            };
                            format_ident!("split_{scalar}{scalar_bits}x{lane_count}")
                        };

                    let split_method_2x =
                        split_method_name(vec_ty.scalar, vec_ty.scalar_bits, elems_per_vec * 4);
                    let split_method =
                        split_method_name(vec_ty.scalar, vec_ty.scalar_bits, elems_per_vec * 2);

                    let split_code = quote! {
                        let (lower, upper) = self.#split_method_2x(a);
                        let (v0_vec, v1_vec) = self.#split_method(lower);
                        let (v2_vec, v3_vec) = self.#split_method(upper);

                        let v0: v128 = v0_vec.into();
                        let v1: v128 = v1_vec.into();
                        let v2: v128 = v2_vec.into();
                        let v3: v128 = v3_vec.into();
                    };

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, #arg) -> #ret_ty {
                            #split_code

                            // InterleaveLowerLanes(v0, v2) and InterleaveLowerLanes(v1, v3)
                            let v02_lower = #shuffle_fn::<#lower_indices>(v0, v2);
                            let v13_lower = #shuffle_fn::<#lower_indices>(v1, v3);

                            // InterleaveUpperLanes(v0, v2) and InterleaveUpperLanes(v1, v3)
                            let v02_upper = #shuffle_fn::<#upper_indices>(v0, v2);
                            let v13_upper = #shuffle_fn::<#upper_indices>(v1, v3);

                            // Interleave lower and upper to get final result
                            let out0 = #shuffle_fn::<#lower_indices>(v02_lower, v13_lower);
                            let out1 = #shuffle_fn::<#upper_indices>(v02_lower, v13_lower);
                            let out2 = #shuffle_fn::<#lower_indices>(v02_upper, v13_upper);
                            let out3 = #shuffle_fn::<#upper_indices>(v02_upper, v13_upper);

                            unsafe {
                                v128_store(dest[0 * #elems_per_vec..].as_mut_ptr() as *mut v128, out0);
                                v128_store(dest[1 * #elems_per_vec..].as_mut_ptr() as *mut v128, out1);
                                v128_store(dest[2 * #elems_per_vec..].as_mut_ptr() as *mut v128, out2);
                                v128_store(dest[3 * #elems_per_vec..].as_mut_ptr() as *mut v128, out3);
                            }
                        }
                    }
                }
            };

            methods.push(m);
        }
    }

    quote! {
        impl Simd for #level_tok {
            type f32s = f32x4<Self>;
            type u8s = u8x16<Self>;
            type i8s = i8x16<Self>;
            type u16s = u16x8<Self>;
            type i16s = i16x8<Self>;
            type u32s = u32x4<Self>;
            type i32s = i32x4<Self>;
            type mask8s = mask8x16<Self>;
            type mask16s = mask16x8<Self>;
            type mask32s = mask32x4<Self>;

            #[inline(always)]
            fn level(self) -> Level {
                Level::#level_tok(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                #[inline]
                // unsafe not needed here with tf11, but can be justified
                unsafe fn vectorize_simd128<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                unsafe { vectorize_simd128(f) }
            }

            #( #methods )*
        }
    }
}

pub fn mk_wasm128_impl(level: Level) -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl(level);
    let ty_impl = mk_type_impl();
    let level_tok = level.token();

    quote! {
        use core::arch::wasm32::*;

        use crate::{seal::Seal, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "wasm128" level.
        #[derive(Clone, Copy, Debug)]
        pub struct #level_tok {
            _private: (),
        }

        impl #level_tok {
            #[inline]
            pub fn new_unchecked() -> Self {
                Self { _private: () }
            }
        }

        impl Seal for #level_tok {}

        #simd_impl

        #ty_impl
    }
}

fn mk_type_impl() -> TokenStream {
    let mut result = vec![];
    for ty in SIMD_TYPES {
        if ty.n_bits() != 128 {
            continue;
        }
        let simd = ty.rust();
        result.push(quote! {
            impl<S: Simd> SimdFrom<v128, S> for #simd<S> {
                #[inline(always)]
                fn simd_from(arch: v128, simd: S) -> Self {
                    Self {
                        val: unsafe { core::mem::transmute(arch) },
                        simd
                    }
                }
            }
            impl<S: Simd> From<#simd<S>> for v128 {
                #[inline(always)]
                fn from(value: #simd<S>) -> Self {
                    unsafe { core::mem::transmute(value.val) }
                }
            }
        })
    }
    quote! {
        #( #result )*
    }
}
