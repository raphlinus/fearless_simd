// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::Ident;

use crate::ops::load_interleaved_arg_ty;
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
            let b1 = vec_ty.n_bits() > 128 && !matches!(method, "split" | "narrow");
            let b2 = !matches!(method, "load_interleaved_128");

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
                    let expr = if matches!(method, "fract" | "trunc") {
                        quote! {todo!() }
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
                    // let args = [quote! { a.into() }, quote! { b.into() }];
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            /// TODO: copysign
                            todo!()
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
                    let args = [
                        quote! { a.into() },
                        quote! { b.into() },
                        quote! { c.into() },
                    ];

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
                        let expr = Wasm.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                                // TODO: OpSig::Ternary
                                todo!()
                            }
                        }
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
                            todo!()
                        }
                    }
                }
                OpSig::Combine => generic_combine(vec_ty),
                OpSig::Split => generic_split(vec_ty),
                OpSig::Zip(_) => {
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            todo!()
                        }
                    }
                }
                OpSig::Shift => {
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, shift: u32) -> #ret_ty {
                            todo!()
                        }
                    }
                }
                OpSig::Cvt(_, _) | OpSig::Reinterpret(_, _) | OpSig::WidenNarrow(_) => {
                    // let to_ty = &VecType::new(scalar, scalar_bits, vec_ty.len);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            todo!()
                        }
                    }
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

                    let combine_method_name = |scalar_bits: usize, lane_count: usize| -> Ident {
                        format_ident!("combine_u{}x{}", scalar_bits, lane_count)
                    };

                    let combine_method = combine_method_name(vec_ty.scalar_bits, elems_per_vec);
                    let combine_method_2x =
                        combine_method_name(vec_ty.scalar_bits, elems_per_vec * 2);

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
