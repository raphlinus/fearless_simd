// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{
    arch::{Arch, WasmSimd128},
    generic::{generic_combine, generic_op, generic_split},
    ops::{OpSig, ops_for_type},
    types::{SIMD_TYPES, ScalarType, VecType, type_imports},
};

pub fn mk_wasm_simd128_impl() -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl();
    let ty_impl = mk_type_impl();

    quote! {
        use core::arch::wasm32::*;

        use crate::{seal::Seal, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for WASM SIMD128.
        #[derive(Clone, Copy, Debug)]
        pub struct WasmSimd128 {
            // No platform-specific handle needed for WASM
        }

        impl WasmSimd128 {
            #[inline]
            pub unsafe fn new_unchecked() -> Self {
                WasmSimd128 {}
            }
        }

        impl Seal for WasmSimd128 {}

        #simd_impl

        #ty_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let mut methods = vec![];

    for vec_ty in SIMD_TYPES {
        // Wasm simd only explicitly supports 128 bit intrinsics. We could create a synthetic 256
        // bit target (that duplicates operations on low and high lanes. This should also be easily
        // fused by v8).
        if vec_ty.n_bits() != 128 {
            // TODO: Is continue ok here?
            continue;
        }

        let scalar_bits = vec_ty.scalar_bits;
        let ty_name = vec_ty.rust_name();
        let ty = vec_ty.rust();

        for (method, sig) in ops_for_type(vec_ty) {
            let method_name = format!("{method}_{ty_name}");
            let method_ident = Ident::new(&method_name, Span::call_site());

            let method = match sig {
                OpSig::Splat => {
                    let scalar = vec_ty.scalar.rust(scalar_bits);
                    let expr = WasmSimd128.expr(method, vec_ty, &[quote! { val }]);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, val: #scalar) -> #ty<Self> {
                            #expr.simd_into(self)
                        }
                    }
                }
                OpSig::Unary => {
                    let args = [quote! { a.into() }];
                    let expr = WasmSimd128.expr(method, vec_ty, &args);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ty<Self> {
                            #expr.simd_into(self)
                        }
                    }
                }
                OpSig::Binary => {
                    let args = [quote! { a.into() }, quote! { b.into() }];

                    if method == "mul"
                        && (vec_ty
                            == (&VecType {
                                scalar: ScalarType::Unsigned,
                                scalar_bits: 8,
                                len: 16,
                            })
                            || vec_ty
                                == (&VecType {
                                    scalar: ScalarType::Int,
                                    scalar_bits: 8,
                                    len: 16,
                                }))
                    {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ty<Self> {
                                unsafe {
                                    // WASM doesn't have `i8x16_mul` or `u8x16_mul`.
                                    todo!();
                                }
                            }
                        }
                    } else if method == "copysign" {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ty<Self> {
                                // Need to implement copysign.
                                todo!();
                            }
                        }
                    } else {
                        let expr = WasmSimd128.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ty<Self> {
                                #expr.simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Compare => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    let expr = WasmSimd128.expr(method, vec_ty, &args);
                    let ret_ty = vec_ty.mask_ty().rust();
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty<Self> {
                            #expr.simd_into(self)
                        }
                    }
                }
                OpSig::Select => {
                    let mask_ty = vec_ty.mask_ty().rust();
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, mask: #mask_ty<Self>, if_true: #ty<Self>, if_false: #ty<Self>) -> #ty<Self> {
                            v128_bitselect(if_true.into(), if_false.into(), mask.into()).simd_into(self)
                        }
                    }
                }
                OpSig::Combine => generic_combine(vec_ty),
                OpSig::Split => generic_split(vec_ty),
                OpSig::Zip => {
                    // Implement zip/unzip using WASM shuffle instructions
                    match (vec_ty.scalar_bits, vec_ty.len) {
                        (8, 16) => {
                            // i8x16, u8x16, mask8x16
                            let shuffle_lo = if method == "zip" {
                                quote! { i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23> }
                            } else {
                                quote! { i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30> }
                            };
                            let shuffle_hi = if method == "zip" {
                                quote! { i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31> }
                            } else {
                                quote! { i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31> }
                            };
                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> (#ty<Self>, #ty<Self>) {
                                    let x = a.into();
                                    let y = b.into();
                                    (
                                        #shuffle_lo(x, y).simd_into(self),
                                        #shuffle_hi(x, y).simd_into(self),
                                    )
                                }
                            }
                        }
                        (16, 8) => {
                            // i16x8, u16x8, mask16x8
                            let shuffle_lo = if method == "zip" {
                                quote! { i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11> }
                            } else {
                                quote! { i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14> }
                            };
                            let shuffle_hi = if method == "zip" {
                                quote! { i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15> }
                            } else {
                                quote! { i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15> }
                            };
                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> (#ty<Self>, #ty<Self>) {
                                    let x = a.into();
                                    let y = b.into();
                                    (
                                        #shuffle_lo(x, y).simd_into(self),
                                        #shuffle_hi(x, y).simd_into(self),
                                    )
                                }
                            }
                        }
                        (32, 4) => {
                            // f32x4, i32x4, u32x4, mask32x4
                            let shuffle_lo = if method == "zip" {
                                quote! { i32x4_shuffle::<0, 4, 1, 5> }
                            } else {
                                quote! { i32x4_shuffle::<0, 2, 4, 6> }
                            };
                            let shuffle_hi = if method == "zip" {
                                quote! { i32x4_shuffle::<2, 6, 3, 7> }
                            } else {
                                quote! { i32x4_shuffle::<1, 3, 5, 7> }
                            };
                            quote! {
                                #[inline(always)]
                                fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> (#ty<Self>, #ty<Self>) {
                                    let x = a.into();
                                    let y = b.into();
                                    (
                                        #shuffle_lo(x, y).simd_into(self),
                                        #shuffle_hi(x, y).simd_into(self),
                                    )
                                }
                            }
                        }

                        _ => {
                            // Use generic implementation for other types (e.g., i64x2)
                            continue;
                        }
                    }
                }
            };
            methods.push(method);
        }
    }

    for vec_ty in SIMD_TYPES {
        if vec_ty.n_bits() != 256 {
            continue;
        }

        for (method, sig) in ops_for_type(vec_ty) {
            if method == "split" {
                methods.push(generic_split(vec_ty));
            } else if method != "combine" {
                methods.push(generic_op(method, sig, vec_ty));
            }
        }
    }

    quote! {
        impl Simd for WasmSimd128 {
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
                Level::WasmSimd128(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                #[target_feature(enable = "simd128")]
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
