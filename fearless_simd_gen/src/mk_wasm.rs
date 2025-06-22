// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

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
            if vec_ty.n_bits() > 128 && method != "split" {
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
                OpSig::Unary if method == "not" => {
                    let args = [quote! { a.into() }];
                    let expr = Wasm.expr(method, vec_ty, &args);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            /// TODO: If v128 is used, we need to reinterpret it.
                            todo!()
                        }
                    }
                }
                OpSig::Unary => {
                    let args = [quote! { a.into() }];
                    let expr = if method == "fract" {
                        quote! {todo!() }
                    }   else {
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
                OpSig::Binary if method == "xor" || method == "or" || method == "and" => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    let expr = Wasm.expr(method, vec_ty, &args);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            /// TODO: If v128 is used we need to reinterpret it accurately...
                            todo!()
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
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                // TODO: WASM doesn't have `i8x16_mul` or `u8x16_mul`.
                                todo!()
                            }
                        }
                    } else {
                        let expr = Wasm.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                #expr.simd_into(self)
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
                            }   else {
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
                },
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
