// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{
    ops::{CORE_OPS, FLOAT_OPS, INT_OPS, MASK_OPS, OpSig, ops_for_type},
    types::{SIMD_TYPES, VecType, type_imports},
};

pub fn mk_simd_trait() -> TokenStream {
    let imports = type_imports();
    let mut methods = vec![];
    // Float methods
    for vec_ty in SIMD_TYPES {
        let ty_name = vec_ty.rust_name();
        let ty = vec_ty.rust();
        for (method, sig) in &ops_for_type(&vec_ty) {
            let method_name = format!("{method}_{ty_name}");
            let method = Ident::new(&method_name, Span::call_site());
            let args = match sig {
                OpSig::Splat => {
                    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                    quote! { self, val: #scalar }
                }
                OpSig::Unary => quote! { self, a: #ty<Self> },
                OpSig::Binary | OpSig::Compare | OpSig::Combine => {
                    quote! { self, a: #ty<Self>, b: #ty<Self> }
                }
                OpSig::Select => {
                    let mask_ty = vec_ty.mask_ty().rust();
                    quote! { self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self> }
                }
                OpSig::Split => {
                    let ret_ty =
                        VecType::new(vec_ty.scalar, vec_ty.scalar_bits, vec_ty.len / 2).rust();
                    methods.extend(quote! {
                        fn #method(self, a: #ty<Self>) -> (#ret_ty<Self>, #ret_ty<Self>);
                    });
                    continue;
                }
                OpSig::Zip => {
                    methods.extend(quote! {
                        fn #method(self, a: #ty<Self>, b: #ty<Self>) -> (#ty<Self>, #ty<Self>);
                    });
                    continue;
                }
            };
            let ret_ty = match sig {
                OpSig::Compare => vec_ty.mask_ty().rust(),
                OpSig::Combine => {
                    VecType::new(vec_ty.scalar, vec_ty.scalar_bits, vec_ty.len * 2).rust()
                }
                _ => vec_ty.rust(),
            };
            methods.extend(quote! {
                fn #method(#args) -> #ret_ty<Self>;
            });
        }
    }
    let mut code = quote! {
        use crate::{seal::Seal, Level, SimdElement, SimdInto};
        #imports
        /// TODO: docstring
        // TODO: Seal
        pub trait Simd: Sized + Clone + Copy + Send + Sync + Seal + 'static {
            type f32s: SimdFloat<f32, Self>;
            type u8s: SimdInt<u8, Self>;
            type i8s: SimdInt<i8, Self>;
            type u16s: SimdInt<u16, Self>;
            type i16s: SimdInt<i16, Self>;
            type u32s: SimdInt<u32, Self>;
            type i32s: SimdInt<i32, Self>;
            type mask8s: SimdMask<i8, Self>;
            type mask16s: SimdMask<i16, Self>;
            type mask32s: SimdMask<i32, Self>;
            fn level(self) -> Level;

            /// Call function with CPU features enabled.
            ///
            /// For performance, the provided function should be `#[inline(always)]`.
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R;
            #( #methods )*
        }
    };
    code.extend(mk_simd_base());
    code.extend(mk_simd_float());
    code.extend(mk_simd_int());
    code.extend(mk_simd_mask());
    code
}

fn mk_simd_base() -> TokenStream {
    quote! {
        pub trait SimdBase<Element: SimdElement, S: Simd>:
            Copy + Sync + Send + 'static
            + crate::Bytes
        {
            const N: usize;
            type Mask: SimdMask<Element::Mask, S>;
            fn as_slice(&self) -> &[Element];
            fn as_mut_slice(&mut self) -> &mut [Element];
            fn from_slice(simd: S, slice: &[Element]) -> Self;
            fn splat(simd: S, val: Element) -> Self;
        }
    }
}

fn mk_simd_float() -> TokenStream {
    let methods = methods_for_vec_trait(FLOAT_OPS);
    quote! {
        pub trait SimdFloat<Element: SimdElement, S: Simd>: SimdBase<Element, S>
            + core::ops::Neg<Output = Self>
            + core::ops::Add<Output = Self>
            + core::ops::Add<Element, Output = Self>
            + core::ops::Sub<Output = Self>
            + core::ops::Sub<Element, Output = Self>
            + core::ops::Mul<Output = Self>
            + core::ops::Mul<Element, Output = Self>
            + core::ops::Div<Output = Self>
            + core::ops::Div<Element, Output = Self>
        {
            #( #methods )*
        }
    }
}

fn mk_simd_int() -> TokenStream {
    let methods = methods_for_vec_trait(INT_OPS);
    quote! {
        pub trait SimdInt<Element: SimdElement, S: Simd>: SimdBase<Element, S>
            + core::ops::Add<Output = Self>
            + core::ops::Add<Element, Output = Self>
            + core::ops::Sub<Output = Self>
            + core::ops::Sub<Element, Output = Self>
            + core::ops::Mul<Output = Self>
            + core::ops::Mul<Element, Output = Self>
            + core::ops::BitAnd<Output = Self>
            + core::ops::BitAnd<Element, Output = Self>
            + core::ops::BitOr<Output = Self>
            + core::ops::BitOr<Element, Output = Self>
            + core::ops::BitXor<Output = Self>
            + core::ops::BitXor<Element, Output = Self>
        {
            #( #methods )*
        }
    }
}

fn mk_simd_mask() -> TokenStream {
    let methods = methods_for_vec_trait(MASK_OPS);
    quote! {
        pub trait SimdMask<Element: SimdElement, S: Simd>: SimdBase<Element, S>
            + core::ops::Not<Output = Self>
            + core::ops::BitAnd<Output = Self>
            + core::ops::BitOr<Output = Self>
            + core::ops::BitXor<Output = Self>
        {
            #( #methods )*
        }
    }
}

fn methods_for_vec_trait(ops: &[(&str, OpSig)]) -> Vec<TokenStream> {
    let mut methods = vec![];
    for (method, sig) in ops {
        if CORE_OPS.contains(method) || *method == "splat" {
            continue;
        }
        let method_name = Ident::new(method, Span::call_site());
        let args = match sig {
            OpSig::Splat => continue,
            OpSig::Unary => quote! { self },
            OpSig::Binary | OpSig::Compare | OpSig::Zip => {
                quote! { self, rhs: impl SimdInto<Self, S> }
            }
            // select is currently done by trait, but maybe we'll implement for
            // masks.
            OpSig::Select => continue,
            // These signatures involve types not in the Simd trait
            OpSig::Split | OpSig::Combine => continue,
        };
        let ret_ty = match sig {
            OpSig::Compare => quote! { Self::Mask },
            OpSig::Zip => quote! { (Self, Self) },
            _ => quote! { Self },
        };
        methods.push(quote! {
            fn #method_name(#args) -> #ret_ty;
        })
    }
    methods
}
