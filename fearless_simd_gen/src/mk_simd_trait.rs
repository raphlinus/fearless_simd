// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{
    ops::{OpSig, ops_for_type},
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
    quote! {
        use crate::{seal::Seal, Level};
        #imports
        /// TODO: docstring
        // TODO: Seal
        pub trait Simd: Sized + Clone + Copy + Send + Sync + Seal + 'static {
            fn level(self) -> Level;

            /// Call function with CPU features enabled.
            ///
            /// For performance, the provided function should be `#[inline(always)]`.
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R;
            #( #methods )*
        }
    }
}
