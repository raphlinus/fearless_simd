// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{
    ops::{OpSig, ops_for_type},
    types::{SIMD_TYPES, ScalarType, VecType},
};

pub fn mk_simd_types() -> TokenStream {
    let mut result = quote! {
        use crate::{Bytes, Select, Simd, SimdFrom, SimdInto};
    };
    for ty in SIMD_TYPES {
        let name = ty.rust();
        let align = ty.n_bits() / 8;
        let align_lit = Literal::usize_unsuffixed(align);
        let len = Literal::usize_unsuffixed(ty.len);
        let rust_scalar = ty.scalar.rust(ty.scalar_bits);
        let select = Ident::new(&format!("select_{}", ty.rust_name()), Span::call_site());
        let bytes = VecType::new(ScalarType::Unsigned, 8, align).rust();
        let mask = ty.mask_ty().rust();
        let scalar_impl = if ty.scalar != ScalarType::Mask {
            let splat = Ident::new(&format!("splat_{}", ty.rust_name()), Span::call_site());
            quote! {
                impl<S: Simd> SimdFrom<#rust_scalar, S> for #name<S> {
                    #[inline(always)]
                    fn simd_from(value: #rust_scalar, simd: S) -> Self {
                        simd.#splat(value)
                    }
                }
            }
        } else {
            // TODO: consider implementing splat for masks; portable_simd does
            quote! {}
        };
        let impl_block = simd_impl(ty);
        result.extend(quote! {
            #[derive(Clone, Copy)]
            #[repr(C, align(#align_lit))]
            pub struct #name<S: Simd> {
                pub val: [#rust_scalar; #len],
                pub simd: S,
            }

            impl<S: Simd> SimdFrom<[#rust_scalar; #len], S> for #name<S> {
                #[inline(always)]
                fn simd_from(val: [#rust_scalar; #len], simd: S) -> Self {
                    Self { val, simd }
                }
            }

            impl<S: Simd> From<#name<S>> for [#rust_scalar; #len] {
                #[inline(always)]
                fn from(value: #name<S>) -> Self {
                    value.val
                }
            }

            impl<S: Simd> std::ops::Deref for #name<S> {
                type Target = [#rust_scalar; #len];
                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    &self.val
                }
            }

            impl<S: Simd> std::ops::DerefMut for #name<S> {
                #[inline(always)]
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.val
                }
            }

            #scalar_impl

            impl<S: Simd> Select<#name<S>> for #mask<S> {
                #[inline(always)]
                fn select(self, if_true: #name<S>, if_false: #name<S>) -> #name<S> {
                    self.simd.#select(self, if_true, if_false)
                }
            }

            impl<S: Simd> Bytes for #name<S> {
                type Bytes = #bytes<S>;

                #[inline(always)]
                fn to_bytes(self) -> Self::Bytes {
                    unsafe {
                        #bytes {
                            val: core::mem::transmute(self.val),
                            simd: self.simd,
                        }
                    }
                }

                #[inline(always)]
                fn from_bytes(value: Self::Bytes) -> Self {
                    unsafe {
                        Self {
                            val: core::mem::transmute(value.val),
                            simd: value.simd,
                        }
                    }
                }
            }

            #impl_block
        });
    }
    result
}

/// Create the impl block for the type
fn simd_impl(ty: &VecType) -> TokenStream {
    let name = ty.rust();
    let ty_name = ty.rust_name();
    let mut methods = vec![];
    for (method, sig) in &ops_for_type(ty) {
        let method_name = Ident::new(method, Span::call_site());
        let trait_method = Ident::new(&format!("{method}_{ty_name}"), Span::call_site());
        let args = match sig {
            OpSig::Splat => continue,
            OpSig::Unary => quote! { self },
            OpSig::Binary | OpSig::Compare | OpSig::Combine => {
                quote! { self, rhs: impl SimdInto<Self, S> }
            }
            // select is currently done by trait, but maybe we'll implement for
            // masks.
            OpSig::Select => continue,
            OpSig::Split => quote! { self },
        };
        let ret_ty = match sig {
            OpSig::Split => continue,
            OpSig::Compare => {
                let mask = ty.mask_ty().rust();
                quote! { #mask<S> }
            }
            OpSig::Combine => {
                let double = VecType::new(ty.scalar, ty.scalar_bits, ty.len * 2).rust();
                quote! { #double<S> }
            }
            _ => quote! { #name<S> },
        };
        let call_args = match sig {
            OpSig::Unary => quote! { self },
            OpSig::Binary | OpSig::Compare | OpSig::Combine => {
                quote! { self, rhs.simd_into(self.simd) }
            }
            _ => quote! { todo!() },
        };
        methods.push(quote! {
            #[inline(always)]
            pub fn #method_name(#args) -> #ret_ty {
                self.simd.#trait_method(#call_args)
            }
        })
    }
    quote! {
        impl<S: Simd> #name<S> {
            #( #methods )*
        }
    }
}
