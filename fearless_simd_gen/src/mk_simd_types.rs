// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{
    ops::{CORE_OPS, OpSig, ops_for_type},
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
///
/// This may go away, as possibly all methods will be subsumed by the vec_impl.
fn simd_impl(ty: &VecType) -> TokenStream {
    let name = ty.rust();
    let ty_name = ty.rust_name();
    let mut methods = vec![];
    for (method, sig) in ops_for_type(ty) {
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
            OpSig::Zip => continue,
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
        });
    }
    let vec_impl = simd_vec_impl(ty);
    quote! {
        impl<S: Simd> #name<S> {
            #( #methods )*
        }
        #vec_impl
    }
}

fn simd_vec_impl(ty: &VecType) -> TokenStream {
    let name = ty.rust();
    let ty_name = ty.rust_name();
    let scalar = ty.scalar.rust(ty.scalar_bits);
    let len = Literal::usize_unsuffixed(ty.len);
    let vec_trait = match ty.scalar {
        ScalarType::Float => "SimdFloat",
        ScalarType::Unsigned | ScalarType::Int => "SimdInt",
        ScalarType::Mask => "SimdMask",
    };
    let zero = match ty.scalar {
        ScalarType::Float => quote! { 0.0 },
        _ => quote! { 0 },
    };
    let vec_trait_id = Ident::new(vec_trait, Span::call_site());
    let splat = Ident::new(&format!("splat_{}", ty.rust_name()), Span::call_site());
    let mut methods = vec![];
    for (method, sig) in ops_for_type(ty) {
        if CORE_OPS.contains(&method) {
            continue;
        }
        let method_name = Ident::new(method, Span::call_site());
        let trait_method = Ident::new(&format!("{method}_{ty_name}"), Span::call_site());
        let args = match sig {
            OpSig::Splat => continue,
            OpSig::Unary => quote! { self },
            OpSig::Binary | OpSig::Compare | OpSig::Zip => {
                quote! { self, rhs: impl SimdInto<Self, S> }
            }
            // select is currently done by trait, but maybe we'll implement for
            // masks.
            OpSig::Select => continue,
            OpSig::Split | OpSig::Combine => continue,
        };
        let ret_ty = match sig {
            OpSig::Compare => {
                let mask = ty.mask_ty().rust();
                quote! { #mask<S> }
            }
            OpSig::Combine => {
                let double = VecType::new(ty.scalar, ty.scalar_bits, ty.len * 2).rust();
                quote! { #double<S> }
            }
            OpSig::Zip => quote! { (Self, Self) },
            _ => quote! { #name<S> },
        };
        let call_args = match sig {
            OpSig::Unary => quote! { self },
            OpSig::Binary | OpSig::Compare | OpSig::Combine | OpSig::Zip => {
                quote! { self, rhs.simd_into(self.simd) }
            }
            _ => quote! { todo!() },
        };
        methods.push(quote! {
            #[inline(always)]
            fn #method_name(#args) -> #ret_ty {
                self.simd.#trait_method(#call_args)
            }
        });
    }
    let mask_ty = ty.mask_ty().rust();
    quote! {
        impl<S: Simd> crate::SimdBase<#scalar, S> for #name<S> {
            const N: usize = #len;
            type Mask = #mask_ty<S>;

            #[inline(always)]
            fn as_slice(&self) -> &[#scalar] {
                &self.val
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [#scalar] {
                &mut self.val
            }

            #[inline(always)]
            fn from_slice(simd: S, slice: &[#scalar]) -> Self {
                let mut val = [#zero; #len];
                val.copy_from_slice(slice);
                Self { val, simd }
            }

            #[inline(always)]
            fn splat(simd: S, val: #scalar) -> Self {
                simd.#splat(val)
            }
        }
        impl<S: Simd> crate::#vec_trait_id<#scalar, S> for #name<S> {
            #( #methods )*
        }
    }
}
