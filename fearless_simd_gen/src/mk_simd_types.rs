// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{
    ops::{CORE_OPS, OpSig, TyFlavor, ops_for_type},
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
        let simd_from_items =make_list( (0..ty.len).map(|idx| quote! { val[#idx] })
            .collect::<Vec<_>>());
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
                    // Note: Previously, we would just straight up copy `val`. However, at least on 
                    // ARM, this would always lead to it being compiled to a `memset_pattern16`, at least
                    // for scalar f32x4, which significantly slowed down the `render_strips` benchmark.
                    // Assigning each index individually seems to circumvent this quirk.
                    // TODO: Investigate whether this has detrimental effects for other numeric
                    // types.
                    Self { val: #simd_from_items, simd }
                }
            }

            impl<S: Simd> From<#name<S>> for [#rust_scalar; #len] {
                #[inline(always)]
                fn from(value: #name<S>) -> Self {
                    value.val
                }
            }

            impl<S: Simd> core::ops::Deref for #name<S> {
                type Target = [#rust_scalar; #len];
                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    &self.val
                }
            }

            impl<S: Simd> core::ops::DerefMut for #name<S> {
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
    for (method, sig) in ops_for_type(ty, true) {
        let method_name = Ident::new(method, Span::call_site());
        let trait_method = Ident::new(&format!("{method}_{ty_name}"), Span::call_site());
        if matches!(
            sig,
            OpSig::Unary | OpSig::Binary | OpSig::Compare | OpSig::Combine | OpSig::Cvt(_, _)
        ) {
            if let Some(args) = sig.vec_trait_args() {
                let ret_ty = sig.ret_ty(ty, TyFlavor::VecImpl);
                let call_args = match sig {
                    OpSig::Unary | OpSig::Cvt(_, _) => quote! { self },
                    OpSig::Binary | OpSig::Compare | OpSig::Combine => {
                        quote! { self, rhs.simd_into(self.simd) }
                    }
                    OpSig::Ternary => {
                        quote! { self, op1.simd_into(self.simd), op2.simd_into(self.simd) }
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
        }
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
    for (method, sig) in ops_for_type(ty, false) {
        if CORE_OPS.contains(&method) || matches!(sig, OpSig::Combine) {
            continue;
        }
        let method_name = Ident::new(method, Span::call_site());
        let trait_method = Ident::new(&format!("{method}_{ty_name}"), Span::call_site());
        if let Some(args) = sig.vec_trait_args() {
            let ret_ty = sig.ret_ty(ty, TyFlavor::VecImpl);
            let call_args = match sig {
                OpSig::Unary => quote! { self },
                OpSig::Binary | OpSig::Compare | OpSig::Combine | OpSig::Zip => {
                    quote! { self, rhs.simd_into(self.simd) }
                }
                OpSig::Ternary => {
                    quote! { self, op1.simd_into(self.simd), op2.simd_into(self.simd) }
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
    }
    let mask_ty = ty.mask_ty().rust();
    let block_ty = VecType::new(ty.scalar, ty.scalar_bits, 128 / ty.scalar_bits).rust();
    let block_splat_body = match ty.n_bits() {
        64 => quote! {
            block.split().0
        },
        128 => quote! {
            block
        },
        256 => quote! {
            block.combine(block)
        },
        512 => quote! {
            let block2 = block.combine(block);
            block2.combine(block2)
        },
        _ => unreachable!(),
    };
    quote! {
        impl<S: Simd> crate::SimdBase<#scalar, S> for #name<S> {
            const N: usize = #len;
            type Mask = #mask_ty<S>;
            type Block = #block_ty<S>;

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

            #[inline(always)]
            fn block_splat(block: Self::Block) -> Self {
                #block_splat_body
            }

        }
        impl<S: Simd> crate::#vec_trait_id<#scalar, S> for #name<S> {
            #( #methods )*
        }
    }
}

fn make_list(items: Vec<TokenStream>) -> TokenStream {
    quote!([#( #items, )*])
}