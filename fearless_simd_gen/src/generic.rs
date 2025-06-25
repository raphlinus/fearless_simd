// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::ops::reinterpret_ty;
use crate::{
    ops::{OpSig, TyFlavor},
    types::{ScalarType, VecType},
};

/// Implementation of combine based on `copy_from_slice`
pub fn generic_combine(ty: &VecType) -> TokenStream {
    let ty_rust = ty.rust();
    let n = ty.len;
    let n2 = n * 2;
    let result = VecType::new(ty.scalar, ty.scalar_bits, n2).rust();
    let name = Ident::new(&format!("combine_{}", ty.rust_name()), Span::call_site());
    let default = match ty.scalar {
        ScalarType::Float => quote! { 0.0 },
        _ => quote! { 0 },
    };
    quote! {
        #[inline(always)]
        fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>) -> #result<Self> {
            let mut result = [#default; #n2];
            result[0..#n].copy_from_slice(&a.val);
            result[#n..#n2].copy_from_slice(&b.val);
            result.simd_into(self)
        }
    }
}

/// Implementation of split based on `copy_from_slice`
pub fn generic_split(ty: &VecType) -> TokenStream {
    let ty_rust = ty.rust();
    let n = ty.len;
    let nhalf = n / 2;
    let result = VecType::new(ty.scalar, ty.scalar_bits, nhalf).rust();
    let name = Ident::new(&format!("split_{}", ty.rust_name()), Span::call_site());
    let default = match ty.scalar {
        ScalarType::Float => quote! { 0.0 },
        _ => quote! { 0 },
    };
    quote! {
        #[inline(always)]
        fn #name(self, a: #ty_rust<Self>) -> (#result<Self>, #result<Self>) {
            let mut b0 = [#default; #nhalf];
            let mut b1 = [#default; #nhalf];
            b0.copy_from_slice(&a.val[0..#nhalf]);
            b1.copy_from_slice(&a.val[#nhalf..#n]);
            (b0.simd_into(self), b1.simd_into(self))
        }
    }
}

/// Implementation based on split/combine
///
/// Only suitable for lane-wise and block-wise operations
pub fn generic_op(op: &str, sig: OpSig, ty: &VecType) -> TokenStream {
    let ty_rust = ty.rust();
    let name = Ident::new(&format!("{op}_{}", ty.rust_name()), Span::call_site());
    let split = Ident::new(&format!("split_{}", ty.rust_name()), Span::call_site());
    let half = VecType::new(ty.scalar, ty.scalar_bits, ty.len / 2);
    let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
    let do_half = Ident::new(&format!("{op}_{}", half.rust_name()), Span::call_site());
    let ret_ty = sig.ret_ty(ty, TyFlavor::SimdTrait);
    match sig {
        OpSig::Splat => {
            let scalar = ty.scalar.rust(ty.scalar_bits);
            quote! {
                #[inline(always)]
                fn #name(self, a: #scalar) -> #ret_ty {
                    let half = self.#do_half(a);
                    self.#combine(half, half)
                }
            }
        }
        OpSig::Unary => {
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Binary => {
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine(self.#do_half(a0, b0), self.#do_half(a1, b1))
                }
            }
        }
        OpSig::Shift => {
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>, b: u32) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0, b), self.#do_half(a1, b))
                }
            }
        }
        OpSig::Ternary => {
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>, c: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    let (c0, c1) = self.#split(c);
                    self.#combine(self.#do_half(a0, b0, c0), self.#do_half(a1, b1, c1))
                }
            }
        }
        OpSig::Compare => {
            let half_mask = VecType::new(ScalarType::Mask, ty.scalar_bits, ty.len / 2);
            let combine_mask = Ident::new(
                &format!("combine_{}", half_mask.rust_name()),
                Span::call_site(),
            );
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine_mask(self.#do_half(a0, b0), self.#do_half(a1, b1))
                }
            }
        }
        OpSig::Select => {
            let mask_ty = VecType::new(ScalarType::Mask, ty.scalar_bits, ty.len);
            let mask = mask_ty.rust();
            let split_mask =
                Ident::new(&format!("split_{}", mask_ty.rust_name()), Span::call_site());
            quote! {
                #[inline(always)]
                fn #name(self, a: #mask<Self>, b: #ty_rust<Self>, c: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split_mask(a);
                    let (b0, b1) = self.#split(b);
                    let (c0, c1) = self.#split(c);
                    self.#combine(self.#do_half(a0, b0, c0), self.#do_half(a1, b1, c1))
                }
            }
        }
        OpSig::Zip(zip1) => {
            let (e1, e2, e3) = if zip1 {
                (
                    quote! {
                        (a0, _)
                    },
                    quote! {
                        (b0, _)
                    },
                    quote! {
                        a0, b0
                    },
                )
            } else {
                (
                    quote! {
                        (_, a1)
                    },
                    quote! {
                        (_, b1)
                    },
                    quote! {
                        a1, b1
                    },
                )
            };

            let zip_low_half =
                Ident::new(&format!("zip_low_{}", half.rust_name()), Span::call_site());
            let zip_high_half =
                Ident::new(&format!("zip_high_{}", half.rust_name()), Span::call_site());

            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>) -> #ret_ty {
                    let #e1 = self.#split(a);
                    let #e2 = self.#split(b);
                    self.#combine(self.#zip_low_half(#e3), self.#zip_high_half(#e3))
                }
            }
        }
        OpSig::Cvt(scalar, scalar_bits) => {
            let half = VecType::new(scalar, scalar_bits, ty.len / 2);
            let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Reinterpret(scalar, scalar_bits) => {
            let mut half = reinterpret_ty(ty, scalar, scalar_bits);
            half.len = half.len / 2;
            let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::WidenNarrow(mut half) => {
            half.len = half.len / 2;
            let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
            quote! {
                #[inline(always)]
                fn #name(self, a: #ty_rust<Self>) -> #ret_ty {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Split => generic_split(ty),
        OpSig::Combine => generic_combine(ty),
        OpSig::LoadInterleaved(_, _) => unimplemented!(),
    }
}
