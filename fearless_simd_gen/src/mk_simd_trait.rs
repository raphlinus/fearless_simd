// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{
    ops::{CORE_OPS, FLOAT_OPS, INT_OPS, MASK_OPS, OpSig, TyFlavor, ops_for_type},
    types::{SIMD_TYPES, type_imports},
};

pub fn mk_simd_trait() -> TokenStream {
    let imports = type_imports();
    let mut methods = vec![];
    // Float methods
    for vec_ty in SIMD_TYPES {
        let ty_name = vec_ty.rust_name();
        for (method, sig) in &ops_for_type(&vec_ty, true) {
            let method_name = format!("{method}_{ty_name}");
            let method = Ident::new(&method_name, Span::call_site());
            let args = sig.simd_trait_args(vec_ty);
            let ret_ty = sig.ret_ty(vec_ty, TyFlavor::SimdTrait);
            methods.extend(quote! {
                fn #method(#args) -> #ret_ty;
            });
        }
    }
    let mut code = quote! {
        use crate::{seal::Seal, Level, SimdElement, SimdInto};
        #imports
        /// TODO: docstring
        // TODO: Seal
        pub trait Simd: Sized + Clone + Copy + Send + Sync + Seal + 'static {
            type f32s: SimdFloat<f32, Self, Block = f32x4<Self>>;
            type u8s: SimdInt<u8, Self, Block = u8x16<Self>>;
            type i8s: SimdInt<i8, Self, Block = i8x16<Self>>;
            type u16s: SimdInt<u16, Self, Block = u16x8<Self>>;
            type i16s: SimdInt<i16, Self, Block = i16x8<Self>>;
            type u32s: SimdInt<u32, Self, Block = u32x4<Self>>;
            type i32s: SimdInt<i32, Self, Block = i32x4<Self>>;
            type mask8s: SimdMask<i8, Self, Block = mask8x16<Self>>;
            type mask16s: SimdMask<i16, Self, Block = mask16x8<Self>>;
            type mask32s: SimdMask<i32, Self, Block = mask32x4<Self>>;
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
            /// A SIMD vector mask with the same number of elements.
            ///
            /// The mask element is represented as an integer which is
            /// all-0 for `false` and all-1 for `true`. When we get deep
            /// into AVX-512, we need to think about predication masks.
            ///
            /// One possiblity to consider is that the SIMD trait grows
            /// `maskAxB` associated types.
            type Mask: SimdMask<Element::Mask, S>;
            /// A 128 bit SIMD vector of the same scalar type.
            type Block: SimdBase<Element, S>;
            fn as_slice(&self) -> &[Element];
            fn as_mut_slice(&mut self) -> &mut [Element];
            /// Create a SIMD vector from a slice.
            ///
            /// The slice must be the proper width.
            fn from_slice(simd: S, slice: &[Element]) -> Self;
            fn splat(simd: S, val: Element) -> Self;
            fn block_splat(block: Self::Block) -> Self;
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
        if CORE_OPS.contains(method) || matches!(sig, OpSig::Splat | OpSig::Combine) {
            continue;
        }
        let method_name = Ident::new(method, Span::call_site());
        if let Some(args) = sig.vec_trait_args() {
            let ret_ty = match sig {
                OpSig::Compare => quote! { Self::Mask },
                OpSig::Zip => quote! { (Self, Self) },
                _ => quote! { Self },
            };
            methods.push(quote! {
                fn #method_name(#args) -> #ret_ty;
            });
        }
    }
    methods
}
