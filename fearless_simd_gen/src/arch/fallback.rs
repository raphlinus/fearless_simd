// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub(crate) fn translate_op(op: &str, is_float: bool) -> Option<&'static str> {
    Some(match op {
        "abs" => "abs",
        "copysign" => "copysign",
        "neg" => "neg",
        "floor" => "floor",
        "fract" => "fract",
        "trunc" => "trunc",
        "sqrt" => "sqrt",
        "add" => "add",
        "sub" => if is_float { "sub" } else { "wrapping_sub" },
        "mul" => "mul",
        "div" => "div",
        "simd_eq" => "eq",
        "simd_lt" => "lt",
        "simd_le" => "le",
        "simd_ge" => "ge",
        "simd_gt" => "gt",
        "not" => "not",
        "and" => "bitand",
        "or" => "bitor",
        "xor" => "bitxor",
        "shr" => "shr",
        // TODO: Do we need to polyfill so behavior is consistent with NEON?
        "max" => "max",
        "min" => "min",
        "max_precise" => "max",
        "min_precise" => "min",
        _ => return None,
    })
}

pub fn simple_intrinsic(name: &str, ty: &VecType) -> TokenStream {
    let ty_prefix = Fallback.arch_ty(ty);
    let ident = Ident::new(name, Span::call_site());

    quote! {#ty_prefix::#ident}
}

pub struct Fallback;

impl Arch for Fallback {
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        let scalar = match ty.scalar {
            ScalarType::Float => "f",
            ScalarType::Unsigned => "u",
            ScalarType::Int | ScalarType::Mask => "i",
        };
        let name = format!("{}{}", scalar, ty.scalar_bits);
        let ident = Ident::new(&name, Span::call_site());
        quote! { #ident }
    }

    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
        if let Some(translated) = translate_op(op, ty.scalar == ScalarType::Float) {
            let intrinsic = simple_intrinsic(translated, ty);
            quote! { #intrinsic ( #( #args ),* ) }
        } else {
            match op {
                _ => unimplemented!("missing {op}"),
            }
        }
    }
}
