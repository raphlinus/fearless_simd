// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub struct Wasm;

fn translate_op(op: &str) -> Option<&'static str> {
    Some(match op {
        "abs" => "abs",
        "neg" => "neg",
        "floor" => "floor",
        "sqrt" => "sqrt",
        "add" => "add",
        "sub" => "sub",
        "mul" => "mul",
        "div" => "div",
        "simd_eq" => "eq",
        "simd_lt" => "lt",
        "simd_le" => "le",
        "simd_ge" => "ge",
        "simd_gt" => "gt",
        "not" => "not",
        "and" => "and",
        "or" => "or",
        "xor" => "xor",
        "max" => "max",
        "min" => "min",
        "max_precise" => "pmax",
        "min_precise" => "pmin",
        "splat" => "splat",
        // TODO: Only target-feature "relaxed-simd" has "relaxed_madd".
        _ => return None,
    })
}

fn simple_intrinsic(name: &str, ty: &VecType) -> TokenStream {
    let ty_prefix = Wasm.arch_ty(ty);
    let ident = Ident::new(name, Span::call_site());
    let combined_ident = Ident::new(
        &format!("{}_{}", ty_prefix.to_string(), ident.to_string()),
        Span::call_site(),
    );
    quote! { #combined_ident }
}

fn v128_intrinsic(name: &str) -> TokenStream {
    let ty_prefix = Ident::new("v128", Span::call_site());
    let ident = Ident::new(name, Span::call_site());
    let combined_ident = Ident::new(
        &format!("{}_{}", ty_prefix.to_string(), ident.to_string()),
        Span::call_site(),
    );
    quote! { #combined_ident }
}

impl Arch for Wasm {
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        let scalar = match ty.scalar {
            ScalarType::Float => "f",
            ScalarType::Unsigned => "u",
            ScalarType::Int | ScalarType::Mask => "i",
        };
        let name = format!("{}{}x{}", scalar, ty.scalar_bits, ty.len);
        let ident = Ident::new(&name, Span::call_site());
        quote! { #ident }
    }

    // expects args and return value in arch dialect
    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
        if let Some(translated) = translate_op(op) {
            let intrinsic = match translated {
                "not" => v128_intrinsic(translated),
                "and" => v128_intrinsic(translated),
                "or" => v128_intrinsic(translated),
                "xor" => v128_intrinsic(translated),
                _ => simple_intrinsic(translated, ty),
            };

            quote! { #intrinsic ( #( #args ),* ) }
        } else {
            match op {
                // Add any special case operations here if needed
                _ => unimplemented!("missing {op}"),
            }
        }
    }
}
