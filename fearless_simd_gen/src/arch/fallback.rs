use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use crate::arch::Arch;
use crate::types::{ScalarType, VecType};

pub(crate) fn translate_op(op: &str) -> Option<&'static str> {
    Some(match op {
        "abs" => "abs",
        "copysign" => "copysign",
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
        "and" => "bitand",
        "or" => "bitor",
        "xor" => "bitxor",
        // TODO: Do we need to polyfill so behavior is consistent with NEON?
        "max" => "max",
        "min" => "min",
        "max_precise" => "max",
        "min_precise" => "min",
        // TODO: Should we really use `multiply_add` here, or just simulate with `+` and `*`.
        "madd" => "multiply_add",
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
        if let Some(translated) = translate_op(op) {
            let intrinsic = simple_intrinsic(translated, ty);
            quote! { #intrinsic ( #( #args ),* ) }
        }   else {
            unimplemented!("missing {op}")
        }
    }
}