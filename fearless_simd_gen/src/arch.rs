// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::types::{ScalarType, VecType};

pub trait Arch {
    fn arch_ty(&self, ty: &VecType) -> TokenStream;
    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream;
}

pub struct Neon;

fn translate_op(op: &str) -> Option<&'static str> {
    Some(match op {
        "abs" => "vabs",
        "neg" => "vneg",
        "sqrt" => "vsqrt",
        "add" => "vadd",
        "sub" => "vsub",
        "mul" => "vmul",
        "div" => "vdiv",
        "simd_eq" => "vceq",
        "simd_lt" => "vclt",
        "simd_le" => "vcle",
        "simd_ge" => "vcge",
        "simd_gt" => "vcgt",
        "not" => "vmvn",
        "and" => "vand",
        "or" => "vorr",
        "xor" => "veor",
        _ => return None,
    })
}

impl Arch for Neon {
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        let scalar = match ty.scalar {
            ScalarType::Float => "float",
            ScalarType::Unsigned => "uint",
            ScalarType::Int | ScalarType::Mask => "int",
        };
        let name = format!("{}{}x{}_t", scalar, ty.scalar_bits, ty.len);
        let ident = Ident::new(&name, Span::call_site());
        quote! { #ident }
    }

    // expects args and return value in arch dialect
    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
        if let Some(xlat) = translate_op(op) {
            let intrinsic = simple_intrinsic(xlat, ty);
            return quote! { #intrinsic ( #( #args ),* ) };
        }
        match op {
            "splat" => {
                let intrinsic = split_intrinsic("vdup", "n", ty);
                quote! { #intrinsic ( #( #args ),* ) }
            }
            _ => unimplemented!("missing {op}"),
        }
    }
}

fn neon_array_type(ty: &VecType) -> (&'static str, &'static str, usize) {
    let scalar_c = match ty.scalar {
        ScalarType::Float => "f",
        ScalarType::Unsigned => "u",
        ScalarType::Int | ScalarType::Mask => "s",
    };
    (opt_q(ty), scalar_c, ty.scalar_bits)
}

pub fn opt_q(ty: &VecType) -> &'static str {
    match ty.n_bits() {
        64 => "",
        128 => "q",
        _ => panic!("unsupported simd width"),
    }
}

pub fn simple_intrinsic(name: &str, ty: &VecType) -> Ident {
    let (opt_q, scalar_c, size) = neon_array_type(ty);
    Ident::new(
        &format!("{name}{opt_q}_{scalar_c}{size}"),
        Span::call_site(),
    )
}

pub fn split_intrinsic(name: &str, name2: &str, ty: &VecType) -> Ident {
    let (opt_q, scalar_c, size) = neon_array_type(ty);
    Ident::new(
        &format!("{name}{opt_q}_{name2}_{scalar_c}{size}"),
        Span::call_site(),
    )
}

/// WASM SIMD 128

pub struct WasmSimd128;

impl Arch for WasmSimd128 {
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        // WASM SIMD128 uses v128 for all types
        if ty.n_bits() == 128 {
            quote! { v128 }
        } else {
            panic!("WASM SIMD128 only supports 128-bit vectors, got {}-bit", ty.n_bits());
        }
    }

    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
        let intrinsic = wasm_intrinsic(op, ty);
        quote! { #intrinsic ( #( #args ),* ) }
    }
}

fn wasm_intrinsic(op: &str, ty: &VecType) -> Ident {
    let type_suffix = match (ty.scalar, ty.scalar_bits, ty.len) {
        (ScalarType::Float, 32, 4) => "f32x4",
        (ScalarType::Int, 8, 16) => "i8x16",
        (ScalarType::Int, 16, 8) => "i16x8",
        (ScalarType::Int, 32, 4) => "i32x4",
        (ScalarType::Int, 64, 2) => "i64x2",
        (ScalarType::Unsigned, 8, 16) => "u8x16",
        (ScalarType::Unsigned, 16, 8) => "u16x8",
        (ScalarType::Unsigned, 32, 4) => "u32x4",
        (ScalarType::Unsigned, 64, 2) => "u64x2",
        (ScalarType::Mask, _, _) => {
            // Masks use the same operations as their corresponding integer types
            match ty.scalar_bits {
                8 => "i8x16",
                16 => "i16x8",
                32 => "i32x4",
                64 => "i64x2",
                _ => panic!("Unsupported mask bit width: {}", ty.scalar_bits),
            }
        }
        _ => panic!("Unsupported type for WASM SIMD: {:?}", ty),
    };
    
    let op_name = match op {
        "splat" => format!("{}_splat", type_suffix),
        "abs" => {
            match ty.scalar {
                ScalarType::Float => format!("{}_abs", type_suffix),
                ScalarType::Int => format!("{}_abs", type_suffix),
                _ => panic!("abs not supported for {:?}", ty.scalar),
            }
        }
        "neg" => {
            match ty.scalar {
                ScalarType::Float => format!("{}_neg", type_suffix),
                ScalarType::Int => format!("{}_neg", type_suffix),
                _ => panic!("neg not supported for {:?}", ty.scalar),
            }
        }
        "sqrt" => {
            if ty.scalar == ScalarType::Float {
                format!("{}_sqrt", type_suffix)
            } else {
                panic!("sqrt only supported for float types")
            }
        }
        "add" => format!("{}_add", type_suffix),
        "sub" => format!("{}_sub", type_suffix),
        "mul" => format!("{}_mul", type_suffix),
        "div" => {
            if ty.scalar == ScalarType::Float {
                format!("{}_div", type_suffix)
            } else {
                panic!("div only supported for float types in WASM SIMD")
            }
        }
        "simd_eq" => format!("{}_eq", type_suffix),
        "simd_lt" => {
            match ty.scalar {
                ScalarType::Float | ScalarType::Int => format!("{}_lt", type_suffix),
                ScalarType::Unsigned => format!("{}_lt", type_suffix),
                _ => panic!("lt comparison not supported for {:?}", ty.scalar),
            }
        }
        "simd_le" => {
            match ty.scalar {
                ScalarType::Float | ScalarType::Int => format!("{}_le", type_suffix),
                ScalarType::Unsigned => format!("{}_le", type_suffix),
                _ => panic!("le comparison not supported for {:?}", ty.scalar),
            }
        }
        "simd_gt" => {
            match ty.scalar {
                ScalarType::Float | ScalarType::Int => format!("{}_gt", type_suffix),
                ScalarType::Unsigned => format!("{}_gt", type_suffix),
                _ => panic!("gt comparison not supported for {:?}", ty.scalar),
            }
        }
        "simd_ge" => {
            match ty.scalar {
                ScalarType::Float | ScalarType::Int => format!("{}_ge", type_suffix),
                ScalarType::Unsigned => format!("{}_ge", type_suffix),
                _ => panic!("ge comparison not supported for {:?}", ty.scalar),
            }
        }
        "and" => "v128_and".to_string(),
        "or" => "v128_or".to_string(),
        "xor" => "v128_xor".to_string(),
        "not" => "v128_not".to_string(),
        _ => panic!("Unsupported operation for WASM SIMD: {}", op),
    };
    
    Ident::new(&op_name, Span::call_site())
}