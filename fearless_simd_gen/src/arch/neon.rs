// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub struct Neon;

fn translate_op(op: &str) -> Option<&'static str> {
    Some(match op {
        "abs" => "vabs",
        "neg" => "vneg",
        "floor" => "vrndm",
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
        "max" => "vmax",
        "min" => "vmin",
        "shr" => "vshl",
        "max_precise" => "vmaxnm",
        "min_precise" => "vminnm",
        "madd" => "vfma",
        "msub" => "vfms",
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
        let name = if ty.n_bits() == 256 {
            format!("{}{}x{}x2_t", scalar, ty.scalar_bits, ty.len / 2)
        } else if ty.n_bits() == 512 {
            format!("{}{}x{}x4_t", scalar, ty.scalar_bits, ty.len / 4)
        } else {
            format!("{}{}x{}_t", scalar, ty.scalar_bits, ty.len)
        };
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
            "fract" => {
                let to = VecType::new(ScalarType::Int, ty.scalar_bits, ty.len);
                let c1 = cvt_intrinsic("vcvt", &to, ty);
                let c2 = cvt_intrinsic("vcvt", ty, &to);
                let sub = simple_intrinsic("vsub", ty);
                quote! {
                    let c1 = #c1(a.into());
                    let c2 = #c2(c1);

                    #sub(a.into(), c2)
                }
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

pub fn cvt_intrinsic(name: &str, to_ty: &VecType, from_ty: &VecType) -> Ident {
    let (opt_q, from_scalar_c, from_size) = neon_array_type(from_ty);
    let (_opt_q, to_scalar_c, to_size) = neon_array_type(to_ty);
    Ident::new(
        &format!("{name}{opt_q}_{to_scalar_c}{to_size}_{from_scalar_c}{from_size}"),
        Span::call_site(),
    )
}
