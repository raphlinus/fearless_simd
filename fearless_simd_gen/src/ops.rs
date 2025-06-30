// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::TokenStream;
use quote::quote;

use crate::types::{ScalarType, VecType};

#[derive(Clone, Copy)]
pub enum OpSig {
    Splat,
    Unary,
    Binary,
    Ternary,
    Compare,
    Select,
    Combine,
    Split,
    Zip(bool),
    Cvt(ScalarType, usize),
    Reinterpret(ScalarType, usize),
    WidenNarrow(VecType),
    // TODO: Make clear that this is right-shift
    Shift,
    // First argument is the base block size (i.e. 128), second argument
    // is how many blocks. For example, `LoadInterleaved(128, 4)` would correspond to the
    // NEON instructions `vld4q_f32`, while `LoadInterleaved(64, 4)` would correspond to
    // `vld4_f32`.
    LoadInterleaved(u16, u16), 
    StoreInterleaved(u16, u16), // TODO: fma
}

pub const FLOAT_OPS: &[(&str, OpSig)] = &[
    ("splat", OpSig::Splat),
    ("abs", OpSig::Unary),
    ("neg", OpSig::Unary),
    ("sqrt", OpSig::Unary),
    ("add", OpSig::Binary),
    ("sub", OpSig::Binary),
    ("mul", OpSig::Binary),
    ("div", OpSig::Binary),
    ("copysign", OpSig::Binary),
    ("simd_eq", OpSig::Compare),
    ("simd_lt", OpSig::Compare),
    ("simd_le", OpSig::Compare),
    ("simd_ge", OpSig::Compare),
    ("simd_gt", OpSig::Compare),
    ("zip_low", OpSig::Zip(true)),
    ("zip_high", OpSig::Zip(false)),
    ("max", OpSig::Binary),
    ("max_precise", OpSig::Binary),
    ("min", OpSig::Binary),
    ("min_precise", OpSig::Binary),
    ("madd", OpSig::Ternary),
    ("msub", OpSig::Ternary),
    ("floor", OpSig::Unary),
    ("fract", OpSig::Unary),
    ("trunc", OpSig::Unary),
    // TODO: simd_ne, but this requires additional implementation work on Neon
    ("select", OpSig::Select),
];

pub const INT_OPS: &[(&str, OpSig)] = &[
    ("splat", OpSig::Splat),
    ("not", OpSig::Unary),
    ("add", OpSig::Binary),
    ("sub", OpSig::Binary),
    ("mul", OpSig::Binary),
    ("and", OpSig::Binary),
    ("or", OpSig::Binary),
    ("xor", OpSig::Binary),
    ("shr", OpSig::Shift),
    ("simd_eq", OpSig::Compare),
    ("simd_lt", OpSig::Compare),
    ("simd_le", OpSig::Compare),
    ("simd_ge", OpSig::Compare),
    ("simd_gt", OpSig::Compare),
    ("zip_low", OpSig::Zip(true)),
    ("zip_high", OpSig::Zip(false)),
    ("select", OpSig::Select),
    ("min", OpSig::Binary),
    ("max", OpSig::Binary),
];

pub const MASK_OPS: &[(&str, OpSig)] = &[
    ("splat", OpSig::Splat),
    ("not", OpSig::Unary),
    ("and", OpSig::Binary),
    ("or", OpSig::Binary),
    ("xor", OpSig::Binary),
    ("select", OpSig::Select),
    ("simd_eq", OpSig::Compare),
];

/// Ops covered by core::ops
pub const CORE_OPS: &[&str] = &[
    "not", "neg", "add", "sub", "mul", "div", "and", "or", "xor", "shr",
];

pub fn ops_for_type(ty: &VecType, cvt: bool) -> Vec<(&str, OpSig)> {
    let base = match ty.scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    let mut ops = base.to_vec();
    if ty.n_bits() < 512 {
        ops.push(("combine", OpSig::Combine));
    }
    if ty.n_bits() > 128 {
        ops.push(("split", OpSig::Split));
    }

    if ty.scalar == ScalarType::Unsigned && ty.n_bits() == 512 {
        ops.push(("load_interleaved_128", OpSig::LoadInterleaved(128, 4)));
    }
    
    if matches!(ty.scalar, ScalarType::Unsigned | ScalarType::Float) && ty.n_bits() == 512 {
        ops.push(("store_interleaved_128", OpSig::StoreInterleaved(128, 4)));
    }

    if cvt {
        if matches!(ty.scalar, ScalarType::Unsigned) {
            if let Some(widened) = ty.widened() {
                ops.push(("widen", OpSig::WidenNarrow(widened)));
            }

            if let Some(narrowed) = ty.narrowed() {
                ops.push(("narrow", OpSig::WidenNarrow(narrowed)));
            }
        }

        if valid_reinterpret(ty, ScalarType::Unsigned, 8) {
            ops.push((
                "reinterpret_u8",
                OpSig::Reinterpret(ScalarType::Unsigned, 8),
            ));
        }

        match (ty.scalar, ty.scalar_bits) {
            (ScalarType::Float, 32) => ops.push(("cvt_u32", OpSig::Cvt(ScalarType::Unsigned, 32))),
            (ScalarType::Unsigned, 32) => ops.push(("cvt_f32", OpSig::Cvt(ScalarType::Float, 32))),
            _ => (),
        }
    }
    ops
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TyFlavor {
    /// Types for methods in the `Simd` trait; `f32x4<Self>`
    SimdTrait,
    /// Types for methods in the vec trait; `f32x4<S>`
    VecImpl,
}

impl OpSig {
    pub fn simd_trait_args(&self, vec_ty: &VecType) -> TokenStream {
        let ty = vec_ty.rust();
        match self {
            OpSig::Splat => {
                let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                quote! { self, val: #scalar }
            }
            OpSig::LoadInterleaved(block_size, i) => {
                let ty = load_interleaved_arg_ty(*block_size, *i, vec_ty);
                quote! { self, #ty }
            }
            OpSig::StoreInterleaved(block_size, i) => {
                let ty = store_interleaved_arg_ty(*block_size, *i, vec_ty);
                quote! { self, #ty }
            }
            OpSig::Unary
            | OpSig::Split
            | OpSig::Cvt(_, _)
            | OpSig::Reinterpret(_, _)
            | OpSig::WidenNarrow(_) => quote! { self, a: #ty<Self> },
            OpSig::Binary | OpSig::Compare | OpSig::Combine | OpSig::Zip(_) => {
                quote! { self, a: #ty<Self>, b: #ty<Self> }
            }
            OpSig::Shift => {
                quote! { self, a: #ty<Self>, shift: u32 }
            }
            OpSig::Ternary => {
                quote! { self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self> }
            }
            OpSig::Select => {
                let mask_ty = vec_ty.mask_ty().rust();
                quote! { self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self> }
            }
        }
    }

    pub fn vec_trait_args(&self) -> Option<TokenStream> {
        let args = match self {
            OpSig::Splat | OpSig::LoadInterleaved(_, _) | OpSig::StoreInterleaved(_, _) => return None,
            OpSig::Unary | OpSig::Cvt(_, _) | OpSig::Reinterpret(_, _) | OpSig::WidenNarrow(_) => {
                quote! { self }
            }
            OpSig::Binary | OpSig::Compare | OpSig::Zip(_) | OpSig::Combine => {
                quote! { self, rhs: impl SimdInto<Self, S> }
            }
            OpSig::Shift => {
                quote! { self, shift: u32 }
            }
            OpSig::Ternary => {
                quote! { self, op1: impl SimdInto<Self, S>, op2: impl SimdInto<Self, S> }
            }
            // select is currently done by trait, but maybe we'll implement for
            // masks.
            OpSig::Select => return None,
            // These signatures involve types not in the Simd trait
            OpSig::Split => return None,
        };
        Some(args)
    }

    pub fn ret_ty(&self, ty: &VecType, flavor: TyFlavor) -> TokenStream {
        let quant = match flavor {
            TyFlavor::SimdTrait => quote! { <Self> },
            TyFlavor::VecImpl => quote! { <S> },
        };
        match self {
            OpSig::Splat
            | OpSig::Unary
            | OpSig::Binary
            | OpSig::Select
            | OpSig::Ternary
            | OpSig::Shift
            | OpSig::LoadInterleaved(_, _)  => {
                let rust = ty.rust();
                quote! { #rust #quant }
            }
            OpSig::Compare => {
                let rust = ty.mask_ty().rust();
                quote! { #rust #quant }
            }
            OpSig::Combine => {
                let n2 = ty.len * 2;
                let result = VecType::new(ty.scalar, ty.scalar_bits, n2).rust();
                quote! { #result #quant }
            }
            OpSig::Split => {
                let len = ty.len / 2;
                let result = VecType::new(ty.scalar, ty.scalar_bits, len).rust();
                quote! { ( #result #quant, #result #quant ) }
            }
            OpSig::Zip(_) => {
                let rust = ty.rust();
                quote! { #rust #quant }
            }
            OpSig::Cvt(scalar, scalar_bits) => {
                let result = VecType::new(*scalar, *scalar_bits, ty.len).rust();
                quote! { #result #quant }
            }
            OpSig::Reinterpret(scalar, scalar_bits) => {
                let result = reinterpret_ty(ty, *scalar, *scalar_bits).rust();
                quote! { #result #quant }
            }
            OpSig::WidenNarrow(t) => {
                let result = t.rust();
                quote! { #result #quant }
            }
            | OpSig::StoreInterleaved(_, _) => quote! {()}
        }
    }
}

pub(crate) fn load_interleaved_arg_ty(block_size: u16, i: u16, vec_ty: &VecType) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = (block_size * i) as usize / vec_ty.scalar_bits;
    quote! { src: &[#scalar; #len] }
}

pub(crate) fn store_interleaved_arg_ty(block_size: u16, i: u16, vec_ty: &VecType) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = (block_size * i) as usize / vec_ty.scalar_bits;
    quote! { dest: &mut [#scalar; #len] }
}

pub(crate) fn reinterpret_ty(src: &VecType, dst_scalar: ScalarType, dst_bits: usize) -> VecType {
    VecType::new(dst_scalar, dst_bits, src.n_bits() / dst_bits)
}

pub(crate) fn valid_reinterpret(src: &VecType, dst_scalar: ScalarType, dst_bits: usize) -> bool {
    if src.scalar == dst_scalar && src.scalar_bits == dst_bits {
        return false;
    }

    if matches!(src.scalar, ScalarType::Mask | ScalarType::Float) {
        return false;
    }

    true
}
