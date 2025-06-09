// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::types::{ScalarType, VecType};

#[derive(Clone, Copy)]
pub enum OpSig {
    Splat,
    Unary,
    Binary,
    Compare,
    Select,
    Combine,
    Split,
    Zip,
    // TODO: fma
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
    ("zip", OpSig::Zip),
    ("unzip", OpSig::Zip),
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
    ("simd_eq", OpSig::Compare),
    ("simd_lt", OpSig::Compare),
    ("simd_le", OpSig::Compare),
    ("simd_ge", OpSig::Compare),
    ("simd_gt", OpSig::Compare),
    ("zip", OpSig::Zip),
    ("unzip", OpSig::Zip),
    ("select", OpSig::Select),
];

pub const MASK_OPS: &[(&str, OpSig)] = &[
    ("splat", OpSig::Splat),
    ("not", OpSig::Unary),
    ("and", OpSig::Binary),
    ("or", OpSig::Binary),
    ("xor", OpSig::Binary),
    ("select", OpSig::Select),
    ("zip", OpSig::Zip),
    ("unzip", OpSig::Zip),
    ("simd_eq", OpSig::Compare),
];

/// Ops covered by core::ops
pub const CORE_OPS: &[&str] = &["not", "neg", "add", "sub", "mul", "div", "and", "or", "xor"];

pub fn ops_for_type(ty: &VecType) -> Vec<(&str, OpSig)> {
    let base = match ty.scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    let mut ops = base.to_vec();
    if ty.n_bits() < 256 {
        ops.push(("combine", OpSig::Combine));
    }
    if ty.n_bits() > 128 {
        ops.push(("split", OpSig::Split));
    }
    ops
}
