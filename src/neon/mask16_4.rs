// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`mask16x4`] SIMD values.

use core::arch::aarch64::*;

use crate::{
    macros::{impl_binop, impl_cmp_mask, impl_simd_from_into, impl_unaryop},
    mask16x4,
};

impl_simd_from_into!(mask16x4, int16x4_t);

impl_unaryop!("neon": not(mask16x4) = vmvn_s16);
impl_unaryop!("neon": neg(mask16x4) = vneg_s16);
impl_unaryop!("neon": abs(mask16x4) = vabs_s16);
impl_binop!("neon": add(mask16x4) = vadd_s16);
impl_binop!("neon": sub(mask16x4) = vsub_s16);
impl_binop!("neon": mul(mask16x4) = vmul_s16);
impl_binop!("neon": bitand(mask16x4) = vand_s16);
impl_binop!("neon": bitor(mask16x4) = vorr_s16);
impl_binop!("neon": bitxor(mask16x4) = veor_s16);
impl_cmp_mask!("neon": simd_eq(mask16x4) = vceq_s16);
impl_cmp_mask!("neon": simd_le(mask16x4) = vcle_s16);
impl_cmp_mask!("neon": simd_lt(mask16x4) = vclt_s16);
impl_cmp_mask!("neon": simd_gt(mask16x4) = vcgt_s16);
impl_cmp_mask!("neon": simd_ge(mask16x4) = vcge_s16);

// TODO: we might want to convert from bool
#[target_feature(enable = "neon")]
#[inline]
pub fn splat(value: i16) -> mask16x4 {
    unsafe { vdup_n_s16(value).into() }
}

#[target_feature(enable = "neon")]
#[inline]
pub fn simd_ne(a: mask16x4, b: mask16x4) -> mask16x4 {
    not(simd_eq(a, b))
}
