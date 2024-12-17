// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`mask16x8`] SIMD values.

use core::arch::aarch64::*;

use crate::{
    macros::{impl_binop, impl_cmp_mask, impl_simd_from_into, impl_unaryop},
    mask16x8,
};

impl_simd_from_into!(mask16x8, int16x8_t);

impl_unaryop!("neon": not(mask16x8) = vmvnq_s16);
impl_unaryop!("neon": neg(mask16x8) = vnegq_s16);
impl_unaryop!("neon": abs(mask16x8) = vabsq_s16);
impl_binop!("neon": add(mask16x8) = vaddq_s16);
impl_binop!("neon": sub(mask16x8) = vsubq_s16);
impl_binop!("neon": mul(mask16x8) = vmulq_s16);
impl_binop!("neon": bitand(mask16x8) = vandq_s16);
impl_binop!("neon": bitor(mask16x8) = vorrq_s16);
impl_binop!("neon": bitxor(mask16x8) = veorq_s16);
impl_cmp_mask!("neon": simd_eq(mask16x8) = vceqq_s16);
impl_cmp_mask!("neon": simd_le(mask16x8) = vcleq_s16);
impl_cmp_mask!("neon": simd_lt(mask16x8) = vcltq_s16);
impl_cmp_mask!("neon": simd_gt(mask16x8) = vcgtq_s16);
impl_cmp_mask!("neon": simd_ge(mask16x8) = vcgeq_s16);

// TODO: we might want to convert from bool
#[target_feature(enable = "neon")]
#[inline]
pub fn splat(value: i16) -> mask16x8 {
    unsafe { vdupq_n_s16(value).into() }
}

#[target_feature(enable = "neon")]
#[inline]
pub fn simd_ne(a: mask16x8, b: mask16x8) -> mask16x8 {
    not(simd_eq(a, b))
}
