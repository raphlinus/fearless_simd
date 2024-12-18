// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`mask32x4`] SIMD values.

use core::arch::aarch64::*;

use crate::{
    macros::{impl_binop, impl_cmp_mask, impl_select, impl_simd_from_into, impl_unaryop},
    mask32x4,
};

impl_simd_from_into!(mask32x4, int32x4_t);

impl_unaryop!("neon": not(mask32x4) = vmvnq_s32);
impl_unaryop!("neon": neg(mask32x4) = vnegq_s32);
impl_unaryop!("neon": abs(mask32x4) = vabsq_s32);
impl_binop!("neon": add(mask32x4) = vaddq_s32);
impl_binop!("neon": sub(mask32x4) = vsubq_s32);
impl_binop!("neon": mul(mask32x4) = vmulq_s32);
impl_binop!("neon": bitand(mask32x4) = vandq_s32);
impl_binop!("neon": bitor(mask32x4) = vorrq_s32);
impl_binop!("neon": bitxor(mask32x4) = veorq_s32);
impl_cmp_mask!("neon": simd_eq(mask32x4) = vceqq_s32);
impl_cmp_mask!("neon": simd_le(mask32x4) = vcleq_s32);
impl_cmp_mask!("neon": simd_lt(mask32x4) = vcltq_s32);
impl_cmp_mask!("neon": simd_gt(mask32x4) = vcgtq_s32);
impl_cmp_mask!("neon": simd_ge(mask32x4) = vcgeq_s32);
impl_select!(mask "neon": (mask32x4) = vbslq_s32, vreinterpretq_u32_s32);

// TODO: we might want to convert from bool
#[target_feature(enable = "neon")]
#[inline]
pub fn splat(value: i32) -> mask32x4 {
    unsafe { vdupq_n_s32(value).into() }
}

#[target_feature(enable = "neon")]
#[inline]
pub fn simd_ne(a: mask32x4, b: mask32x4) -> mask32x4 {
    not(simd_eq(a, b))
}
