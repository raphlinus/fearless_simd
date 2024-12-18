// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`u32x4`] SIMD values.

use core::arch::aarch64::*;

use crate::{
    f32x4,
    macros::{impl_binop, impl_cast, impl_cmp, impl_select, impl_simd_from_into, impl_unaryop},
    mask32x4, u32x4,
};

impl_simd_from_into!(u32x4, uint32x4_t);

// TODO: neg op probably cast through i32
impl_unaryop!("neon": not(u32x4) = vmvnq_u32);
impl_binop!("neon": add(u32x4) = vaddq_u32);
impl_binop!("neon": sub(u32x4) = vsubq_u32);
impl_binop!("neon": mul(u32x4) = vmulq_u32);
impl_binop!("neon": min(u32x4) = vminq_u32);
impl_binop!("neon": max(u32x4) = vmaxq_u32);
impl_cmp!("neon": simd_eq(u32x4) = vceqq_u32);
impl_cmp!("neon": simd_le(u32x4) = vcleq_u32);
impl_cmp!("neon": simd_lt(u32x4) = vcltq_u32);
impl_cmp!("neon": simd_gt(u32x4) = vcgtq_u32);
impl_cmp!("neon": simd_ge(u32x4) = vcgeq_u32);
impl_cast!("neon": cast_f32(u32x4) -> f32x4 = vcvtq_f32_u32);
impl_select!("neon": (u32x4) = vbslq_u32, vreinterpretq_u32_s32);

#[target_feature(enable = "neon")]
#[inline]
pub fn splat(value: u32) -> u32x4 {
    unsafe { vdupq_n_u32(value).into() }
}

#[target_feature(enable = "neon")]
#[inline]
pub fn simd_ne(a: u32x4, b: u32x4) -> mask32x4 {
    super::mask32_4::not(simd_eq(a, b))
}
