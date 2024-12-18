// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`f32x4`] SIMD values.

use core::arch::aarch64::*;

use crate::{
    f16x4, f32x4,
    macros::{impl_binop, impl_cast, impl_cmp, impl_simd_from_into, impl_ternary, impl_unaryop},
    mask32x4, u32x4,
};

use super::neon_f16_cvt;

impl_simd_from_into!(f32x4, float32x4_t);

impl_unaryop!("neon": neg(f32x4) = vnegq_f32);
impl_unaryop!("neon": abs(f32x4) = vabsq_f32);
impl_unaryop!("neon": floor(f32x4) = vrndmq_f32);
impl_unaryop!("neon": ceil(f32x4) = vrndpq_f32);
impl_unaryop!("neon": round(f32x4) = vrndaq_f32);
impl_unaryop!("neon": round_ties_even(f32x4) = vrndnq_f32);
impl_unaryop!("neon": trunc(f32x4) = vrndq_f32);
impl_unaryop!("neon": sqrt(f32x4) = vsqrtq_f32);
impl_binop!("neon": add(f32x4) = vaddq_f32);
impl_binop!("neon": sub(f32x4) = vsubq_f32);
impl_binop!("neon": mul(f32x4) = vmulq_f32);
impl_binop!("neon": div(f32x4) = vdivq_f32);
impl_binop!("neon": min(f32x4) = vminnmq_f32);
impl_binop!("neon": max(f32x4) = vmaxnmq_f32);
impl_ternary!("neon": mul_add(f32x4) = vfmaq_f32);
impl_ternary!("neon": mul_sub(f32x4) = vfmsq_f32);
impl_cmp!("neon": simd_eq(f32x4) = vceqq_f32);
impl_cmp!("neon": simd_le(f32x4) = vcleq_f32);
impl_cmp!("neon": simd_lt(f32x4) = vcltq_f32);
impl_cmp!("neon": simd_gt(f32x4) = vcgtq_f32);
impl_cmp!("neon": simd_ge(f32x4) = vcgeq_f32);
impl_cast!("neon": trunc_cast_u32(f32x4) -> u32x4 = vcvtq_u32_f32);
impl_cast!("neon": round_cast_u32(f32x4) -> u32x4 = vcvtaq_u32_f32);
impl_cast!("neon": floor_cast_u32(f32x4) -> u32x4 = vcvtmq_u32_f32);
impl_cast!("neon": ceil_cast_u32(f32x4) -> u32x4 = vcvtpq_u32_f32);
impl_cast!("neon": round_ties_even_cast_u32(f32x4) -> u32x4 = vcvtnq_u32_f32);

neon_f16_cvt!(cvt_f16(f32x4) -> f16x4 = "fcvtn {0:v}.4h, {1:v}.4s" (float32x4_t) -> uint16x4_t);

#[target_feature(enable = "neon")]
#[inline]
pub fn splat(value: f32) -> f32x4 {
    unsafe { vdupq_n_f32(value).into() }
}

#[target_feature(enable = "neon")]
#[inline]
pub fn simd_ne(a: f32x4, b: f32x4) -> mask32x4 {
    super::mask32_4::not(simd_eq(a, b))
}
