// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::arch::aarch64::*;
use core::mem::transmute;

use crate::{f32x4, mask32x4, u32x4, Simd};
use crate::macros::{impl_binop, impl_cast, impl_cmp, impl_cmp_mask, impl_select, impl_simd_from_into, impl_ternary, impl_unaryop};

impl_simd_from_into!(f32x4, float32x4_t);

impl f32x4 {
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat(value: f32) -> Self {
        unsafe {
            transmute(vdupq_n_f32(value))
        }
    }

    impl_unaryop!("neon": neg, vnegq_f32);
    impl_unaryop!("neon": abs, vabsq_f32);
    impl_unaryop!("neon": floor, vrndmq_f32);
    impl_unaryop!("neon": ceil, vrndpq_f32);
    impl_unaryop!("neon": round, vrndaq_f32);
    impl_unaryop!("neon": round_ties_even, vrndnq_f32);
    impl_unaryop!("neon": trunc, vrndq_f32);
    impl_unaryop!("neon": sqrt, vsqrtq_f32);
    impl_binop!("neon": add, vaddq_f32);
    impl_binop!("neon": sub, vsubq_f32);
    impl_binop!("neon": mul, vmulq_f32);
    impl_binop!("neon": div, vdivq_f32);
    impl_binop!("neon": min, vminnmq_f32);
    impl_binop!("neon": max, vmaxnmq_f32);
    impl_ternary!("neon": mul_add, vfmaq_f32);
    impl_ternary!("neon": mul_sub, vfmsq_f32);
    impl_cmp!("neon": simd_eq, vceqq_f32);
    impl_cmp!("neon": simd_le, vcleq_f32);
    impl_cmp!("neon": simd_lt, vcltq_f32);
    impl_cmp!("neon": simd_gt, vcgtq_f32);
    impl_cmp!("neon": simd_ge, vcgeq_f32);
    impl_cast!("neon": trunc_cast_u32 -> u32x4, vcvtq_u32_f32);
    impl_cast!("neon": round_cast_u32 -> u32x4, vcvtaq_u32_f32);
    impl_cast!("neon": floor_cast_u32 -> u32x4, vcvtmq_u32_f32);
    impl_cast!("neon": ceil_cast_u32 -> u32x4, vcvtpq_u32_f32);
    impl_cast!("neon": round_ties_even_cast_u32 -> u32x4, vcvtnq_u32_f32);

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn simd_ne(self, rhs: Self) -> <Self as Simd>::Mask {
        self.simd_eq(rhs).not()
    }
}

impl_simd_from_into!(mask32x4, int32x4_t);

impl mask32x4 {
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat(value: bool) -> Self {
        unsafe {
            transmute(vdupq_n_s32(-(value as i32)))
        }
    }

    impl_unaryop!("neon": not, vmvnq_s32);
    impl_binop!("neon": bitand, vandq_s32);
    impl_binop!("neon": bitor, vorrq_s32);
    impl_binop!("neon": bitxor, veorq_s32);
    impl_cmp_mask!("neon": simd_eq, vceqq_s32);
    impl_select!("neon": vbslq_s32);

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn simd_ne(self, rhs: Self) -> Self {
        self.simd_eq(rhs).not()
    }
}
