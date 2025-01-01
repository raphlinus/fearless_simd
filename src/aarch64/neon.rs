// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for the Neon level.

use core::arch::aarch64::*;

use crate::{
    f32x4,
    impl_macros::{impl_op, impl_simd_from_into},
    mask16x4, mask16x8, mask32x4,
    seal::Seal,
    Simd, SimdFrom, SimdInto,
};

use super::Level;

/// The SIMD token for the "neon" level.
#[derive(Clone, Copy, Debug)]
pub struct Neon {
    pub neon: crate::core_arch::aarch64::Neon,
}

impl Neon {
    pub unsafe fn new_unchecked() -> Self {
        Neon {
            neon: crate::core_arch::aarch64::Neon::new_unchecked(),
        }
    }
}

impl_simd_from_into!(f32x4, float32x4_t);
impl_simd_from_into!(mask32x4, int32x4_t);
impl_simd_from_into!(mask16x4, int16x4_t);
impl_simd_from_into!(mask16x8, int16x8_t);

impl Seal for Neon {}

impl Simd for Neon {
    #[inline(always)]
    fn level(self) -> Level {
        Level::Neon(self)
    }

    fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
        #[target_feature(enable = "neon")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn vectorize_neon<F: FnOnce() -> R, R>(f: F) -> R {
            f()
        }
        unsafe { vectorize_neon(f) }
    }

    #[inline(always)]
    fn splat_f32x4(self, val: f32) -> f32x4<Self> {
        self.neon.vdupq_n_f32(val).simd_into(self)
    }

    impl_op!(add_f32x4(a: f32x4, b: f32x4) -> f32x4 = neon.vaddq_f32);
    impl_op!(sub_f32x4(a: f32x4, b: f32x4) -> f32x4 = neon.vsubq_f32);
    impl_op!(mul_f32x4(a: f32x4, b: f32x4) -> f32x4 = neon.vmulq_f32);
    impl_op!(div_f32x4(a: f32x4, b: f32x4) -> f32x4 = neon.vdivq_f32);
    impl_op!(mul_add_f32x4(a: f32x4, b: f32x4, c: f32x4) -> f32x4 = neon.vfmaq_f32(c, a, b));

    impl_op!(simd_gt_f32x4(a: f32x4, b: f32x4) -> mask32x4 = neon.vreinterpretq_s32_u32(neon.vcgtq_f32));
    impl_op!(select_f32x4(a: mask32x4, b: f32x4, c: f32x4) -> f32x4
        = neon.vbslq_f32(neon.vreinterpretq_u32_s32(a), b, c));
    impl_op!(sqrt_f32x4(a: f32x4) -> f32x4 = neon.vsqrtq_f32);
    impl_op!(abs_f32x4(a: f32x4) -> f32x4 = neon.vabsq_f32);

    #[inline(always)]
    fn copysign_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self> {
        let sign_mask = self.neon.vdupq_n_u32(1 << 31);
        self.neon
            .vbslq_f32(sign_mask, b.into(), a.into())
            .simd_into(self)
    }
}
