// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for the Neon level.

use core::arch::aarch64::*;

use crate::{
    f32x4, f32x8,
    impl_macros::{impl_op, impl_simd_from_into},
    mask16x4, mask16x8, mask32x4, mask32x8,
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
    #[inline]
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

    #[inline]
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

    impl_op!(simd_eq_f32x4(a: f32x4, b: f32x4) -> mask32x4 = neon.vreinterpretq_s32_u32(neon.vceqq_f32));
    impl_op!(simd_lt_f32x4(a: f32x4, b: f32x4) -> mask32x4 = neon.vreinterpretq_s32_u32(neon.vcltq_f32));
    impl_op!(simd_le_f32x4(a: f32x4, b: f32x4) -> mask32x4 = neon.vreinterpretq_s32_u32(neon.vcleq_f32));
    impl_op!(simd_ge_f32x4(a: f32x4, b: f32x4) -> mask32x4 = neon.vreinterpretq_s32_u32(neon.vcgeq_f32));
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

    #[inline(always)]
    fn simd_ne_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> mask32x4<Self> {
        self.not_mask32x4(self.simd_eq_f32x4(a, b))
    }

    #[inline(always)]
    fn splat_f32x8(self, val: f32) -> f32x8<Self> {
        self.combine_f32x4(self.splat_f32x4(val), self.splat_f32x4(val))
    }

    #[inline(always)]
    fn add_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_f32x4(self.add_f32x4(a0, b0), self.add_f32x4(a1, b1))
    }

    #[inline(always)]
    fn sub_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_f32x4(self.sub_f32x4(a0, b0), self.sub_f32x4(a1, b1))
    }

    #[inline(always)]
    fn mul_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_f32x4(self.mul_f32x4(a0, b0), self.mul_f32x4(a1, b1))
    }

    #[inline(always)]
    fn div_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_f32x4(self.div_f32x4(a0, b0), self.div_f32x4(a1, b1))
    }

    #[inline(always)]
    fn mul_add_f32x8(self, a: f32x8<Self>, b: f32x8<Self>, c: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        let (c0, c1) = self.split_f32x8(c);
        self.combine_f32x4(
            self.mul_add_f32x4(a0, b0, c0),
            self.mul_add_f32x4(a1, b1, c1),
        )
    }

    #[inline(always)]
    fn abs_f32x8(self, a: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        self.combine_f32x4(self.abs_f32x4(a0), self.abs_f32x4(a1))
    }

    #[inline(always)]
    fn sqrt_f32x8(self, a: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        self.combine_f32x4(self.sqrt_f32x4(a0), self.sqrt_f32x4(a1))
    }

    #[inline(always)]
    fn copysign_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_f32x4(self.copysign_f32x4(a0, b0), self.copysign_f32x4(a1, b1))
    }

    #[inline(always)]
    fn simd_eq_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> mask32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_mask32x4(self.simd_eq_f32x4(a0, b0), self.simd_eq_f32x4(a1, b1))
    }

    #[inline(always)]
    fn simd_ne_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> mask32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_mask32x4(self.simd_ne_f32x4(a0, b0), self.simd_ne_f32x4(a1, b1))
    }

    #[inline(always)]
    fn simd_lt_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> mask32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_mask32x4(self.simd_lt_f32x4(a0, b0), self.simd_lt_f32x4(a1, b1))
    }

    #[inline(always)]
    fn simd_le_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> mask32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_mask32x4(self.simd_le_f32x4(a0, b0), self.simd_le_f32x4(a1, b1))
    }

    #[inline(always)]
    fn simd_ge_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> mask32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_mask32x4(self.simd_ge_f32x4(a0, b0), self.simd_ge_f32x4(a1, b1))
    }

    #[inline(always)]
    fn simd_gt_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> mask32x8<Self> {
        let (a0, a1) = self.split_f32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        self.combine_mask32x4(self.simd_gt_f32x4(a0, b0), self.simd_gt_f32x4(a1, b1))
    }

    #[inline(always)]
    fn select_f32x8(self, a: mask32x8<Self>, b: f32x8<Self>, c: f32x8<Self>) -> f32x8<Self> {
        let (a0, a1) = self.split_mask32x8(a);
        let (b0, b1) = self.split_f32x8(b);
        let (c0, c1) = self.split_f32x8(c);
        self.combine_f32x4(self.select_f32x4(a0, b0, c0), self.select_f32x4(a1, b1, c1))
    }

    #[inline(always)]
    fn combine_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x8<Self> {
        let mut result = [0.0; 8];
        result[0..4].copy_from_slice(&a.val);
        result[4..8].copy_from_slice(&b.val);
        result.simd_into(self)
    }

    #[inline(always)]
    fn split_f32x8(self, a: f32x8<Self>) -> (f32x4<Self>, f32x4<Self>) {
        let mut b0 = [0.0; 4];
        let mut b1 = [0.0; 4];
        b0.copy_from_slice(&a.val[0..4]);
        b1.copy_from_slice(&a.val[4..8]);
        (b0.simd_into(self), b1.simd_into(self))
    }
}

// These should go into the Simd trait
impl Neon {
    impl_op!(not_mask32x4(a: mask32x4) -> mask32x4 = neon.vmvnq_s32);

    #[inline(always)]
    pub fn combine_mask32x4(self, a: mask32x4<Self>, b: mask32x4<Self>) -> mask32x8<Self> {
        let mut result = [0; 8];
        result[0..4].copy_from_slice(&a.val);
        result[4..8].copy_from_slice(&b.val);
        result.simd_into(self)
    }

    #[inline(always)]
    pub fn split_mask32x8(self, a: mask32x8<Self>) -> (mask32x4<Self>, mask32x4<Self>) {
        let mut b0 = [0; 4];
        let mut b1 = [0; 4];
        b0.copy_from_slice(&a.val[0..4]);
        b1.copy_from_slice(&a.val[4..8]);
        (b0.simd_into(self), b1.simd_into(self))
    }
}
