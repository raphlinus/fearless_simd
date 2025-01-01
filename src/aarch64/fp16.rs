// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for the fp16 level in aarch64.

use crate::{
    core_arch::aarch64::{float16x4_t, float16x8_t},
    f16, f16x4, f16x8, f32x4,
    impl_macros::{impl_op, impl_simd_from_into},
    mask16x4, mask16x8, mask32x4,
    seal::Seal,
    Select, Simd, SimdFrom, SimdInto,
};

use super::Level;

/// The SIMD token for the "neon" level.
#[derive(Clone, Copy, Debug)]
pub struct Fp16 {
    pub neon: crate::core_arch::aarch64::Neon,
    pub fp16: crate::core_arch::aarch64::Fp16,
}

impl Fp16 {
    pub unsafe fn new_unchecked() -> Self {
        Fp16 {
            neon: crate::core_arch::aarch64::Neon::new_unchecked(),
            fp16: crate::core_arch::aarch64::Fp16::new_unchecked(),
        }
    }

    pub fn to_neon(self) -> super::Neon {
        // Safety: Fp16 is a superset of Neon
        unsafe { super::Neon::new_unchecked() }
    }
}

impl_simd_from_into!(f16x4, float16x4_t);
impl_simd_from_into!(f16x8, float16x8_t);

impl Seal for Fp16 {}

impl Simd for Fp16 {
    #[inline(always)]
    fn level(self) -> Level {
        Level::Fp16(self)
    }

    fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
        #[target_feature(enable = "neon,fp16")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn vectorize_fp16<F: FnOnce() -> R, R>(f: F) -> R {
            f()
        }
        unsafe { vectorize_fp16(f) }
    }

    // Lots of copy-paste here, probably should be replaced with delegation to neon,
    // and that in turn might be a macro.
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

/// Methods specific to Fp16 - this is mostly f16 math.
impl Fp16 {
    #[inline(always)]
    pub fn splat_f16x4(self, val: f16) -> f16x4<Self> {
        self.fp16.vdup_n_f16(val).simd_into(self)
    }

    #[inline(always)]
    pub fn add_f16x4(self, a: f16x4<Self>, b: f16x4<Self>) -> f16x4<Self> {
        self.fp16.vadd_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn sub_f16x4(self, a: f16x4<Self>, b: f16x4<Self>) -> f16x4<Self> {
        self.fp16.vsub_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn mul_f16x4(self, a: f16x4<Self>, b: f16x4<Self>) -> f16x4<Self> {
        self.fp16.vmul_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn div_f16x4(self, a: f16x4<Self>, b: f16x4<Self>) -> f16x4<Self> {
        self.fp16.vdiv_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn sqrt_f16x4(self, a: f16x4<Self>) -> f16x4<Self> {
        self.fp16.vsqrt_f16(a.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn select_f16x4(self, mask: mask16x4<Self>, a: f16x4<Self>, b: f16x4<Self>) -> f16x4<Self> {
        self.fp16
            .vbsl_f16(
                self.neon.vreinterpret_u16_s16(mask.into()),
                a.into(),
                b.into(),
            )
            .simd_into(self)
    }

    #[inline(always)]
    pub fn splat_f16x8(self, val: f16) -> f16x8<Self> {
        self.fp16.vdupq_n_f16(val).simd_into(self)
    }

    #[inline(always)]
    pub fn add_f16x8(self, a: f16x8<Self>, b: f16x8<Self>) -> f16x8<Self> {
        self.fp16.vaddq_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn sub_f16x8(self, a: f16x8<Self>, b: f16x8<Self>) -> f16x8<Self> {
        self.fp16.vsubq_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn mul_f16x8(self, a: f16x8<Self>, b: f16x8<Self>) -> f16x8<Self> {
        self.fp16.vmulq_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn div_f16x8(self, a: f16x8<Self>, b: f16x8<Self>) -> f16x8<Self> {
        self.fp16.vdivq_f16(a.into(), b.into()).simd_into(self)
    }

    #[inline(always)]
    pub fn sqrt_f16x8(self, a: f16x8<Self>) -> f16x8<Self> {
        self.fp16.vsqrtq_f16(a.into()).simd_into(self)
    }
}

impl f16x4<Fp16> {
    #[inline(always)]
    pub fn cvt_f32(self) -> f32x4<Fp16> {
        self.simd
            .fp16
            .vcvt_f32_f16(self.into())
            .simd_into(self.simd)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        self.simd.fp16.vsqrt_f16(self.into()).simd_into(self.simd)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        self.simd.fp16.vabs_f16(self.into()).simd_into(self.simd)
    }

    #[inline(always)]
    pub fn copysign(self, other: Self) -> Self {
        let sign_mask = self.simd.neon.vdup_n_u16(1 << 15);
        self.simd
            .neon
            .vbsl_u16(sign_mask, other.into(), self.into())
            .simd_into(self.simd)
    }
}

impl f16x8<Fp16> {
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        self.simd.fp16.vsqrtq_f16(self.into()).simd_into(self.simd)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        self.simd.fp16.vabsq_f16(self.into()).simd_into(self.simd)
    }

    #[inline(always)]
    pub fn copysign(self, other: Self) -> Self {
        let sign_mask = self.simd.neon.vdupq_n_u16(1 << 15);
        self.simd
            .neon
            .vbslq_u16(sign_mask, other.into(), self.into())
            .simd_into(self.simd)
    }
}

impl f32x4<Fp16> {
    #[inline(always)]
    pub fn cvt_f16(self) -> f16x4<Fp16> {
        self.simd
            .fp16
            .vcvt_f16_f32(self.into())
            .simd_into(self.simd)
    }
}

impl Select<f16x4<Fp16>> for mask16x4<Fp16> {
    fn select(self, if_true: f16x4<Fp16>, if_false: f16x4<Fp16>) -> f16x4<Fp16> {
        self.simd
            .fp16
            .vbsl_f16(
                self.simd.neon.vreinterpret_u16_s16(self.into()),
                if_false.into(),
                if_true.into(),
            )
            .simd_into(self.simd)
    }
}

impl Select<f16x8<Fp16>> for mask16x8<Fp16> {
    fn select(self, if_true: f16x8<Fp16>, if_false: f16x8<Fp16>) -> f16x8<Fp16> {
        self.simd
            .fp16
            .vbslq_f16(
                self.simd.neon.vreinterpretq_u16_s16(self.into()),
                if_false.into(),
                if_true.into(),
            )
            .simd_into(self.simd)
    }
}

macro_rules! impl_op_binary {
    ( $trait:ident, $ty:ident, $scalar:ident, $opfn:ident, $simd_fn:ident) => {
        impl core::ops::$trait for $ty<Fp16> {
            type Output = Self;
            #[inline(always)]
            fn $opfn(self, rhs: Self) -> Self::Output {
                self.simd.$simd_fn(self, rhs)
            }
        }
        impl core::ops::$trait<f16> for $ty<Fp16> {
            type Output = Self;

            #[inline(always)]
            fn $opfn(self, rhs: f16) -> Self::Output {
                self.simd.$simd_fn(self, rhs.simd_into(self.simd))
            }
        }

        impl core::ops::$trait<$ty<Fp16>> for f16 {
            type Output = $ty<Fp16>;

            #[inline(always)]
            fn $opfn(self, rhs: $ty<Fp16>) -> Self::Output {
                rhs.simd.$simd_fn(self.simd_into(rhs.simd), rhs)
            }
        }
    };
}

impl SimdFrom<f16, Fp16> for f16x4<Fp16> {
    fn simd_from(value: f16, simd: Fp16) -> Self {
        simd.splat_f16x4(value)
    }
}

impl SimdFrom<f16, Fp16> for f16x8<Fp16> {
    fn simd_from(value: f16, simd: Fp16) -> Self {
        simd.splat_f16x8(value)
    }
}

impl_op_binary!(Add, f16x4, f16, add, add_f16x4);
impl_op_binary!(Sub, f16x4, f16, sub, sub_f16x4);
impl_op_binary!(Mul, f16x4, f16, mul, mul_f16x4);
impl_op_binary!(Div, f16x4, f16, div, div_f16x4);
