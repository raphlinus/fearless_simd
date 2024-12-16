// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::arch::aarch64::*;
use core::arch::asm;

use crate::f32x4;
use crate::macros::impl_simd_from_into;
use crate::{Simd, f16, f16x4, f16x8};

macro_rules! impl_unaryop {
    ($opfn:ident, $arch:ty, $asm:expr) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(self) -> Self {
            unsafe {
                let inp: $arch = self.into();
                let result: $arch;
                asm!(
                    $asm,
                    out(vreg) result,
                    in(vreg) inp,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result.into()
            }
        }
    };
}

macro_rules! impl_binop {
    ($opfn:ident, $arch:ty, $asm:expr) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(self, rhs: Self) -> Self {
            unsafe {
                let inp1: $arch = self.into();
                let inp2: $arch = rhs.into();
                let result: $arch;
                asm!(
                    $asm,
                    out(vreg) result,
                    in(vreg) inp1,
                    in(vreg) inp2,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result.into()
            }
        }
    };
}

macro_rules! impl_ternop {
    ($opfn:ident, $arch:ty, $asm:expr) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(self, a: Self, b: Self) -> Self {
            unsafe {
                let inp1: $arch = self.into();
                let inp2: $arch = a.into();
                let inp3: $arch = b.into();
                let result: $arch;
                asm!(
                    $asm,
                    inout(vreg) inp1 => result,
                    in(vreg) inp2,
                    in(vreg) inp3,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result.into()
            }
        }
    };
}

macro_rules! impl_cmp {
    ($opfn:ident, $arch:ty, $asm:expr) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(self, rhs: Self) -> <Self as Simd>::Mask {
            unsafe {
                let inp1: $arch = self.into();
                let inp2: $arch = rhs.into();
                let result: $arch;
                asm!(
                    $asm,
                    out(vreg) result,
                    in(vreg) inp1,
                    in(vreg) inp2,
                    options(pure, nomem, nostack, preserves_flags)
                );
                core::mem::transmute::<$arch, <Self as Simd>::Mask>(result)
            }
        }
    };
}

macro_rules! impl_cast {
    ( $opfn:ident -> $to:ident,
        $asm:expr, $archfrom:ident -> $archto:ident
    ) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(self) -> $to {
            unsafe {
                let inp: $archfrom = self.into();
                let result: $archto;
                asm!(
                    $asm,
                    out(vreg) result,
                    in(vreg) inp,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result.into()
            }
        }
    };
}

impl_simd_from_into!(f16x8, uint16x8_t);

impl f16x8 {
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat(value: f16) -> Self {
        unsafe {
            vdupq_n_u16(value.to_bits()).into()
        }
    }
    
    #[target_feature(enable = "neon", enable = "fp16")]
    #[inline]
    pub fn splat_f32(value: f32) -> Self {
        unsafe {
            let result: uint16x8_t;
            core::arch::asm!(
                "fcvt {0:h}, {1:s}",
                "dup.8h {0:v}, {0:v}[0]",
                out(vreg) result,
                in(vreg) value,
                options(pure, nomem, nostack, preserves_flags)
            );
            result.into()
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat_f32_const(value: f32) -> Self {
        Self::splat(f16::from_f32_const(value))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn get_low(self) -> f16x4 {
        unsafe {
            vget_low_u16(self.into()).into()
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn get_high(self) -> f16x4 {
        unsafe {
            vget_high_u16(self.into()).into()
        }
    }

    // This function becomes cast_f32 when we implement f32x8
    #[target_feature(enable = "neon", enable = "fp16")]
    #[inline]
    pub fn to_f32_array(self) -> [f32; 8] {
        unsafe {
            let inp = core::mem::transmute::<Self, uint16x8_t>(self);
            let lo: float32x4_t;
            let hi_u16x8;
            core::arch::asm!(
                "fcvtl {0:v}.4s, {1:v}.4h",
                "fcvtl2 {1:v}.4s, {1:v}.8h",
                out(vreg) lo,
                inout(vreg) inp => hi_u16x8,
                options(pure, nomem, nostack, preserves_flags)
            );
            let hi = vreinterpretq_f32_u16(hi_u16x8);
            let lo_as_array = &lo as *const float32x4_t as *const [f32; 4];
            let hi_as_array = &hi as *const float32x4_t as *const [f32; 4];
            let mut tmp = core::mem::MaybeUninit::uninit();
            core::ptr::copy_nonoverlapping(lo_as_array, tmp.as_mut_ptr() as *mut [f32; 4], 1);
            core::ptr::copy_nonoverlapping(hi_as_array, (tmp.as_mut_ptr() as *mut [f32; 4]).add(1), 1);
            tmp.assume_init()
        }
    }

    impl_unaryop!(abs, uint16x8_t, "fabs.8h {0:v}, {1:v}");
    impl_unaryop!(floor, uint16x8_t, "frintm.8h {0:v}, {1:v}");
    impl_unaryop!(ceil, uint16x8_t, "frintp.8h {0:v}, {1:v}");
    impl_unaryop!(round_ties_even, uint16x8_t, "frintn.8h {0:v}, {1:v}");
    impl_unaryop!(trunc, uint16x8_t, "frintz.8h {0:v}, {1:v}");
    impl_unaryop!(sqrt, uint16x8_t, "fsqrt.8h {0:v}, {1:v}");
    impl_binop!(add, uint16x8_t, "fadd.8h {0:v}, {1:v}, {2:v}");
    impl_binop!(sub, uint16x8_t, "fsub.8h {0:v}, {1:v}, {2:v}");
    impl_binop!(mul, uint16x8_t, "fmul.8h {0:v}, {1:v}, {2:v}");
    impl_binop!(div, uint16x8_t, "fdiv.8h {0:v}, {1:v}, {2:v}");
    impl_binop!(min, uint16x8_t, "fminm.8h {0:v}, {1:v}, {2:v}");
    impl_binop!(max, uint16x8_t, "fmaxm.8h {0:v}, {1:v}, {2:v}");
    impl_ternop!(mul_add, uint16x8_t, "fmla.8h {0:v}, {1:v}, {2:v}");
    impl_ternop!(mul_sub, uint16x8_t, "fmls.8h {0:v}, {1:v}, {2:v}");
    impl_cmp!(simd_eq, uint16x8_t, "fcmeq.8h {0:v}, {1:v}, {2:v}");
    impl_cmp!(simd_ne, uint16x8_t, "fcmne.8h {0:v}, {1:v}, {2:v}");
    impl_cmp!(simd_le, uint16x8_t, "fcmle.8h {0:v}, {1:v}, {2:v}");
    impl_cmp!(simd_lt, uint16x8_t, "fcmlt.8h {0:v}, {1:v}, {2:v}");
    impl_cmp!(simd_gt, uint16x8_t, "fcmgt.8h {0:v}, {1:v}, {2:v}");
    impl_cmp!(simd_ge, uint16x8_t, "fcmge.8h {0:v}, {1:v}, {2:v}");
}

// f16x4

impl_simd_from_into!(f16x4, uint16x4_t);

impl f16x4 {
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat(value: f16) -> Self {
        unsafe {
            vdup_n_u16(value.to_bits()).into()
        }
    }

    #[target_feature(enable = "neon", enable = "fp16")]
    #[inline]
    pub fn splat_f32(value: f32) -> Self {
        unsafe {
            let result: uint16x4_t;
            core::arch::asm!(
                "fcvt {0:h}, {1:s}",
                "dup.4h {0:v}, {0:v}[0]",
                out(vreg) result,
                in(vreg) value,
                options(pure, nomem, nostack, preserves_flags)
            );
            result.into()
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat_f32_const(value: f32) -> Self {
        Self::splat(f16::from_f32_const(value))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn combine(self, high: Self) -> f16x8 {
        unsafe {
            vcombine_u16(self.into(), high.into()).into()
        }
    }

    impl_binop!(add, uint16x4_t, "fadd.4h {0:v}, {1:v}, {2:v}");
    impl_binop!(sub, uint16x4_t, "fsub.4h {0:v}, {1:v}, {2:v}");
    impl_binop!(mul, uint16x4_t, "fmul.4h {0:v}, {1:v}, {2:v}");
    impl_binop!(div, uint16x4_t, "fdiv.4h {0:v}, {1:v}, {2:v}");
    impl_binop!(min, uint16x4_t, "fmin.4h {0:v}, {1:v}, {2:v}");
    impl_binop!(max, uint16x4_t, "fmax.4h {0:v}, {1:v}, {2:v}");
    impl_unaryop!(reverse, uint16x4_t, "rev64.4h {0:v}, {1:v}");
    impl_ternop!(mul_add, uint16x4_t, "fmla.4h {0:v}, {1:v}, {2:v}");
    impl_cast!(cast_f32 -> f32x4, "fcvtl {0:v}.4s {1:v}.4h", uint16x4_t -> float32x4_t);
}

impl f32x4 {
    impl_cast!(cast_f16 -> f16x4, "fcvtn {0:v}.4h {1:v}.4s", float32x4_t -> uint16x4_t);
}
