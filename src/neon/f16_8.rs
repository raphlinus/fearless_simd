// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`f16x8`] SIMD values.

use core::arch::aarch64::*;

use crate::macros::impl_simd_from_into;
use crate::{f16, f16x8, f32x8, mask16x8};

use super::{neon_f16_binop, neon_f16_cmp, neon_f16_ternary, neon_f16_unaryop};

impl_simd_from_into!(f16x8, uint16x8_t);

neon_f16_unaryop!(abs(f16x8) = "fabs.8h {0:v}, {1:v}", uint16x8_t);
neon_f16_unaryop!(floor(f16x8) = "frintm.8h {0:v}, {1:v}", uint16x8_t);
neon_f16_unaryop!(ceil(f16x8) = "frintp.8h {0:v}, {1:v}", uint16x8_t);
neon_f16_unaryop!(
    round_ties_even(f16x8) = "frintn.8h {0:v}, {1:v}",
    uint16x8_t
);
neon_f16_unaryop!(round(f16x8) = "frinta.8h {0:v}, {1:v}", uint16x8_t);
neon_f16_unaryop!(trunc(f16x8) = "frintz.8h {0:v}, {1:v}", uint16x8_t);
neon_f16_unaryop!(sqrt(f16x8) = "fsqrt.8h {0:v}, {1:v}", uint16x8_t);
neon_f16_binop!(add(f16x8) = "fadd.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_binop!(sub(f16x8) = "fsub.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_binop!(mul(f16x8) = "fmul.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_binop!(div(f16x8) = "fdiv.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_binop!(min(f16x8) = "fminm.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_binop!(max(f16x8) = "fmaxm.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_ternary!(mul_add(f16x8) = "fmla.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_ternary!(mul_sub(f16x8) = "fmls.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_cmp!(simd_eq(f16x8) = "fcmeq.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_cmp!(simd_le(f16x8) = "fcmle.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_cmp!(simd_lt(f16x8) = "fcmlt.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_cmp!(simd_gt(f16x8) = "fcmgt.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
neon_f16_cmp!(simd_ge(f16x8) = "fcmge.8h {0:v}, {1:v}, {2:v}", uint16x8_t);

#[target_feature(enable = "neon")]
#[inline]
pub fn splat(value: f16) -> f16x8 {
    unsafe { vdupq_n_u16(value.to_bits()).into() }
}

#[target_feature(enable = "fp16")]
#[inline]
pub fn splat_f32(value: f32) -> f16x8 {
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
pub fn splat_f32_const(value: f32) -> f16x8 {
    splat(f16::from_f32_const(value))
}

#[target_feature(enable = "fp16")]
#[inline]
pub fn simd_ne(a: f16x8, b: f16x8) -> mask16x8 {
    super::mask16_8::not(simd_eq(a, b))
}

#[target_feature(enable = "fp16")]
#[inline]
pub fn cvt_f32(value: f16x8) -> f32x8 {
    unsafe {
        let inp: uint16x8_t = value.into();
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
