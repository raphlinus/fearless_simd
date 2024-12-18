// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`f16x4`] SIMD values.

use core::arch::aarch64::*;

use crate::macros::impl_simd_from_into;
use crate::{f16, f16x4, f32x4, mask16x4};

use super::{neon_f16_binop, neon_f16_cmp, neon_f16_cvt, neon_f16_ternary, neon_f16_unaryop};

impl_simd_from_into!(f16x4, uint16x4_t);

neon_f16_unaryop!(abs(f16x4) = "fabs.4h {0:v}, {1:v}", uint16x4_t);
neon_f16_unaryop!(floor(f16x4) = "frintm.4h {0:v}, {1:v}", uint16x4_t);
neon_f16_unaryop!(ceil(f16x4) = "frintp.4h {0:v}, {1:v}", uint16x4_t);
neon_f16_unaryop!(
    round_ties_even(f16x4) = "frintn.4h {0:v}, {1:v}",
    uint16x4_t
);
neon_f16_unaryop!(round(f16x4) = "frinta.4h {0:v}, {1:v}", uint16x4_t);
neon_f16_unaryop!(trunc(f16x4) = "frintz.4h {0:v}, {1:v}", uint16x4_t);
neon_f16_unaryop!(sqrt(f16x4) = "fsqrt.4h {0:v}, {1:v}", uint16x4_t);
neon_f16_binop!(add(f16x4) = "fadd.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_binop!(sub(f16x4) = "fsub.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_binop!(mul(f16x4) = "fmul.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_binop!(div(f16x4) = "fdiv.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_binop!(min(f16x4) = "fminm.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_binop!(max(f16x4) = "fmaxm.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_ternary!(mul_add(f16x4) = "fmla.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_ternary!(mul_sub(f16x4) = "fmls.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_cmp!(simd_eq(f16x4) = "fcmeq.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_cmp!(simd_le(f16x4) = "fcmle.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_cmp!(simd_lt(f16x4) = "fcmlt.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_cmp!(simd_gt(f16x4) = "fcmgt.4h {0:v}, {1:v}, {2:v}", uint16x4_t);
neon_f16_cmp!(simd_ge(f16x4) = "fcmge.4h {0:v}, {1:v}, {2:v}", uint16x4_t);

#[target_feature(enable = "neon")]
#[inline]
pub fn splat(value: f16) -> f16x4 {
    unsafe { vdup_n_u16(value.to_bits()).into() }
}

#[target_feature(enable = "fp16")]
#[inline]
pub fn splat_f32(value: f32) -> f16x4 {
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
pub fn splat_f32_const(value: f32) -> f16x4 {
    splat(f16::from_f32_const(value))
}

#[target_feature(enable = "fp16")]
#[inline]
pub fn simd_ne(a: f16x4, b: f16x4) -> mask16x4 {
    super::mask16_4::not(simd_eq(a, b))
}

neon_f16_cvt!(cvt_f32(f16x4) -> f32x4 = "fcvtl {0:v}.4s, {1:v}.4h" (uint16x4_t) -> float32x4_t);
