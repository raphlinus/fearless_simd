// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`f16x8`] SIMD values.

use core::arch::aarch64::*;
use core::arch::asm;

use crate::f16x8;
use crate::macros::impl_simd_from_into;
use crate::mask16x8;

impl_simd_from_into!(f16x8, int16x8_t);

macro_rules! impl_unaryop {
    ( $opfn:ident ( $ty:ty ) = $asm:expr, $arch:ty ) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty) -> $ty {
            unsafe {
                let inp: $arch = a.into();
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
    ( $opfn:ident ( $ty:ty ) = $asm:expr, $arch:ty ) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> $ty {
            unsafe {
                let inp1: $arch = a.into();
                let inp2: $arch = b.into();
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

macro_rules! impl_ternary {
    ( $opfn:ident ( $ty:ty ) = $asm:expr, $arch:ty ) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> $ty {
            unsafe {
                let inp1: $arch = a.into();
                let inp2: $arch = b.into();
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

macro_rules! impl_cmp {
    ( $opfn:ident ( $ty:ty ) = $asm:expr, $arch:ty ) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> <$ty as $crate::Simd>::Mask {
            unsafe {
                let inp1: $arch = a.into();
                let inp2: $arch = b.into();
                let result: $arch;
                asm!(
                    $asm,
                    out(vreg) result,
                    in(vreg) inp1,
                    in(vreg) inp2,
                    options(pure, nomem, nostack, preserves_flags)
                );
                core::mem::transmute(result)
            }
        }
    };
}

impl_unaryop!(abs(f16x8) = "fabs.8h {0:v}, {1:v}", uint16x8_t);
impl_unaryop!(floor(f16x8) = "frintm.8h {0:v}, {1:v}", uint16x8_t);
impl_unaryop!(ceil(f16x8) = "frintp.8h {0:v}, {1:v}", uint16x8_t);
impl_unaryop!(round_ties_even(f16x8) = "frintn.8h {0:v}, {1:v}", uint16x8_t);
impl_unaryop!(trunc(f16x8) = "frintz.8h {0:v}, {1:v}", uint16x8_t);
impl_unaryop!(sqrt(f16x8) = "fsqrt.8h {0:v}, {1:v}", uint16x8_t);
impl_binop!(add(f16x8) = "fadd.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_binop!(sub(f16x8) = "fsub.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_binop!(mul(f16x8) = "fmul.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_binop!(div(f16x8) = "fdiv.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_binop!(min(f16x8) = "fminm.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_binop!(max(f16x8) = "fmaxm.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_ternary!(mul_add(f16x8) = "fmla.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_ternary!(mul_sub(f16x8) = "fmls.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_cmp!(simd_eq(f16x8) = "fcmeq.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_cmp!(simd_le(f16x8) = "fcmle.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_cmp!(simd_lt(f16x8) = "fcmlt.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_cmp!(simd_gt(f16x8) = "fcmgt.8h {0:v}, {1:v}, {2:v}", uint16x8_t);
impl_cmp!(simd_ge(f16x8) = "fcmge.8h {0:v}, {1:v}, {2:v}", uint16x8_t);

#[target_feature(enable = "neon", enable = "fp16")]
#[inline]
pub fn simd_ne(a: f16x8, b: f16x8) -> mask16x8 {
    super::mask16_8::not(simd_eq(a, b))
}
