// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to fp16 intrinsics on aarch64.

use core::arch::aarch64::*;

use crate::{core_arch::aarch64::Neon, f16};

/// A token for FP16 intrinsics on aarch64.
#[derive(Clone, Copy, Debug)]
pub struct Fp16 {
    neon: Neon,
}

macro_rules! neon_f16_unaryop {
    ( $opfn:ident ( $ty:ty ) -> $ret:ty = $asm:literal ) => {
        #[inline(always)]
        pub fn $opfn(self, a: $ty) -> $ret {
            #[target_feature(enable = "fp16")]
            #[inline]
            pub unsafe fn inner(a: $ty) -> $ret {
                let result;
                core::arch::asm!(
                    $asm,
                    out(vreg) result,
                    in(vreg) a,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result
            }
            unsafe { inner(a) }
        }
    };
}

macro_rules! neon_f16_binop {
    ( $opfn:ident ( $tya:ty, $tyb:ty ) -> $ret:ty = $asm:literal ) => {
        #[inline(always)]
        pub fn $opfn(self, a: $tya, b: $tyb) -> $ret {
            #[target_feature(enable = "fp16")]
            #[inline]
            pub unsafe fn inner(a: $tya, b: $tyb) -> $ret {
                let result;
                core::arch::asm!(
                    $asm,
                    out(vreg) result,
                    in(vreg) a,
                    in(vreg) b,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result
            }
            unsafe { inner(a, b) }
        }
    };
}

macro_rules! neon_f16_binop_inout {
    ( $opfn:ident ( $tya:ty, $tyb:ty ) -> $ret:ty = $asm:literal ) => {
        #[inline(always)]
        pub fn $opfn(self, a: $tya, b: $tyb) -> $ret {
            #[target_feature(enable = "fp16")]
            #[inline]
            pub unsafe fn inner(a: $tya, b: $tyb) -> $ret {
                let result;
                core::arch::asm!(
                    $asm,
                    inout(vreg) a => result,
                    in(vreg) b,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result
            }
            unsafe { inner(a, b) }
        }
    };
}

macro_rules! neon_f16_ternary {
    ( $opfn:ident ( $tya:ty, $tyb:ty, $tyc:ty ) -> $ret:ty = $asm:literal ) => {
        #[inline(always)]
        pub fn $opfn(self, a: $tya, b: $tyb, c: $tyc) -> $ret {
            #[target_feature(enable = "fp16")]
            #[inline]
            pub unsafe fn inner(a: $tya, b: $tyb, c: $tyc) -> $ret {
                let result;
                core::arch::asm!(
                    $asm,
                    inout(vreg) a => result,
                    in(vreg) b,
                    in(vreg) c,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result
            }
            unsafe { inner(a, b, c) }
        }
    };
}

// When f16 support lands in Rust, this type will be defined in core::arch::arch64,
// hopefully compatibly.
pub type float16x4_t = uint16x4_t;
pub type float16x8_t = uint16x8_t;

impl Fp16 {
    /// Create a SIMD token.
    ///
    /// # Safety:
    ///
    /// The required CPU features must be available.
    #[inline]
    pub unsafe fn new_unchecked() -> Self {
        Self {
            neon: Neon::new_unchecked(),
        }
    }

    // This is a somewhat curated set for now, but we should make it reasonably complete.
    neon_f16_unaryop!(vabs_f16(float16x4_t) -> float16x4_t = "fabs.4h {0:v}, {1:v}");
    neon_f16_unaryop!(vrnd_f16(float16x4_t) -> float16x4_t = "frintz.4h {0:v}, {1:v}");
    neon_f16_unaryop!(vrnda_f16(float16x4_t) -> float16x4_t = "frinta.4h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndm_f16(float16x4_t) -> float16x4_t = "frintm.4h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndn_f16(float16x4_t) -> float16x4_t = "frintn.4h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndp_f16(float16x4_t) -> float16x4_t = "frintp.4h {0:v}, {1:v}");
    neon_f16_unaryop!(vsqrt_f16(float16x4_t) -> float16x4_t = "fsqrt.4h {0:v}, {1:v}");
    neon_f16_binop!(vadd_f16(float16x4_t, float16x4_t) -> float16x4_t = "fadd.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vsub_f16(float16x4_t, float16x4_t) -> float16x4_t = "fsub.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vmul_f16(float16x4_t, float16x4_t) -> float16x4_t = "fmul.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vdiv_f16(float16x4_t, float16x4_t) -> float16x4_t = "fdiv.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vmax_f16(float16x4_t, float16x4_t) -> float16x4_t = "fmax.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vmaxnm_f16(float16x4_t, float16x4_t) -> float16x4_t = "fmaxnm.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vmin_f16(float16x4_t, float16x4_t) -> float16x4_t = "fmin.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vminnm_f16(float16x4_t, float16x4_t) -> float16x4_t = "fminnm.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vceq_f16(float16x4_t, float16x4_t) -> uint16x4_t = "fcmeq.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vcle_f16(float16x4_t, float16x4_t) -> uint16x4_t = "fcmle.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vclt_f16(float16x4_t, float16x4_t) -> uint16x4_t = "fcmlt.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vcgt_f16(float16x4_t, float16x4_t) -> uint16x4_t = "fcmgt.4h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vcge_f16(float16x4_t, float16x4_t) -> uint16x4_t = "fcmge.4h {0:v}, {1:v}, {2:v}");
    neon_f16_ternary!(vmla_f16(float16x4_t, float16x4_t, float16x4_t) -> float16x4_t = "fmla.4h {0:v}, {1:v}, {2:v}");

    neon_f16_unaryop!(vabsq_f16(float16x8_t) -> float16x8_t = "fabs.8h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndq_f16(float16x8_t) -> float16x8_t = "frintz.8h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndaq_f16(float16x8_t) -> float16x8_t = "frinta.8h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndmq_f16(float16x8_t) -> float16x8_t = "frintm.8h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndnq_f16(float16x8_t) -> float16x8_t = "frintn.8h {0:v}, {1:v}");
    neon_f16_unaryop!(vrndpq_f16(float16x8_t) -> float16x8_t = "frintp.8h {0:v}, {1:v}");
    neon_f16_unaryop!(vsqrtq_f16(float16x8_t) -> float16x8_t = "fsqrt.8h {0:v}, {1:v}");
    neon_f16_binop!(vaddq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fadd.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vsubq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fsub.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vmulq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fmul.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vdivq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fdiv.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vmaxq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fmax.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vmaxnmq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fmaxnm.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vminq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fmin.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vminnmq_f16(float16x8_t, float16x8_t) -> float16x8_t = "fminnm.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vceqq_f16(float16x8_t, float16x8_t) -> uint16x8_t = "fcmeq.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vcleq_f16(float16x8_t, float16x8_t) -> uint16x8_t = "fcmle.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vcltq_f16(float16x8_t, float16x8_t) -> uint16x8_t = "fcmlt.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vcgtq_f16(float16x8_t, float16x8_t) -> uint16x8_t = "fcmgt.8h {0:v}, {1:v}, {2:v}");
    neon_f16_binop!(vcgeq_f16(float16x8_t, float16x8_t) -> uint16x8_t = "fcmge.8h {0:v}, {1:v}, {2:v}");
    neon_f16_unaryop!(vceqzq_f16(float16x8_t) -> uint16x8_t = "fcmeq.8h {0:v}, {1:v}, #0");
    neon_f16_ternary!(vmlaq_f16(float16x8_t, float16x8_t, float16x8_t) -> float16x8_t = "fmla.8h {0:v}, {1:v}, {2:v}");

    neon_f16_unaryop!(vcvt_f32_f16(float16x4_t) -> float32x4_t = "fcvtl {0:v}.4s, {1:v}.4h");
    neon_f16_unaryop!(vcvt_high_f32_f16(float16x8_t) -> float32x4_t = "fcvtl2 {0:v}.4s, {1:v}.8h");
    neon_f16_unaryop!(vcvt_f16_f32(float32x4_t) -> float16x4_t = "fcvtn {0:v}.4h, {1:v}.4s");
    neon_f16_binop_inout!(vcvt_high_f16_f32(float16x4_t, float32x4_t) -> float16x4_t = "fcvtn2 {0:v}.8h, {1:v}.4s");
    neon_f16_unaryop!(vcvtq_f16_u16(uint16x8_t) -> float16x8_t = "ucvtf.8h {0:v}, {1:v}");
    neon_f16_unaryop!(vcvtnq_u16_f16(float16x8_t) -> uint16x8_t = "fcvtnu.8h {0:v}, {1:v}");

    #[inline(always)]
    pub fn vbsl_f16(self, mask: uint16x4_t, a: float16x4_t, b: float16x4_t) -> float16x4_t {
        self.vreinterpret_f16_u16(self.neon.vbsl_u16(
            mask,
            self.vreinterpret_u16_f16(a),
            self.vreinterpret_u16_f16(b),
        ))
    }

    #[inline(always)]
    pub fn vbslq_f16(self, mask: uint16x8_t, a: float16x8_t, b: float16x8_t) -> float16x8_t {
        self.vreinterpretq_f16_u16(self.neon.vbslq_u16(
            mask,
            self.vreinterpretq_u16_f16(a),
            self.vreinterpretq_u16_f16(b),
        ))
    }

    #[inline(always)]
    pub fn vdup_n_f16(self, value: f16) -> float16x4_t {
        self.vreinterpret_f16_u16(self.neon.vdup_n_u16(value.to_bits()))
    }

    #[inline(always)]
    pub fn vdupq_n_f16(self, value: f16) -> float16x8_t {
        self.vreinterpretq_f16_u16(self.neon.vdupq_n_u16(value.to_bits()))
    }

    #[inline(always)]
    pub fn vreinterpret_f16_u16(self, a: uint16x4_t) -> float16x4_t {
        a
    }

    #[inline(always)]
    pub fn vreinterpret_u16_f16(self, a: float16x4_t) -> uint16x4_t {
        a
    }

    #[inline(always)]
    pub fn vreinterpretq_f16_u16(self, a: uint16x8_t) -> float16x8_t {
        a
    }

    #[inline(always)]
    pub fn vreinterpretq_u16_f16(self, a: float16x8_t) -> uint16x8_t {
        a
    }
    // TODO: the other reinterprets, but they can be worked around
}
