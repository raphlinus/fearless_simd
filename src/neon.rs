// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub mod f16_4;
pub mod f16_8;
pub mod f32_4;
pub mod mask16_4;
pub mod mask16_8;
pub mod mask32_4;
pub mod u32_4;

pub use f16_8 as f16s;
pub use f32_4 as f32s;
pub use mask16_8 as mask16s;
pub use mask32_4 as mask32s;
pub use u32_4 as u32s;

macro_rules! neon_f16_unaryop {
    ( $opfn:ident ( $ty:ty ) = $asm:literal, $arch:ty ) => {
        #[target_feature(enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty) -> $ty {
            unsafe {
                let inp: $arch = a.into();
                let result: $arch;
                core::arch::asm!(
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
pub(self) use neon_f16_unaryop;

macro_rules! neon_f16_binop {
    ( $opfn:ident ( $ty:ty ) = $asm:literal, $arch:ty ) => {
        #[target_feature(enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> $ty {
            unsafe {
                let inp1: $arch = a.into();
                let inp2: $arch = b.into();
                let result: $arch;
                core::arch::asm!(
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
pub(self) use neon_f16_binop;

macro_rules! neon_f16_ternary {
    ( $opfn:ident ( $ty:ty ) = $asm:literal, $arch:ty ) => {
        #[target_feature(enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty, c: $ty) -> $ty {
            unsafe {
                let inp1: $arch = a.into();
                let inp2: $arch = b.into();
                let inp3: $arch = c.into();
                let result: $arch;
                // Note order; intrinsic computes v0 + v1 * v2,
                // we want a * b + c
                core::arch::asm!(
                    $asm,
                    inout(vreg) inp3 => result,
                    in(vreg) inp1,
                    in(vreg) inp2,
                    options(pure, nomem, nostack, preserves_flags)
                );
                result.into()
            }
        }
    };
}
pub(self) use neon_f16_ternary;

macro_rules! neon_f16_cmp {
    ( $opfn:ident ( $ty:ty ) = $asm:literal, $arch:ty ) => {
        #[target_feature(enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> <$ty as $crate::Simd>::Mask {
            unsafe {
                let inp1: $arch = a.into();
                let inp2: $arch = b.into();
                let result: $arch;
                core::arch::asm!(
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
pub(self) use neon_f16_cmp;

macro_rules! neon_f16_cvt {
    ( $opfn:ident ( $from:ty ) -> $to:ty = $asm:literal ( $archfrom:ty ) -> $archto:ty) => {
        #[target_feature(enable = "fp16")]
        #[inline]
        pub fn $opfn(a: $from) -> $to {
            unsafe {
                let inp: $archfrom = a.into();
                let result: $archto;
                core::arch::asm!(
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
pub(self) use neon_f16_cvt;
