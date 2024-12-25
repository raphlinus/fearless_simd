// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Implementation of core ops for SIMD types

use crate::{f32x4, Simd, SimdInto};

// Todo: macro-ify + more ops

macro_rules! impl_op_binary {
    ( $trait:ident, $ty:ident, $scalar:ident, $opfn:ident, $simd_fn:ident) => {
        impl<S: Simd> core::ops::$trait for $ty<S> {
            type Output = Self;
            #[inline(always)]
            fn $opfn(self, rhs: Self) -> Self::Output {
                self.simd.$simd_fn(self, rhs)
            }
        }
        impl<S: Simd> core::ops::$trait<f32> for $ty<S> {
            type Output = Self;

            #[inline(always)]
            fn $opfn(self, rhs: f32) -> Self::Output {
                self.simd.$simd_fn(self, rhs.simd_into(self.simd))
            }
        }

        impl<S: Simd> core::ops::$trait<$ty<S>> for f32 {
            type Output = $ty<S>;

            #[inline(always)]
            fn $opfn(self, rhs: $ty<S>) -> Self::Output {
                rhs.simd.$simd_fn(self.simd_into(rhs.simd), rhs)
            }
        }
    };
}

impl_op_binary!(Add, f32x4, f32, add, add_f32x4);
impl_op_binary!(Mul, f32x4, f32, mul, mul_f32x4);
