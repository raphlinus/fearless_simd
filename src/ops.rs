// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Implementation of core ops for SIMD types

use crate::{f32x4, Simd};

// Todo: macro-ify + more ops

impl<S: Simd> core::ops::Add for f32x4<S> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.simd.add_f32x4(self, rhs)
    }
}
