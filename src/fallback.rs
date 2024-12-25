// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fallback implementations of SIMD methods.

// pub mod f32_4;

// Discussion question: what's the natural width? Most other implementations
// implementations default to 1, but perhaps we want to express the idea
// of operating on larger chunks of data, and encouraging autovectorization.
// pub use f32_4 as f32s;

use crate::{f32x4, mask32x4, seal::Seal, Level, Simd, SimdInto};

/// A SIMD token representing portable fallback.
///
/// Unlike other tokens, this one can always be constructed safely.
#[derive(Clone, Copy, Debug)]
pub struct Fallback {
    _private: (),
}

impl Fallback {
    pub fn new() -> Self {
        Fallback { _private: () }
    }
}

fn mask(b: bool) -> i32 {
    -(b as i32)
}

fn sel1(a: i32, b: f32, c: f32) -> f32 {
    if a < 0 {
        b
    } else {
        c
    }
}

impl Seal for Fallback {}

impl Simd for Fallback {
    fn level(self) -> Level {
        Level::Fallback(self)
    }

    #[inline]
    fn splat_f32x4(self, val: f32) -> f32x4<Self> {
        [val; 4].simd_into(self)
    }

    #[inline]
    fn add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self> {
        let val = [
            a.val[0] + b.val[0],
            a.val[1] + b.val[1],
            a.val[2] + b.val[2],
            a.val[3] + b.val[3],
        ];
        val.simd_into(self)
    }

    #[inline]
    fn mul_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self> {
        let val = [
            a.val[0] * b.val[0],
            a.val[1] * b.val[1],
            a.val[2] * b.val[2],
            a.val[3] * b.val[3],
        ];
        val.simd_into(self)
    }

    #[inline]
    fn mul_add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self> {
        let val = [
            a.val[0].mul_add(b.val[0], c.val[0]),
            a.val[1].mul_add(b.val[1], c.val[1]),
            a.val[2].mul_add(b.val[2], c.val[2]),
            a.val[3].mul_add(b.val[3], c.val[3]),
        ];
        val.simd_into(self)
    }

    #[inline]
    fn abs_f32x4(self, a: f32x4<Self>) -> f32x4<Self> {
        a.val.map(f32::abs).simd_into(self)
    }

    #[inline]
    fn simd_gt_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> mask32x4<Self> {
        [
            mask(a.val[0] > b.val[0]),
            mask(a.val[1] > b.val[1]),
            mask(a.val[2] > b.val[2]),
            mask(a.val[3] > b.val[3]),
        ]
        .simd_into(self)
    }

    #[inline]
    fn select_f32x4(self, a: mask32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self> {
        [
            sel1(a.val[0], b.val[0], c.val[0]),
            sel1(a.val[1], b.val[1], c.val[1]),
            sel1(a.val[2], b.val[2], c.val[2]),
            sel1(a.val[3], b.val[3], c.val[3]),
        ]
        .simd_into(self)
    }
}
