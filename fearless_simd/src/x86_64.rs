// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for x86_64 SIMD capabilities.

use crate::{Fallback, WithSimd};
pub use avx2::Avx2;

mod avx2;

/// The level enum for x86_64 architectures.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    Fallback(Fallback),
    Avx2(Avx2),
    // TODO: Avx512 (either nightly or pending stabilization)
}

impl Level {
    pub fn new() -> Self {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("lzcnt")
        {
            unsafe { Level::Avx2(Avx2::new_unchecked()) }
        } else {
            Level::Fallback(Fallback::new())
        }
    }

    #[inline]
    pub fn as_avx2(self) -> Option<Avx2> {
        if let Level::Avx2(avx2) = self {
            Some(avx2)
        } else {
            None
        }
    }

    #[inline]
    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[target_feature(enable = "avx2,bmi2,f16c,fma,lzcnt")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn dispatch_avx2<W: WithSimd>(f: W, avx2: Avx2) -> W::Output {
            f.with_simd(avx2)
        }
        match self {
            Level::Fallback(fallback) => f.with_simd(fallback),
            Level::Avx2(avx2) => unsafe { dispatch_avx2(f, avx2) },
        }
    }
}
