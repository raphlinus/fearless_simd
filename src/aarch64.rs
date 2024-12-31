// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for Aarch64 SIMD capabilities.

pub mod neon;

use crate::{Fallback, WithSimd};
use neon::Neon;

/// The level enum for aarch64 architectures.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    Fallback(Fallback),
    Neon(Neon),
    // TODO: fp16
}

impl Level {
    pub fn new() -> Self {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { Level::Neon(Neon::new_unchecked()) }
        } else {
            Level::Fallback(Fallback::new())
        }
    }

    pub fn as_neon(self) -> Option<Neon> {
        if let Level::Neon(neon) = self {
            Some(neon)
        } else {
            None
        }
    }

    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[target_feature(enable = "neon")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn dispatch_neon<W: WithSimd>(f: W, neon: Neon) -> W::Output {
            f.with_simd(neon)
        }
        match self {
            Level::Fallback(fallback) => f.with_simd(fallback),
            Level::Neon(neon) => unsafe { dispatch_neon(f, neon) },
        }
    }
}