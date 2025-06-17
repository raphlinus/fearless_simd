// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for Aarch64 SIMD capabilities.

use crate::{Fallback, WithSimd};
pub use crate::generated::Neon;

/// The level enum for aarch64 architectures.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    Fallback(Fallback),
    Neon(Neon),
    //Fp16(Fp16),
}

impl Level {
    pub fn new() -> Self {
        if std::arch::is_aarch64_feature_detected!("neon") {
            /*
            if std::arch::is_aarch64_feature_detected!("fp16") {
                return unsafe { Level::Fp16(Fp16::new_unchecked()) }
            }
            */
            unsafe { Level::Neon(Neon::new_unchecked()) }
        } else {
            Self::fallback()
        }
    }

    #[inline]
    pub fn as_neon(self) -> Option<Neon> {
        match self {
            Level::Neon(neon) => Some(neon),
            _ => None,
            //Level::Fp16(fp16) => Some(fp16.to_neon()),
        }
    }
    
    #[inline]
    pub fn fallback() -> Self {
        Self::Fallback(Fallback::new())
    }

    /*
    #[inline]
    pub fn as_fp16(self) -> Option<Fp16> {
        match self {
            Level::Fp16(fp16) => Some(fp16),
            _ => None,
        }
    }
    */

    #[inline]
    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[target_feature(enable = "neon")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn dispatch_neon<W: WithSimd>(f: W, neon: Neon) -> W::Output {
            f.with_simd(neon)
        }
        
        #[inline]
        fn dispatch_fallback<W: WithSimd>(f: W, fallback: Fallback) -> W::Output {
            f.with_simd(fallback)
        }
        /*
        #[target_feature(enable = "neon,fp16")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn dispatch_fp16<W: WithSimd>(f: W, fp16: Fp16) -> W::Output {
            f.with_simd(fp16)
        }
        */
        match self {
            Level::Neon(neon) => unsafe { dispatch_neon(f, neon) },
            Level::Fallback(fallback) => dispatch_fallback(f, fallback)
            //Level::Fp16(fp16) => unsafe { dispatch_fp16(f, fp16) },
        }
    }
}
