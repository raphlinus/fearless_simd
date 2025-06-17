// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A helper library to make SIMD more friendly.

#![allow(non_camel_case_types)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod core_arch;
mod impl_macros;

mod generated;
mod macros;
mod traits;

pub use generated::*;
pub use traits::*;

// For now, only bring in f16 on aarch64. We can also bring it in
// on x86_64, but only Sapphire Rapids supports it.

#[cfg(all(target_arch = "aarch64", feature = "half"))]
pub type f16 = half::f16;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
mod half_assed;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
pub use half_assed::f16;

#[cfg(all(not(feature = "libm"), not(feature = "std")))]
compile_error!("fearless_simd requires either the `std` or `libm` feature");

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub mod aarch64 {
    pub use crate::generated::Neon;
}

/// The level enum with the specific SIMD capabilities available.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    Fallback(Fallback),
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    Neon(Neon),
}

impl Level {
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    pub fn new() -> Self {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { Level::Neon(Neon::new_unchecked()) }
        } else {
            Self::fallback()
        }
    }

    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    pub fn new() -> Self {
        Self::fallback()
    }
    
    #[cfg(any(not(feature = "std"), not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
    pub fn new() -> Self {
        Self::fallback()
    }

    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    #[inline]
    pub fn as_neon(self) -> Option<Neon> {
        match self {
            Level::Neon(neon) => Some(neon),
            _ => None,
        }
    }

    #[inline]
    pub fn fallback() -> Self {
        Self::Fallback(Fallback::new())
    }

    #[inline]
    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[cfg(all(feature = "std", target_arch = "aarch64"))]
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

        match self {
            #[cfg(all(feature = "std", target_arch = "aarch64"))]
            Level::Neon(neon) => unsafe { dispatch_neon(f, neon) },
            Level::Fallback(fallback) => dispatch_fallback(f, fallback)
        }
    }
}