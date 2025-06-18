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

#[cfg(target_arch = "wasm32")]
pub mod wasm32 {
    pub use crate::generated::WasmSimd128;
}

/// The level enum with the specific SIMD capabilities available.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    Fallback(Fallback),
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    Neon(Neon),
    #[cfg(target_arch = "wasm32")]
    WasmSimd128(WasmSimd128),
}

impl Level {
    pub fn new() -> Self {
        #[cfg(all(feature = "std", target_arch = "aarch64"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { Level::Neon(Neon::new_unchecked()) };
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return Level::WasmSimd128(WasmSimd128::new_unchecked());
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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

    #[cfg(target_arch = "wasm32")]
    #[inline]
    pub fn as_wasm_simd128(self) -> Option<WasmSimd128> {
        match self {
            Level::WasmSimd128(simd128) => Some(simd128),
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

        #[cfg(target_arch = "wasm32")]
        #[target_feature(enable = "simd128")]
        #[inline]
        fn dispatch_simd128<W: WithSimd>(f: W, simd128: WasmSimd128) -> W::Output {
            f.with_simd(simd128)
        }

        #[inline]
        fn dispatch_fallback<W: WithSimd>(f: W, fallback: Fallback) -> W::Output {
            f.with_simd(fallback)
        }

        match self {
            #[cfg(all(feature = "std", target_arch = "aarch64"))]
            Level::Neon(neon) => unsafe { dispatch_neon(f, neon) },
            #[cfg(target_arch = "wasm32")]
            Level::WasmSimd128(simd128) => dispatch_simd128(f, simd128),
            Level::Fallback(fallback) => dispatch_fallback(f, fallback),
        }
    }
}
