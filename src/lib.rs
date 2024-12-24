// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A helper library to make SIMD more friendly.

// TODO: remove when target_feature is stabilized (hopefully soon)
#![feature(target_feature_11)]
#![allow(non_camel_case_types)]

mod base;
mod detect;
mod macros;

pub use base::*;
pub use detect::*;

#[cfg(target_arch = "aarch64")]
pub mod neon;

// For now, only bring in f16 on aarch64. We can also bring it in
// on x86_64, but only Sapphire Rapids supports it.

#[cfg(all(target_arch = "aarch64", feature = "half"))]
pub type f16 = half::f16;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
mod half_assed;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
pub use half_assed::f16;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

pub mod fallback;

// Experimental
pub mod token;
