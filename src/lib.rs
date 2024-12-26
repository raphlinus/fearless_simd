// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A helper library to make SIMD more friendly.

#![allow(non_camel_case_types)]

mod base;
mod fallback;
mod macros;
mod ops;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub use avx2::Level;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::Level;

pub use base::*;
// TODO: be more consistent, either bring avx2 items to crate level
// or make fallback a module too.
pub use fallback::*;

/*
mod detect;

pub use detect::*;

// For now, only bring in f16 on aarch64. We can also bring it in
// on x86_64, but only Sapphire Rapids supports it.

#[cfg(all(target_arch = "aarch64", feature = "half"))]
pub type f16 = half::f16;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
mod half_assed;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
pub use half_assed::f16;

*/

// Experimental
// pub mod token;
