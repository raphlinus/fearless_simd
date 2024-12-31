// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A helper library to make SIMD more friendly.

#![allow(non_camel_case_types)]

mod base;
pub mod core_arch;
mod fallback;
mod impl_macros;
mod macros;
mod ops;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub use avx2::Level;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "aarch64")]
pub use aarch64::Level;

pub use base::*;
// TODO: be more consistent, either bring avx2 items to crate level
// or make fallback a module too.
pub use fallback::*;
