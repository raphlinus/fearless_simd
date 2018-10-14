//! A helper library to make SIMD more friendly.

mod fallback;
mod traits;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

pub use traits::{SimdF32, SimdMask};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use avx::{AvxF32, AvxMask};
