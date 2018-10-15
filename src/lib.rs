//! A helper library to make SIMD more friendly.

mod fallback;
mod traits;
mod combinators;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse42;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

pub use traits::{SimdF32, SimdMask};

pub use combinators::SimdFnF32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use avx::{AvxF32, AvxMask};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse42::{Sse42F32, Sse42Mask};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::{count, GeneratorF32};
