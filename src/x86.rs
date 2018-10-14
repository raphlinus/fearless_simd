//! Runtime detection of x86 and x86_64 capabilities.

use traits::SimdF32;
use avx::AvxF32;

pub enum SimdCaps {
    Avx(AvxF32),
    Fallback,
    // TODO: other levels
}

pub fn detect() -> SimdCaps {
    if is_x86_feature_detected!("avx") {
        unsafe { SimdCaps::Avx(AvxF32::create()) }
    } else {
        SimdCaps::Fallback
    }
}
