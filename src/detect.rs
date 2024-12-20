// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fast detection of CPU level.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::cell::UnsafeCell;

    struct SimdCache(UnsafeCell<SimdLevelX86_64>);

    // Safety: we will only allow concurrent writes from unsafe functions
    // with safety documentation.
    unsafe impl Sync for SimdCache {}

    static SIMD_CACHE: SimdCache = SimdCache(UnsafeCell::new(SimdLevelX86_64::Uninit));

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
    #[repr(u32)]
    pub enum SimdLevelX86_64 {
        Uninit,
        Nothing,
        Avx2,
    }

    fn get_avx2_level() -> SimdLevelX86_64 {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("lzcnt")
        {
            SimdLevelX86_64::Avx2
        } else {
            SimdLevelX86_64::Nothing
        }
    }

    /// Initialize the flag for detection of the avx2 level.
    ///
    /// Applications should call this on startup.
    ///
    /// Safety: When this function is called, there must be no other threads
    /// running that access the feature detection flag.
    pub unsafe fn init_simd_detect() {
        unsafe {
            if *SIMD_CACHE.0.get() == SimdLevelX86_64::Uninit {
                *SIMD_CACHE.0.get() = get_avx2_level();
            }
        }
    }

    #[inline]
    pub fn is_avx2_detected() -> bool {
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "fma",
            target_feature = "lzcnt",
        ))]
        {
            true
        }
        #[cfg(not(all(
            target_feature = "avx2",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "fma",
            target_feature = "lzcnt",
        )))]
        // Safety: no data race as per contract of `init_simd_detect`
        unsafe {
            *SIMD_CACHE.0.get() >= SimdLevelX86_64::Avx2
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use avx2::*;

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use core::cell::UnsafeCell;

    struct SimdCache(UnsafeCell<SimdLevelAarch64>);

    // Safety: we will only allow concurrent writes from unsafe functions
    // with safety documentation.
    unsafe impl Sync for SimdCache {}

    static SIMD_CACHE: SimdCache = SimdCache(UnsafeCell::new(SimdLevelAarch64::Uninit));

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
    #[repr(u32)]
    pub enum SimdLevelAarch64 {
        Uninit,
        Nothing,
        Neon,
        Fp16,
    }

    fn get_neon_level() -> SimdLevelAarch64 {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            SimdLevelAarch64::Fp16
        } else if std::arch::is_aarch64_feature_detected!("neon") {
            SimdLevelAarch64::Neon
        } else {
            SimdLevelAarch64::Nothing
        }
    }

    /// Initialize the flag for detection of the avx2 level.
    ///
    /// Applications should call this on startup.
    ///
    /// Safety: When this function is called, there must be no other threads
    /// running that access the feature detection flag.
    pub unsafe fn init_simd_detect() {
        unsafe {
            if *SIMD_CACHE.0.get() == SimdLevelAarch64::Uninit {
                *SIMD_CACHE.0.get() = get_neon_level();
            }
        }
    }

    #[inline]
    pub fn is_neon_detected() -> bool {
        #[cfg(target_feature = "neon")]
        {
            true
        }
        #[cfg(not(target_feature = "neon"))]
        // Safety: no data race as per contract of `init_simd_detect`
        unsafe {
            *SIMD_CACHE.0.get() >= SimdLevelAarch64::Neon
        }
    }

    #[inline]
    pub fn is_fp16_detected() -> bool {
        #[cfg(target_feature = "fp16")]
        {
            true
        }
        #[cfg(not(target_feature = "fp16"))]
        // Safety: no data race as per contract of `init_simd_detect`
        unsafe {
            *SIMD_CACHE.0.get() >= SimdLevelAarch64::Fp16
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
/// Placeholder for SIMD detection code. It doesn't do anything but is marked
/// as unsafe for consistency with other architectures.
pub unsafe fn init_simd_detect() {}
