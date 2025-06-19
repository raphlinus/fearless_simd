// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// A token for fallback SIMD.
#[derive(Clone, Copy, Debug)]
pub struct Fallback {
    _private: (),
}

impl Fallback {
    /// Create a SIMD token.
    #[inline]
    pub fn new() -> Self {
        Self { _private: () }
    }
}
