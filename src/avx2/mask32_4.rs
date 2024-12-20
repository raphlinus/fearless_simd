// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`mask32x4`] SIMD values.

use core::arch::x86_64::*;

use crate::{macros::impl_simd_from_into, mask32x4};

impl_simd_from_into!(mask32x4, __m128);
