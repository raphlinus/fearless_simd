// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{f16, mask16x4, mask16x8};

impl_simd!(f16x8, f16, 8, mask16x8);
impl_simd!(f16x4, f16, 4, mask16x4);
