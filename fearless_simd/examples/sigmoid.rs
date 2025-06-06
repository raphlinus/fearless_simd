// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{Level, Simd, SimdInto, f32x4, simd_dispatch};

#[inline(always)]
fn sigmoid_impl<S: Simd>(simd: S, x: [f32; 4]) -> [f32; 4] {
    let x_simd: f32x4<S> = x.simd_into(simd);
    (x_simd / (1.0 + x_simd * x_simd).sqrt()).into()
}

simd_dispatch!(sigmoid(level, rgba: [f32; 4]) -> [f32; 4] = sigmoid_impl);

fn main() {
    let level = Level::new();
    let rgba = [0.1, -0.2, 0.001, 0.4];
    println!("{:?}", sigmoid(level, rgba));
}
