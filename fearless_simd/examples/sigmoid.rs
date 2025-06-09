// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{Level, Simd, SimdBase, SimdFloat, simd_dispatch};

#[inline(always)]
fn sigmoid_impl<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    let n = S::f32s::N;
    for (x, y) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let a = S::f32s::from_slice(simd, x);
        let b = a / (a * a + 1.0).sqrt();
        y.copy_from_slice(b.as_slice());
    }
}

simd_dispatch!(sigmoid(level, x: &[f32], out: &mut [f32]) = sigmoid_impl);

fn main() {
    let level = Level::new();
    let inp = [0.1, -0.2, 0.001, 0.4, 1., 2., 3., 4.];
    let mut out = [0.; 8];
    sigmoid(level, &inp, &mut out);
    println!("{out:?}");
}
