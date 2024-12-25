// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::WithSimd;

struct Foo;

impl WithSimd for Foo {
    type Output = f32;

    fn with_simd<S: fearless_simd::Simd>(self, simd: S) -> Self::Output {
        let a = simd.splat_f32x4(42.0);
        let b = a + a;
        b[0]
    }
}

fn main() {
    let level = fearless_simd::Level::new();
    let x = level.dispatch(Foo);
    println!("level = {level:?}, x = {x}");
}
