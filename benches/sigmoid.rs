//! Sinewave generation example, as benchmark.

#![feature(test)]

extern crate test;

extern crate fearless_simd;

use fearless_simd::{count, GeneratorF32, SimdF32, SimdFnF32};

use test::Bencher;

/// Generates a smooth tanh approximation with 2e-4 accuracy.
struct Tanh5Fn;
impl SimdFnF32 for Tanh5Fn {
    #[inline(always)]
    fn call<S: SimdF32>(&mut self, x: S) -> S {
        let xx = x * x;
        let x = x + ( xx * 0.00985468 +0.16489087) * (x * xx);
        x * (x * x + 1.0).rsqrt16()
    }
}

// TODO: this generates a linear ramp but should be adapted
// to map an input buffer.
fn gen_tanh(freq: f32, obuf: &mut [f32]) {
    count(0.0, freq).map(Tanh5Fn).collect(obuf);
}

#[bench]
fn tanh(b: &mut Bencher) {
    let mut obuf = [0.0; 64];
    b.iter(|| gen_tanh(test::black_box(0.1), &mut obuf));
}
