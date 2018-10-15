//! Sinewave generation example

extern crate fearless_simd;

use fearless_simd::{count, GeneratorF32, SimdF32, SimdFnF32};

struct Sin9Fn;
impl SimdFnF32 for Sin9Fn {
    fn call<S: SimdF32>(&mut self, x: S) -> S {
        let c0 =   6.28308759;
        let c1 = -41.33318707;
        let c2 =  81.39900205;
        let c3 = -74.66884436;
        let c4 =  33.15324345;

        let a = (x - x.round()).abs() - 0.25;
        let a2 = a * a;
        ((((a2 * c4 + c3) * a2 + c2) * a2 + c1) * a2 + c0) * a
    }
}

fn gen_sinewave(freq: f32, obuf: &mut [f32]) {
    //count(0.25, freq).map(Sin9Fn).collect(obuf);
    count(0.25, freq).map(Sin9Fn).collect(obuf);
}

fn main() {
    let mut obuf = [0.0; 32];
    gen_sinewave(0.1, &mut obuf);
    for i in 0..obuf.len() {
        println!("{}", obuf[i]);
    }
}
