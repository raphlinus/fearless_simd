//! Sinewave generation example

extern crate fearless_simd;

use fearless_simd::{count, GeneratorF32, SimdF32, SimdFnF32, ThunkF32, run_f32};

#[inline(always)]
fn sin9_shaper<S: SimdF32>(x: S) -> S {
    let c0 =   6.28308759;
    let c1 = -41.33318707;
    let c2 =  81.39900205;
    let c3 = -74.66884436;
    let c4 =  33.15324345;

    let a = (x - x.round()).abs() - 0.25;
    let a2 = a * a;
    ((((a2 * c4 + c3) * a2 + c2) * a2 + c1) * a2 + c0) * a    
}

struct Sin9Fn;
impl SimdFnF32 for Sin9Fn {
    #[inline]
    fn call<S: SimdF32>(&mut self, x: S) -> S {
        sin9_shaper(x)
    }
}

fn gen_sinewave(freq: f32, obuf: &mut [f32]) {
    count(0.25, freq).map(Sin9Fn).collect(obuf);
}

struct Sin9Thunk<'a> {
    freq: f32,
    obuf: &'a mut [f32; 32],
}

impl<'a> ThunkF32 for Sin9Thunk<'a> {
    #[inline]
    fn call<S: SimdF32>(self, cap: S) {
        let mut phase = cap.steps() * self.freq + 0.25;
        for i in (0..self.obuf.len()).step_by(cap.width()) {
            sin9_shaper(phase).write_to_slice(&mut self.obuf[i..]);
            phase = phase + self.freq * (cap.width() as f32);
        }
    }
}

fn main() {
    let mut obuf = [0.0; 32];
    gen_sinewave(0.1, &mut obuf);

    let mut obuf2 = [0.0; 32];
    run_f32(Sin9Thunk {
        freq: 0.1,
        obuf: &mut obuf2,
    });
    for i in 0..obuf.len() {
        println!("{} {}", obuf[i], obuf2[i]);
    }
}
