//! Matrix example

extern crate fearless_simd;

use fearless_simd::{F32x4, ThunkF32x4, run_f32x4};

struct Iir<'a> {
    ibuf: &'a [f32; 32],
    obuf: &'a mut [f32; 32],
    coefs: [[f32; 4]; 4],
}

impl<'a> ThunkF32x4 for Iir<'a> {
    #[inline]
    fn call<S: F32x4>(self, cap: S) {
        let c0 = cap.new(self.coefs[0]);
        let c1 = cap.new(self.coefs[1]);
        let c2 = cap.new(self.coefs[2]);
        let c3 = cap.new(self.coefs[3]);
        let mut state = cap.new([0.0, 0.0, 0.0, 0.0]);
        for i in (0..self.ibuf.len()).step_by(2) {
            let x0 = self.ibuf[i];
            let x1 = self.ibuf[i + 1];
            state = c0 * x0 + c1 * x1 + c2 * state.as_vec()[2] + c3 * state.as_vec()[3];
            self.obuf[i] = state.as_vec()[0];
            self.obuf[i + 1] = state.as_vec()[1];
        }
    }
}

fn main() {
    let coefs = [[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let ibuf = [0.0; 32];
    let mut obuf = [0.0; 32];
    run_f32x4(Iir {
        ibuf: &ibuf,
        obuf: &mut obuf,
        coefs
    });
    for i in 0..obuf.len() {
        println!("{}", obuf[i]);
    }
}
