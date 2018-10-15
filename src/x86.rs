//! Runtime detection of x86 and x86_64 capabilities.

use avx::AvxF32;
use combinators::SimdFnF32;
use traits::SimdF32;

pub trait GeneratorF32: Sized {
    type IterF32: Iterator<Item=f32>;
    type IterAvx: Iterator<Item=AvxF32>;
    fn gen_f32(self, cap: f32) -> Self::IterF32;
    fn gen_avx(self, cap: AvxF32) -> Self::IterAvx;

    #[inline]
    fn map<F>(self, f: F) -> F32Map<Self, F>
        where Self: Sized, F: SimdFnF32
    {
        F32Map { inner: self, f }
    }

    #[inline]
    fn collect(self, obuf: &mut [f32]) {
        if is_x86_feature_detected!("avx") {
            unsafe { collect_avx(self, obuf); }
        } else {
            let mut iter = self.gen_f32(0.0);
            for i in (0..obuf.len()).step_by(1) {
                let x = iter.next().unwrap();
                x.write_to_slice(&mut obuf[i..]);
            }
        }
    }
}

#[target_feature(enable = "avx")]
unsafe fn collect_avx<G: GeneratorF32>(gen: G, obuf: &mut [f32]) {
    let mut iter = gen.gen_avx(AvxF32::create());
    for i in (0..obuf.len()).step_by(8) {
        let x = iter.next().unwrap();
        x.write_to_slice(&mut obuf[i..]);
    }
}

pub struct F32Map<G: GeneratorF32, F: SimdFnF32> {
    inner: G,
    f: F,
}

pub struct F32MapIter<S: SimdF32, I: Iterator<Item = S>, F: SimdFnF32> {
    inner: I,
    f: F,
}

impl<S, I, F> Iterator for F32MapIter<S, I, F>
    where S: SimdF32, I: Iterator<Item = S>, F: SimdFnF32
{
    type Item = S;
    fn next(&mut self) -> Option<S> {
        self.inner.next().map(|x| self.f.call(x))
    }
}

impl<G: GeneratorF32, F: SimdFnF32> GeneratorF32 for F32Map<G, F> {
    type IterF32 = F32MapIter<f32, G::IterF32, F>;
    type IterAvx = F32MapIter<AvxF32, G::IterAvx, F>;
    fn gen_f32(self, cap: f32) -> Self::IterF32 {
        F32MapIter { inner: self.inner.gen_f32(cap), f: self.f }
    }
    fn gen_avx(self, cap: AvxF32) -> Self::IterAvx {
        F32MapIter { inner: self.inner.gen_avx(cap), f: self.f }
    }
}


pub struct CountGen {
    init: f32,
    step: f32,
}

pub struct CountStream<S: SimdF32> {
    val: S,
    step: f32,
}

#[inline]
pub fn count(init: f32, step: f32) -> CountGen {
    CountGen { init, step }
}

impl CountGen {
    #[inline]
    fn gen<S: SimdF32>(self, cap: S) -> CountStream<S> {
        CountStream {
            val: cap.steps() * self.step + self.init,
            step: self.step * (cap.width() as f32),
        }        
    }
}

// Note that the following is 100% boilerplate and could be easily
// generated by macro.
//
// Also parametrized `gen` could be a trait item when rust #44265
// lands.
impl GeneratorF32 for CountGen {
    type IterF32 = CountStream<f32>;
    type IterAvx = CountStream<AvxF32>;
    #[inline]
    fn gen_f32(self, cap: f32) -> CountStream<f32> {
        self.gen(cap)
    }
    #[inline]
    fn gen_avx(self, cap: AvxF32) -> CountStream<AvxF32> {
        self.gen(cap)
    }
}

impl<S: SimdF32> Iterator for CountStream<S> {
    type Item = S;
    #[inline]
    fn next(&mut self) -> Option<S> {
        let val = self.val;
        self.val = self.val + self.step;
        Some(val)
    }
}

