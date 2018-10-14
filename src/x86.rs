//! Runtime detection of x86 and x86_64 capabilities.

use avx::AvxF32;
use combinators::SimdFnF32;
use traits::SimdF32;

pub enum SimdCaps {
    Avx(AvxF32),
    Fallback,
    // TODO: other levels
}

pub enum SimdStream<A: Iterator<Item = AvxF32>>
{
    Avx(A),
    Fallback, // TODO: populate iterator
}

struct AvxMap<A: Iterator<Item = AvxF32>, F: FnMut(AvxF32) -> AvxF32> {
    inner: A,
    f: F,
}

struct StreamCount<S: SimdF32> {
    val: S,
    step: f32,
}

impl<S: SimdF32> Iterator for StreamCount<S> {
    type Item = S;
    fn next(&mut self) -> Option<S> {
        let val = self.val;
        self.val = self.val + self.step;
        Some(val)
    }
}

impl<A: Iterator<Item = AvxF32>, F: FnMut(AvxF32) -> AvxF32> Iterator for AvxMap<A, F> {
    type Item = AvxF32;
    fn next(&mut self) -> Option<AvxF32> {
        self.inner.next().map(|x| (self.f)(x))
    }
}

pub fn detect() -> SimdCaps {
    if is_x86_feature_detected!("avx") {
        unsafe { SimdCaps::Avx(AvxF32::create()) }
    } else {
        SimdCaps::Fallback
    }
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn collect_avx<A: Iterator<Item = AvxF32>>(mut iterator: A, obuf: &mut [f32]) {
    for i in (0..obuf.len()).step_by(8) {
        let x = iterator.next().unwrap();
        x.write_to_slice(&mut obuf[i..]);
    }
}

impl<A: Iterator<Item = AvxF32>> SimdStream<A>
{
    #[inline]
    pub fn collect(self, obuf: &mut [f32]) {
        if let SimdStream::Avx(avx_iterator) = self {
            // This is only unsafe to enable the target_feature.
            unsafe { collect_avx(avx_iterator, obuf); }
        }
    }

    pub fn map<F: SimdFnF32>(self, mut f: F) -> SimdStream<impl Iterator<Item=AvxF32>> {
        match self {
            SimdStream::Avx(avx_iterator) => {
                SimdStream::Avx(
                    AvxMap { inner: avx_iterator, f: move |x| f.call(x) }
                )
            }
            _ => unimplemented!(),
        }
    }
}

pub fn count(init: f32, step: f32) -> SimdStream<impl Iterator<Item=AvxF32>> {
    match detect() {
        SimdCaps::Avx(avx) => SimdStream::Avx(StreamCount {
            val: avx.steps() * step + init,
            step: step * (avx.width() as f32),
        }),
        _ => unimplemented!(),
    }
}
