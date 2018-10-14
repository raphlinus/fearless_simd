//! Runtime detection of x86 and x86_64 capabilities.

use avx::AvxF32;
use combinators::SimdFnF32;
use traits::SimdF32;

pub enum SimdCaps {
    Avx(AvxF32),
    Fallback(f32),
    // TODO: other levels
}

pub enum SimdStream<A: Iterator<Item = AvxF32>, F: Iterator<Item = f32>>
{
    Avx(A),
    Fallback(F),
}

struct MapStream<S: SimdF32, A: Iterator<Item = S>, F: FnMut(S) -> S> {
    inner: A,
    f: F,
}

struct CountStream<S: SimdF32> {
    val: S,
    step: f32,
}

impl<S: SimdF32> Iterator for CountStream<S> {
    type Item = S;
    fn next(&mut self) -> Option<S> {
        let val = self.val;
        self.val = self.val + self.step;
        Some(val)
    }
}

impl<S: SimdF32, A: Iterator<Item = S>, F: FnMut(S) -> S> Iterator for MapStream<S, A, F> {
    type Item = S;
    fn next(&mut self) -> Option<S> {
        self.inner.next().map(|x| (self.f)(x))
    }
}

pub fn detect() -> SimdCaps {
    if is_x86_feature_detected!("avx") {
        unsafe { SimdCaps::Avx(AvxF32::create()) }
    } else {
        SimdCaps::Fallback(0.0)
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

#[inline]
fn collect_fallback<A: Iterator<Item = f32>>(mut iterator: A, obuf: &mut [f32]) {
    for i in (0..obuf.len()).step_by(1) {
        let x = iterator.next().unwrap();
        x.write_to_slice(&mut obuf[i..]);
    }
}

impl<A: Iterator<Item = AvxF32>, F: Iterator<Item = f32>> SimdStream<A, F>
{
    #[inline]
    pub fn collect(self, obuf: &mut [f32]) {
        match self {
            SimdStream::Avx(avx_iterator) => {
                assert!(obuf.len() % 8 == 0);
                // This is only unsafe to enable the target_feature.
                unsafe { collect_avx(avx_iterator, obuf); }
            }
            SimdStream::Fallback(iterator) => {
                collect_fallback(iterator, obuf);
            }
        }
    }

    pub fn map<FN: SimdFnF32>(self, mut f: FN)
        -> SimdStream<impl Iterator<Item=AvxF32>, impl Iterator<Item=f32>>
    {
        match self {
            SimdStream::Avx(inner) => {
                SimdStream::Avx(
                    MapStream { inner, f: move |x| f.call(x) }
                )
            }
            SimdStream::Fallback(inner) => {
                SimdStream::Fallback(
                    MapStream { inner, f: move |x| f.call(x) }
                )
            }
        }
    }
}

pub fn count(init: f32, step: f32)
    -> SimdStream<impl Iterator<Item=AvxF32>, impl Iterator<Item=f32>>
{
    match detect() {
        SimdCaps::Avx(cap) => SimdStream::Avx(CountStream {
            val: cap.steps() * step + init,
            step: step * (cap.width() as f32),
        }),
        SimdCaps::Fallback(cap) => SimdStream::Fallback(CountStream {
            val: cap.steps() * step + init,
            step: step * (cap.width() as f32),
        }),
    }
}
