/// A token for fallback SIMD.
#[derive(Clone, Copy, Debug)]
pub struct Fallback {
    _private: (),
}

impl Fallback {
    /// Create a SIMD token.
    #[inline]
    pub fn new() -> Self {
        Self { _private: () }
    }
}