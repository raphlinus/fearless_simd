#![feature(target_feature_11)]

use fearless_simd::simd_dispatch;

simd_dispatch!{
    #[levels = (neon_fp16, neon)]
    pub foo(x: f32) -> u32 {
        println!("body of function, neon = {HAS_NEON}");
        let a = simd::f32s::splat(x);
        if HAS_NEON_FP16 {
            let b = simd::f16s::splat_f32_const(3.14);
            // This doesn't compile, as it's generated in versions other
            // than the fp16 one.
            //let c = simd::f16_8::add(b, b);
        }
        a.to_array()[0] as u32
    }
}

fn main() {
    let a = foo(42.0);
    println!("a = {a}");
}

