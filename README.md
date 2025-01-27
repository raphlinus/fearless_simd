# Towards "fearless SIMD"

This crate proposes an experimental way to use SIMD intrinsics reasonably safely, using the new [target_feature 1.1] feature in Rust. It is currently nightly-only until that feature is stabilized, but that is expected to happen soon.

A [much earlier version][fearless_simd 0.1.1] of this crate experimented with an approach that tried to accomplish safety in safe Rust as of 2018, using types that witnessed the SIMD capability of the CPU. There is a blog post, [Towards fearless SIMD], that wrote up the experiment. That approach couldn't quite be made to work, but was an interesting exploration at the time. A practical development along roughly similar lines is the [pulp] crate.

Some code has been cut and pasted from the [half] crate, which is released under identical license. That crate is also an optional dependency, for more full f16 support.

## SIMD types

The SIMD types in this crate are a thin newtype around the corresponding array, for example `f32x4` is a newtype for `[f32; 4]`. These types are in the crate root and have a number of methods, including loading and storing, that do not require intrinsics. They are not aligned, as modern CPUs support unaligned access efficiently.

There are 

## Levels

A central idea is "levels," which represent a set of target features. Each level has a corresponding module. Each module for a level has a number of submodules, one for each type (though with an underscore instead of `x` to avoid name collision), with a large number of free functions for SIMD operations that operate on that type.

On aarch64, the levels are `neon` and `fp16`. The `fp16` level implies `neon`.

On x86-64, the currently supported level is `avx2`. This is actually short for the x86-64-v3 [microarchitecture level][x86-64 microarchitecture levels], which corresponds roughly to Haswell. The `avx512` level is also planned, which is x86-64-v4.

## Credits

This crate was inspired by [pulp], [std::simd], among others in the Rust ecosystem, though makes many decisions differently. It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.

The [half] code was mentioned above. The proc macro was strongly inspired by the [safe-arch-macro] in [rbrotli-enc].

[pulp]: https://crates.io/crates/pulp
[target_feature 1.1]: https://github.com/rust-lang/rfcs/pull/2396
[Towards fearless SIMD]: https://raphlinus.github.io/rust/simd/2018/10/19/fearless-simd.html
[fearless_simd 0.1.1]: https://crates.io/crates/fearless_simd/0.1.1
[half]: https://crates.io/crates/half
[x86-64 microarchitecture levels]: https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels
[std::simd]: https://doc.rust-lang.org/std/simd/index.html
[safe-arch-macro]: https://github.com/google/rbrotli-enc/blob/ce44d008ff1beff1eee843e808542d01951add45/safe-arch-macro/src/lib.rs
[rbrotli-enc]: https://github.com/google/rbrotli-enc
