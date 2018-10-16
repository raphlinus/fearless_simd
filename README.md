# Towards "fearless SIMD"

This crate provides safe wrappers to make it easier to write SIMD code. It doesn't yet deliver on the promise of "fearless SIMD", but shows a potential path towards it.

It tries to solve these problems:

* Automatically detecting the CPU level and running the best code.

* No `unsafe` required to use.

* Access to advanced SIMD primitives such as rounding and approximate reciprocal.

* Works with stable Rust.

* Can be portable across multiple architectures (but only x86 currently supported by stable rust - might make arm optional so it can be compiled on nightly).

It is limited in scope:

* A focus on f32.

* Mostly (but not entirely) maps and generators for 1D unstructured data.

* No attempt to support aligned load/store. On modern CPUs, unaligned SIMD access is quite performant, and alignment is a very significant burden on the coder.

It's possible the ideas in this crate could be extended to more applications.

## Caveats and future prospects

I ran into a number of limitations of current Rust while writing this. I think it's likely some of these will improve. Partly why I'm publishing this crate is to shine a light on where more work might be useful.

Getting inlining wrong will trigger rust-lang/rust#50154. That said, the `GeneratorF32` trait is designed so that iterator creation happens inside a target_feature wrapper, which should both reduce the chance of triggering that bug, and improve code quality.

That bug is not the only inlining misfeature; the `#[cfg(target_feature)]` macro is resolved too early and does not report whether the feature is enabled if the function is inlined. This is discussed a bit in a [rust-internals thread](https://internals.rust-lang.org/t/packed-simd-cfg-target-feature-does-not-play-well-with-target-feature/8115). It's not clear to me that the [proposed approach forward](https://internals.rust-lang.org/t/using-run-time-feature-detection-in-core/8419) really fixes the issue, because runtime feature doesn't always match `[target_feature(enabled)]`. For example, runtime feature detection may show that AVX-512 is available, but the user may choose to use only AVX2 for [performance reasons](https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/).

I wanted to make the `GeneratorF32` trait processor-independent and fully generic. In other words, I'd like to be able to write this:

```rust
pub trait GeneratorF32: Sized {
    type Iter<S: SimdF32>: Iterator<Item=S>;
    fn gen<S>(self, cap: S) -> Self::Iter<S>;
}
```
This feature is in the works: generic associated types] (rust-lang/rust#44265).

If `x` has a `SimdF32` value, it is possible to write, say, `x + 1.0`, but at the moment `1.0 + x` does not work. The relevant trait bounds do work if added to the `SimdF32` trait, but it would force a lot of boilerplate into client implementations, due to rust-lang/rust#23856. That looks like it might get improved when Chalk lands.

I use the `SimdFnF32` trait to represent a function is generic in the actual SIMD type. Even better would be something like this:

```rust
pub trait GeneratorF32: Sized {
    fn map<F>(self, f: F) where F: for<S: SimdF32> Fn(S) -> S;
}
```

Currently the `for<>` syntax works for higher-ranked lifetimes but not higher-ranked generics in general. I'm not sure this will ever happen, but it shows a potential real-world example for why these exotic higher-ranked types might be useful.

## Comparisons with other approaches

There is a lot of inspiration from [faster], which has similar goals. However, faster relies on compile-time feature determination and doesn't seem to be able to switch at runtime.

The safe wrappers are inspired by [packed_simd]. That crate is more ambitious for exposing a larger fragment of SIMD, but leaves the runtime feature detection to the user.

The C/C++ ecosystem has done quite a bit of work in this space. They have a fairly sophisticated [Function Multi Versioning](https://lwn.net/Articles/691932/) mechanism, with runtime detection resolved by the dynamic loader. To a large extent, this crate tries to gain some of the benefits of that, without requiring extensions to the language or implementation. However, this crate "uses up" a dimension or two of the polymorphic type space, so it's a tradeoff to be examined.

## Benchmarks

These aren't meant to be rigorous, but should give a general impression of performance. The particular benchmark is generation of a sinewave with less than -100dB disortion, and times are given in ns to generate 64 samples.

| CPU       | simd level      | time  |
| --------- | --------------- | ----: |
| i7 7700HQ | AVX             |   30  |
| "         | SSE 4.2         |   49  |
| "         | scalar fallback |  344  |
| "         | sin() scalar    |  506  |
| i5 430M   | SSE4.2          |  303  |
| "         | scalar fallback |  717  |
| "         | sin() scalar    | 1690  |

## Acknowledgements

Errors (including in judgment for going down this path) are my own, but I've benefitted from discussions with many people, including with James McCartney, Andrew Gallant ([burntsushi](https://github.com/BurntSushi)), [talchas](https://github.com/talchas), Colin Rofls, and Alex Crichton.

[faster]: https://github.com/AdamNiederer/faster

[packed_simd]: https://github.com/rust-lang-nursery/packed_simd
