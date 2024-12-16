# Towards "fearless SIMD"

This crate proposes an experimental way to use SIMD intrinsics reasonably safely, using the new [target_feature 1.1] feature in Rust. It is currently nightly-only until that feature is stabilized, but that is expected to happen soon.

A [much earlier version][fearless_simd 0.1.1] of this crate experimented with an approach that tried to accomplish safety in safe Rust as of 2018, using types that witnessed the SIMD capability of the CPU. There is a blog post, [Towards fearless SIMD], that wrote up the experiment. That approach couldn't quite be made to work, but was an interesting exploration at the time. A practical development along roughly similar lines is the [pulp] crate.

Some code has been cut and pasted from the [half] crate, which is released under identical license. That crate is also an optional dependency, for more full f16 support.

[pulp]: https://crates.io/crates/pulp
[target_feature 1.1]: https://github.com/rust-lang/rfcs/pull/2396
[Towards fearless SIMD]: https://raphlinus.github.io/rust/simd/2018/10/19/fearless-simd.html
[fearless_simd 0.1.1]: https://crates.io/crates/fearless_simd/0.1.1
[half]: https://crates.io/crates/pulp
