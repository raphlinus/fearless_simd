<div align="center">

# Fearless SIMD Tests

</div>

This is a development-only crate for testing `fearless_simd`.


### Testing WebAssembly +simd128

Run browser tests with:

```sh
wasm-pack test --headless --chrome
```

Currently these tests only enforce that WASM SIMD and the fallback scalar implementations match when
run in the browser.
