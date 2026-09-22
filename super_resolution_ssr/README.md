# Super Resolution SSR

This workload follows the `super_resolution_10` WebNN/WebGPU zero-copy pipeline
and replaces its real-SR graph with the topology of
`sr_x3_light_240x360_nchw_fp16.tflite`. The graph is rebuilt with
`MLGraphBuilder`; the TFLite file itself is not loaded by WebNN.

## File

- `super_resolution_ssr.html`

## Graph

- Input: FP16 NCHW `[1, 1, 240, 360]`
- Output: FP16 NCHW `[1, 1, 720, 1080]`
- 7 `conv2d`
- 9 `prelu`
- 3 residual `add`
- Bilinear `resample2d`
- 2 pixel-shuffle stages represented by `reshape -> transpose -> reshape`

Constants are generated from a deterministic random seed, so this validates
graph support and pipeline execution rather than numerical equivalence with the
original TFLite weights.

The full pipeline preserves the zero-copy workload structure:

- Video/camera/test-pattern input through a WebGPU external texture.
- GPU preprocessing into exportable FP16 WebNN tensors.
- One batched WebNN dispatch for B, G, and R planes across all tiles.
- GPU rendering directly from the exported output tensor without CPU staging.
- Interop-only mode for isolating tensor ownership and dispatch overhead.

## Run

Serve the repository over HTTP and open:

```text
http://localhost:<port>/super_resolution_ssr/super_resolution_ssr.html
```

Select a WebNN device and WebGPU adapter, then start the synthetic, camera, or
local-video pipeline. The SSR graph is always used.

## Requirements

- Chromium with WebNN enabled.
- FP16 and exportable-tensor support on the selected WebNN backend.
- Suggested launch flag: `--enable-features=WebMachineLearningNeuralNetwork`.
