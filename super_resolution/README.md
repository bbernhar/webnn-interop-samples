# Super Resolution Sample

This sample benchmarks WebNN and WebGPU zero-copy interop using a super resolution style pipeline.

## File

- `super_resolution.html`

## What it measures

- Interop only mode: measures the WebNN <-> WebGPU handoff cycle only (`exportToGPU`, `destroy`, `dispatch`).
- Full pipeline mode: measures end-to-end frame processing (import, preprocess, dispatch, render, present).

## Requirements

- Chromium build with WebNN and WebGPU enabled.
- Support for exportable tensors (`createExportableTensor` and `exportToGPU`).
- Suggested launch flag: `--enable-features=WebMachineLearningNeuralNetwork`.

## Run

1. Open `super_resolution.html` in Chromium (or serve this folder statically).
2. Select model format, WebNN device, GPU adapter, and measurement mode.
3. For full pipeline mode, select a source (synthetic, camera, or local video).
4. Click Start.

## Notes

- The default graph is a minimal passthrough intended to isolate interop overhead.
- Output is rendered directly from exported GPU buffers without CPU staging.
