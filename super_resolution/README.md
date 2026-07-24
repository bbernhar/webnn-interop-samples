# Super Resolution Sample

This sample benchmarks WebNN and WebGPU zero-copy interop using a super resolution style pipeline.

## File

- `super_resolution.html`

## What it measures

- Interop only mode: measures the WebNN <-> WebGPU handoff cycle only (`exportToGPU`, `destroy`, `dispatch`).
- Full pipeline (no SR) mode: measures end-to-end frame processing (import, preprocess, dispatch, render, present) with a minimal passthrough graph.
- Full pipeline (real SR) mode: same end-to-end path but the WebNN dispatch runs a real ESPCN super-resolution model (see below).

## Requirements

- Chromium build with WebNN and WebGPU enabled.
- Support for exportable tensors (`createExportableTensor` and `exportToGPU`).
- Suggested launch flag: `--enable-features=WebMachineLearningNeuralNetwork`.

## Run

1. Open `super_resolution.html` in Chromium (or serve this folder statically).
2. Select model format, WebNN device, GPU adapter, and measurement mode.
3. For full pipeline modes, select a source (synthetic, camera, or local video).
4. Click Start.

## Real SR model (ESPCN)

The "Full pipeline (real SR)" mode runs [super-resolution-10](https://huggingface.co/onnxmodelzoo/super-resolution-10)
(ESPCN / Sub-Pixel CNN): `conv1(1->64, 5x5) -> relu -> conv2(64->64, 3x3) -> relu
-> conv3(64->32, 3x3) -> relu -> conv4(32->9, 3x3) -> pixel-shuffle(x3)`, upscaling
each `224x224` tile to `672x672` (3x).

WebNN has no "load .onnx" API, so the graph is rebuilt op-by-op with
`MLGraphBuilder` using the original weights. `dump_weights.py` extracts the ONNX
initializers into:

- `sr10_weights.bin` - raw little-endian float32, concatenated.
- `sr10_weights.json` - manifest (name / shape / offset / count).

The page fetches both at runtime. To regenerate them from the source model:

```
py -m pip install onnx numpy
py dump_weights.py    # reads super-resolution-10.onnx
```

### How it differs from the original ONNX recipe

Same network topology and weights, but a different way of feeding and wrapping it:

| Aspect | Original ONNX recipe | This demo |
| --- | --- | --- |
| Channel fed to the net | Y only (RGB->YCbCr luma) | B, G, R each as separate 1-channel batch elements |
| Batch dim `N` | 1 image (Y) | `3 * numTiles` (three color planes x tiles), one batched dispatch |
| Color conversion | RGB->YCbCr and back | none - works directly in BGR |
| Chroma (Cb/Cr) | bicubic-upscaled, then merged | n/a - every channel goes through the network |
| Input source | resized 224x224 image, `/255` | video frame sampled per 224x224 tile (already 0-1 from the external texture) |
| Graph origin | loaded `.onnx` | rebuilt via `MLGraphBuilder` from the same weights |

Consequences:

- Avoids all YCbCr/bicubic math (reuses the existing BGR-plane packing) and
  triples the batched compute, which better exercises the interop path.
- Slightly off-distribution: the network was trained on luminance (Y), so applying
  it per B/G/R channel is not the color-accurate path the model card describes.
  Visually fine for a benchmark, but not identical output to the reference pipeline.
- The real-SR output tensor (`[3*numTiles, 1, 672, 672]` f32) is large; high
  resolutions can exceed the storage-buffer binding limit. Start at 640x360.

## Notes

- The default graph (no-SR modes) is a minimal passthrough intended to isolate interop overhead.
- Output is rendered directly from exported GPU buffers without CPU staging.
