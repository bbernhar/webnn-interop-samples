# WebNN + WebGPU Interop Benchmarks

Self-contained, single-file browser benchmarks exercising the **zero-copy
interop path between WebNN (`navigator.ml`) and WebGPU (`navigator.gpu`)** in
Chromium. Each one is a single `.html` file with no build step, no bundler, and
no external assets — open it directly (`file://`) or serve it statically.

## Requirements

- A Chromium build with WebNN enabled and the WebGPU **exportable-tensor
  interop** APIs (`MLContext.createExportableTensor` / `MLContext.exportToGPU`).
- WebGPU support (a working `navigator.gpu` adapter).
- Suggested launch flag: `--enable-features=WebMachineLearningNeuralNetwork`.

These are experimental APIs; behavior depends on your local build and GPU/NPU
backend.

---

## `video_super_resolution.html`

A real-time **Video Super Resolution (VSR)**-style pipeline that keeps all frame
data on the GPU/NPU and never copies pixels back to the CPU. It benchmarks the
per-frame latency of the full zero-copy handoff between a WebGPU compute/render
pipeline and a WebNN graph dispatch.

### What it does

Per video frame:

```
Video frame
  → importExternalTexture        (hardware YUV→RGB; sampled in the compute pass)
  → [preprocess compute shader]  tile the frame, swizzle RGB→BGR,
                                 pack into CHW or HWC layout in the input tensor
  → ml.exportToGPU(inputTensor)  →  gpuBuffer.destroy()   (ownership → WebNN)
  → ml.dispatch(graph, {input}, {output})                 (WebNN dispatch)
  → ml.exportToGPU(outputTensor)                           (ownership → WebGPU)
  → [render fragment shader]     read the output buffer directly: BGR→RGBA +
                                 tile de-stitch → canvas (no intermediate copy)
```

The **exportable-tensor lifecycle** is the heart of it: an `MLTensor` allocated
with `createExportableTensor` is temporarily exported to a `GPUBuffer` for the
WebGPU shaders to read/write, then `destroy()`-ed to return exclusive ownership
to WebNN for the graph dispatch — with no intermediate CPU staging. The render
pass reads the exported output buffer straight from the fragment shader, so
there is no separate stitch/compose copy.

### Key concepts illustrated

- **Zero-copy interop**: a single `MLTensor` is reused every frame; export /
  destroy replaces per-frame buffer allocation.
- **Tiling**: the frame is partitioned into non-overlapping tiles matching the
  model input size (batched into one dispatch), with clamp (edge) padding on
  boundary tiles and cropping on write-back.
- **Layout & channel-order adapters**: switches between CHW and HWC packing and
  performs the RGB↔BGR swizzle in the preprocess compute shader and the render
  fragment shader.
- **Model profiles** (from the feature spec), selectable in the UI:

  | Profile | Tile (H×W) | Layout | Data type |
  |---------|-----------|--------|-----------|
  | A       | 400 × 400 | CHW    | float32   |
  | B       | 400 × 400 | HWC    | float32   |
  | C       | 256 × 288 | HWC    | float32   |

### The WebNN graph

The graph is a **minimal passthrough** built at runtime with `MLGraphBuilder` —
a single multiply-by-1 (`output = input * 1`). Because the math is trivial, the
measured latency reflects the **interop/dispatch overhead** (`exportToGPU` /
`destroy` / `dispatch` synchronization) rather than model compute, which
isolates the cost of the zero-copy path itself. To benchmark a real workload,
swap `buildGraph()` for an actual SR network (and size the output tensor for its
output tile dimensions).

> Scope notes: all profiles run in **float32** (quantized uint8 profiles would
> follow the same path with byte-packed tensors — see the feature spec TODO).
> The passthrough graph is same-resolution, so the output matches the input
> frame size; the output canvas is set to that resolution.

### Measurement modes

The **Measure** control selects what is timed:

- **Interop only (WebNN⇄WebGPU)** *(default)* — isolates just the interop
  handshake. Per sample it repeats `INTEROP_REPS` (64) cycles of
  `exportToGPU(input)` → `destroy` → `dispatch` → `exportToGPU(output)` →
  `destroy`, with **no** video import, preprocess, render, or present. All reps
  reuse the same tensors so the dispatches serialize; a single 4-byte dependent
  copy of the final output makes `onSubmittedWorkDone()` wait for the whole
  chain. Reports **µs per interop cycle**. The canvas does not update in this
  mode (the render/present path is intentionally excluded).

- **Full pipeline** — times the complete per-frame path
  (import → preprocess → export/destroy → dispatch → export → render → present)
  and reports **`ms/frame`**. Use this for the end-to-end VSR latency; note it
  includes video-decode sync, compute passes, and compositor present, so it is
  *not* a clean measure of interop cost.

Both loops run uncapped (not driven by `requestAnimationFrame`) and pace on
`device.queue.onSubmittedWorkDone()`, so the numbers are hardware-completion
latencies rather than display-refresh-limited frame rates.

> Switching the **Measure** mode requires **Stop → Start** (Full pipeline needs
> a video source that Interop-only mode does not start).

### Usage

1. Open `video_super_resolution.html` in a compatible Chromium build.
2. Pick a **Measure** mode, **Profile**, **WebNN device** (`gpu` / `npu`), and —
   for Full pipeline — a **Source** (built-in synthetic animation, camera, or a
   local video file).
3. Click **Start**. The on-screen log reports context creation, interop support,
   resource/geometry setup, and any errors; the readout shows the metric for the
   selected mode (µs/interop-cycle or ms/frame).

---

## `webgpu_stress.html`

A WebGPU stress test that scales triangle count and storage-buffer memory
pressure via a slider.
