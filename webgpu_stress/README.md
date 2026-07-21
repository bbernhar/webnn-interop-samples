# WebGPU Stress Sample

This sample stress-tests WebGPU rendering with scalable storage-buffer pressure and optional WebNN interop activity.

## File

- `webgpu_stress.html`

## What it does

- Renders many triangles and updates workload continuously.
- Uses a stress slider to increase total storage-buffer memory usage.
- Optionally exercises a WebNN interop path that exports tensors to GPU buffers.

## Requirements

- Browser with WebGPU support (`navigator.gpu`).
- Optional WebNN support (`navigator.ml`) for the interop path.

## Run

1. Open `webgpu_stress.html` in a WebGPU-capable Chromium build.
2. Click Start Stress.
3. Increase or decrease the stress slider to change memory pressure.

## Notes

- Maximum stress value is derived from device limits at runtime.
- If WebNN is unavailable, the sample continues in WebGPU-only mode.
