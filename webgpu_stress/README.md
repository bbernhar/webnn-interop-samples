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

1. Open the hosted page: https://bbernhar.github.io/webnn-interop-samples/webgpu_stress/webgpu_stress.html
2. Or run a local static server and open `http://.../webgpu_stress.html`.
3. Click Start Stress.
4. Increase or decrease the stress slider to change memory pressure.

## Notes

- Maximum stress value is derived from device limits at runtime.
- If WebNN is unavailable, the sample continues in WebGPU-only mode.
