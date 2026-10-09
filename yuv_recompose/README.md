# YUV Plane Recompose

Correctness check for the raw-plane interop path: split a 1080p YUV
`VideoFrame` into Y and UV, recompose it with an app-side shader, and compare
against the browser's own YUV-to-RGB conversion.

## File

- `yuv_recompose.html`

## Pipeline

- **Source:** `importExternalTexture(frame)` sampled into an `rgba8unorm`
  target (browser conversion, `srgb` destination).
- **Destination:** `frame.copyTo` extracts Y and UV. Y round-trips through a
  WebNN identity graph via exportable tensors and `exportToGPU`; UV is
  uploaded to an `rg8unorm` texture. An app shader recomposes RGB using
  `VideoFrame.colorSpace` (matrix, range, transfer, primaries) and midpoint
  chroma siting.
- **Delta:** per-channel absolute difference image, mask view, and statistics
  (max/mean, histogram, worst pixel with both RGB values).

The page also reports whether the Y plane survived the WebNN round-trip
exactly.

## Options

| URL param | Values | Meaning |
|---|---|---|
| `device` | `gpu`, `npu` | WebNN device for the Y round-trip |
| `adapter` | `default`, `high-performance`, `low-power`, `fallback` | WebGPU adapter |
| `source` | `test`, `image`, `video`, `camera` | 1080p test pattern, local image (cover-scaled to 1080p), first frame of a local video, or a camera snapshot |
| `path` | `decode`, `cpu` | Still sources only: WebCodecs encode+decode, or a CPU-built BT.709 limited NV12 frame |
| `ypath` | `webnn`, `direct` | Y through WebNN, or a plain `GPUBuffer` to isolate composition error |
| `filter` | `linear`, `nearest` | Sampler filter used by both paths |

`autostart=1` runs on load for `test` and `camera` sources.

## Notes

- Lossy encode/decode does not affect the comparison: both paths consume the
  same decoded frame.
- CPU-backed frames (`path=cpu`) take Chromium's RGB-intermediate external
  texture path; hardware-decoded frames may take the zero-copy NV12 path.
- HDR transfers (PQ/HLG) are not modeled by the app shader and are flagged.

## Run

```text
http://localhost:<port>/yuv_recompose/yuv_recompose.html?source=test&autostart=1
```
