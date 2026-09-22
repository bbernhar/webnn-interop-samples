# WebNN Interop Samples

Samples are organized by workload, one folder per sample:

- `super_resolution_10/`
  - `super_resolution_10.html`
  - `README.md`
- `super_resolution_ssr/`
  - `super_resolution_ssr.html`
  - `README.md`
- `webgpu_stress/`
  - `webgpu_stress.html`
  - `README.md`

## Hosted Demos (GitHub Pages)

Run the workloads directly in a browser from the hosted pages:
- Super Resolution 10: https://bbernhar.github.io/webnn-interop-samples/super_resolution_10/super_resolution_10.html
- Simple SR: https://bbernhar.github.io/webnn-interop-samples/super_resolution_ssr/super_resolution_ssr.html
- WebGPU Stress: https://bbernhar.github.io/webnn-interop-samples/webgpu_stress/webgpu_stress.html

## Requirements

- A Chromium build with WebGPU enabled.
- For interop scenarios, WebNN support and exportable tensor APIs.
- Suggested launch flag: `--enable-features=WebMachineLearningNeuralNetwork`.

## Run

Open each sample's `README.md` for workload-specific usage and notes.
