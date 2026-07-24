# WebNN Interop Samples

Samples are organized by workload, one folder per sample:

- `super_resolution/`
  - `super_resolution.html`
  - `README.md`
- `webgpu_stress/`
  - `webgpu_stress.html`
  - `README.md`

## Hosted Demos (GitHub Pages)

Run the workloads directly in a browser from the hosted pages:
- Super Resolution: https://bbernhar.github.io/webnn-interop-samples/super_resolution/super_resolution.html
- WebGPU Stress: https://bbernhar.github.io/webnn-interop-samples/webgpu_stress/webgpu_stress.html

## Requirements

- A Chromium build with WebGPU enabled.
- For interop scenarios, WebNN support and exportable tensor APIs.
- Suggested launch flag: `--enable-features=WebMachineLearningNeuralNetwork`.

## Run

Open each sample's `README.md` for workload-specific usage and notes.
