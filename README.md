# Research Disclaimer
This project is an educational vulnerability proof of concept (PoC) to evaluate the robustness of AI watermarking (e.g., SynthID). It is intended to help developers and security teams build more resilient watermarking and provenance systems. Do not use this project for unauthorized removal or evasion of watermarks.

# SynthID Watermark Robustness Analysis (WebGPU/WGSL)
This PoC explores how mild signal filtering can affect frequency/spatial watermarks. It runs entirely in your browser using WebGPU and WGSL to apply a tunable, low-intensity spatial filter and measure visual persistence of embedded marks. Images are processed locally; nothing is sent off the device.

## Technical Implementation (WebGPU/WGSL)
- WebGPU compute pipeline applies a configurable box blur with adjustable radius and blend to perturb watermark signals while keeping perceptual similarity.
- WGSL shader performs per-pixel accumulation over a local neighborhood; outputs to a storage texture.
- GPU readback reconstructs the filtered image; if GPU output is zeroed (validation/format issues), a CPU fallback applies the same spatial filter.
- Frontend provides sliders for radius and blend to sweep perturbation strength.

## Security Findings
- Mild spatial filtering can attenuate or perturb embedded watermark signals without major perceptual change, indicating sensitivity to low-pass style operations.
- GPU vs. CPU parity check ensures that results are not artifacts of driver/implementation quirks.
- Highlights that watermark robustness should account for lightweight, commodity filtering operations accessible in browsers.

## Responsible Disclosure
This PoC illustrates gaps in current AI provenance and watermarking schemes (e.g., SynthID) when subjected to benign-looking signal filtering. Findings should be used to improve detection and resilience, not to bypass provenance. If you are responsible for a watermarking or provenance system, consider this PoC as feedback to harden your approach and align with emerging standards.

## Usage
1. Serve the static files (e.g., `python -m http.server 8000`).
2. Open `http://localhost:8000` in a WebGPU-capable browser (Chrome/Edge 121+ with WebGPU enabled).
3. Load an image, adjust radius and blend, then run the robustness test. Observe visual changes and assess watermark persistence downstream.

## License
Research and educational use only. See disclaimer above.
