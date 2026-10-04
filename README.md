# MK AI Video Upscaler — V1 Beta

Mobile-first browser-side AI video upscaler.

- WebGPU + ONNX Runtime Web
- Real-ESRGAN fixed 384 model
- Mediabunny for browser-side media decode/encode and MP4 output
- Audio is decoded/re-encoded locally when supported
- Max input duration: 30 seconds
- Fast mode: full-frame AI enhancement
- Detail mode: 384px tiled AI processing (slower)
- Output target: 2x, capped at 2160px long edge

## GitHub Pages
Upload `index.html`, `style.css`, `app.js`, and `README.md` to the repo root and publish from `main` / root.

## Important
The Fast mode uses a fixed-size model by fitting each source frame into the model's 384x384 input. Detail mode processes 384px tiles and is much slower on mobile. This is an early browser-only implementation; performance depends heavily on the phone GPU/browser and supported codecs.
