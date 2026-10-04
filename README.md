# MK AI Video Upscaler — V0

Mobile-first browser prototype.

## What this version proves

- Video can be selected locally.
- Duration is checked (10 sec prototype limit).
- WebGPU availability is checked.
- Real-ESRGAN general x4 ONNX model is loaded in the browser.
- One video frame is processed locally by the AI model.
- No video upload/backend is used.

## Run

Because browsers restrict some local-file behavior, serve this folder over HTTPS or localhost.

For GitHub Pages, upload these files to a repository and enable Pages.

## Next milestone

1. Decode the whole video frame-by-frame.
2. AI upscale each frame with tiling/chunking to control mobile memory.
3. Encode output in-browser.
4. Preserve original audio.
5. Add progress/cancel/resume handling.
6. Raise duration limit from 10 sec to 30 sec after mobile stability testing.

## Technical basis

ONNX Runtime Web supports WebGPU in current Chromium-based Android browsers. WebCodecs provides browser-native video frame decode/encode primitives. The selected Real-ESRGAN ONNX model is a ~5 MB general-purpose 4x model.
