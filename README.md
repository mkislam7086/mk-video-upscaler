# MK AI Video Upscaler V2

Mobile-first browser-side AI video upscaler. Video stays on the device.

## V2 speed improvements
- Turbo mode: AI every 3rd frame, output remains at original FPS.
- Fast mode: AI every 2nd frame.
- Detail mode: AI every frame.
- Reuses canvas/image buffers to reduce allocation overhead.
- Enables ONNX Runtime WebGPU graph capture when supported by the static model.
- Uses hardware/browser video encoding through WebCodecs/Mediabunny when available.

## Important
Turbo/Fast reduce AI inference count by reusing the latest AI frame for intermediate frames. This improves speed but can reduce temporal detail on fast motion. Detail is the quality mode.
