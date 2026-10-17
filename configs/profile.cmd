@echo off
rem 10GB VRAM (RTX 3080) + 64GB RAM Multi-Engine Profile
rem Supports 4 Engines: LTX-Video, Wan 2.1 14B GGUF, HunyuanVideo GGUF, MiniMax-H3
rem Uses --lowvram, --reserve-vram 1.0, and dynamic RAM offloading for 10GB GPU
set COMFY_HOST=http://127.0.0.1:8188
set COMFY_DIR=C:\AI\H3\ComfyUI-0.36.0
set COMFY_EXTRA_ARGS=--lowvram --reserve-vram 1.0 --disable-pinned-memory --disable-async-offload

rem Default Active Engine (h3 | wan2.1)
if "%VIDEO_ENGINE%"=="" set VIDEO_ENGINE=h3

rem Model & VAE Configurations
set H3_MODEL=minimax_h3_fl2va_pruned-Q4_K.gguf
set H3_AUDIO_VAE=minimax_h3_audio_vae_fp32.safetensors
set H3_VIDEO_VAE=minimax_h3_video_vae_fp16.safetensors
set H3_LORA=minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors

set WAN_MODEL=wan2.1-i2v-14b-480p-Q4_K_M.gguf
set WAN_VAE=wan_2.1_vae.safetensors

set H3_BACKEND=local
echo [profile] 10GB VRAM (RTX 3080) + 64GB RAM: Multi-Engine active [%VIDEO_ENGINE%], ComfyUI args: %COMFY_EXTRA_ARGS%
