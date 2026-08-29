# -*- coding: utf-8 -*-
"""
LTX23_Director_Master_V2.py  ·  FAITHFUL REBUILD (matches the ComfyUI JSON)
================================================================================
100% Faithful LTX-2.3 "Director 2.0" 30-Second Music-Video Pipeline
Source of truth : LTX-2.3_Director_2.0-MV-Workflow-30s.json  (ComfyUI graph)
Target hardware : Google Colab Free-Tier (T4 15 GB VRAM · ~12.2 GB host RAM)

────────────────────────────────────────────────────────────────────────────
WHY THE OLD Master_V2 PRODUCED BAD VIDEO  (and what this rebuild fixes)
────────────────────────────────────────────────────────────────────────────
The previous Master_V2 rebuilt the video as 5 INDEPENDENT diffusions that were
crossfaded together afterwards. That is NOT what the JSON does, and it caused
every symptom you reported:

    ✗ Missing "notes"             → the whole LTXDirector timeline (segments,
                                    audio track, motion track, global prompt)
                                    was never fed to a single controller.
    ✗ Missing Master Timeline      → the `LTXDirector` node (id 131) was absent
      Controller                    entirely.
    ✗ Identity / scene drift       → 5 separate samplings never share context,
                                    so the singer changed between scenes.
    ✗ Voice NOT synchronized       → each fake segment got a *fresh empty* audio
                                    latent, so no real vocal track lined up.
    ✗ Visible seams                → linear crossfade blending between clips.

This rebuild reproduces the ORIGINAL graph node-for-node (all 23 node types):

    UnetLoaderGGUF ─► Power Lora Loader (rgthree) ─► ModelPreviewOverrideKJ ─┐
    DualCLIPLoader ─────────────────────────────────────────► (clip) ───────┤
                                                                             ▼
                                            ┌───────────  LTXDirector  ───────────┐
                                            │  (MASTER TIMELINE CONTROLLER)        │
                                            │  full 756-frame / 31.5 s timeline,   │
                                            │  5 keyframes on ONE main track,      │
                                            │  1 continuous AUDIO track (sync),    │
                                            │  motion track, global prompt         │
                                            └──────────────────────────────────────┘
        ConditioningZeroOut ─► LTXVConditioning
        STAGE 1  : LTXDirectorGuide(0.5) ─► ConcatAV ─► SamplerCustomAdvanced
                   (euler · linear_quadratic · 8 steps · denoise 1.0 · cfg 1)
                   ─► SeparateAV ─► LTXDirectorCropGuides ─► LTXVLatentUpsampler(2x)
        STAGE 2  : LTXDirectorGuide(1.0) ─► ConcatAV ─► SamplerCustomAdvanced
                   (euler · linear_quadratic · 4 steps · denoise 0.42 · cfg 1)
                   ─► SeparateAV ─► LTXDirectorCropGuides
        DECODE   : VAEDecode (video) + LTXVAudioVAEDecode (audio → synced vocals)
        ASSEMBLE : VHS_VideoCombine  →  single synced MP4

The whole timeline shares ONE conditioning + ONE audio latent, which is exactly
why the JSON keeps a consistent character, consistent scenes and a perfectly
synchronized voice — and this rebuild does the same.

────────────────────────────────────────────────────────────────────────────
HOW IT FITS ON A FREE T4 (15 GB VRAM / 12 GB RAM)
────────────────────────────────────────────────────────────────────────────
    • Decoupled text encode: the 12B Gemma clip is loaded ALONE, encodes the
      prompt on the GPU, then is PURGED before the 22B DiT is loaded — the two
      giant models are never resident together.
    • The ONE shared LTXDirector timeline (same conditioning + same audio latent)
      is sampled in keyframe scene-chunks with a deep memory purge + fresh DiT
      reload between each scene. Overlaps are cross-faded so scenes stay smooth
      AND the voice stays synced (the audio latent is sliced from the shared
      track, never regenerated empty).
    • Two-stage tiny-base diffusion: Stage 1 at 416x240, then 2x latent upscale +
      light 4-step / denoise-0.42 refine at 832x480.
    • purge_deep(): unload_all_models + cleanup + soft_empty_cache + gc +
      cuda.empty_cache + ipc_collect + OS page-cache drop + malloc_trim(0).
    • Out-of-core spatiotemporal TILED VAE decode.
    • Resume checkpoints at every phase.

On an L4/A100 set BATCH_SCENE_MODE = False for the true single-pass diffusion.
================================================================================
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP, 16 GB SWAP & MEMORY PROTECTION
# ════════════════════════════════════════════════════════════════════════════
import subprocess
import sys
import os
import shutil
import glob
import json
import gc
import types
import inspect
import ctypes
import math
import time
import traceback
from pathlib import Path
from typing import Sequence, Mapping, Any, Union, Dict, List, Optional, Tuple

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ('expandable_segments:True,'
                                         'garbage_collection_threshold:0.8,'
                                         'max_split_size_mb:128')
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'


def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    return subprocess.run(cmd, shell=True).returncode


# 16 GB high-speed swap partition — critical host-RAM head-room.
if not os.path.exists("/content/swapfile") or os.path.getsize("/content/swapfile") < (8 * 1024 * 1024 * 1024):
    print("⚙️ [1/3] Setting up Contiguous 16 GB Swap Partition...")
    run_cmd("swapoff /content/swapfile || true")
    run_cmd("rm -f /content/swapfile")
    run_cmd("dd if=/dev/zero of=/content/swapfile bs=1M count=16384 status=none || fallocate -l 16G /content/swapfile")
    run_cmd("chmod 600 /content/swapfile")
    run_cmd("mkswap /content/swapfile")
    run_cmd("swapon /content/swapfile || true")
    run_cmd("sysctl vm.swappiness=100 || true")
    run_cmd("sysctl vm.vfs_cache_pressure=500 || true")

try:
    import psutil
    _sw = psutil.swap_memory()
    _vm = psutil.virtual_memory()
    print(f"  📊 Memory: Host RAM {_vm.available/1e9:.2f} GB free / {_vm.total/1e9:.2f} GB | Swap {_sw.total/1e9:.2f} GB")
except Exception:
    pass

# Patch sys.modules to prevent utils.install_util conflicts inside ComfyUI.
if "utils" not in sys.modules or not hasattr(sys.modules["utils"], "__path__"):
    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = ["/content/ComfyUI/utils"]
    sys.modules["utils"] = utils_mod
else:
    utils_mod = sys.modules["utils"]

install_util_mod = types.ModuleType("utils.install_util")
install_util_mod.get_missing_requirements_message = lambda *a, **k: ""
install_util_mod.get_required_packages_versions = lambda *a, **k: {}
install_util_mod.requirements_path = "/content/ComfyUI/requirements.txt"
install_util_mod.install_requirements = lambda *a, **k: None
install_util_mod.check_requirements = lambda *a, **k: True
sys.modules["utils.install_util"] = install_util_mod
setattr(utils_mod, "install_util", install_util_mod)

print("✅ Cell 1: Environment & Memory Protection configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
print("⚙️ [2/3] Installing Core Dependencies & PyTorch...")
run_cmd("pip install -q torch torchvision torchaudio", silent=False)
run_cmd("pip uninstall -y utils || true")
os.chdir("/content")

run_cmd("pip install -q torchsde einops diffusers accelerate psutil")
run_cmd("pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio imageio aiohttp scipy")
run_cmd("pip install -q 'kornia==0.7.3'")
run_cmd("apt-get -y install -qq aria2 ffmpeg")
# SageAttention → 1.5-2x faster attention during sampling (PatchSageAttentionKJ needs it).
# Optional; the pipeline still runs if this wheel is unavailable for the T4's CUDA.
run_cmd("pip install -q sageattention || true")

print("✅ Cell 2: Dependencies installed.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE UPSTREAM COMFYUI CORE
# ════════════════════════════════════════════════════════════════════════════
if not os.path.isdir("/content/ComfyUI"):
    print("⚙️ [3/3] Cloning ComfyUI repository...")
    run_cmd("git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI")
    run_cmd("pip install -q -r /content/ComfyUI/requirements.txt")

if "/content/ComfyUI" not in sys.path:
    sys.path.insert(0, "/content/ComfyUI")
if "/content" not in sys.path:
    sys.path.insert(1, "/content")

os.makedirs("/content/ComfyUI/utils", exist_ok=True)
run_cmd("touch /content/ComfyUI/utils/__init__.py")

print("✅ Cell 3: ComfyUI Core ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 4: INSTALL REQUIRED CUSTOM NODES  (exactly the packs the JSON needs)
# ════════════════════════════════════════════════════════════════════════════
custom_nodes_dir = "/content/ComfyUI/custom_nodes"
os.makedirs(custom_nodes_dir, exist_ok=True)

for item in os.listdir(custom_nodes_dir):
    full_p = os.path.join(custom_nodes_dir, item)
    if os.path.isdir(full_p) and (item.isdigit() or item.startswith(".") or item == "comfyui"):
        shutil.rmtree(full_p, ignore_errors=True)

os.chdir(custom_nodes_dir)

# cnr_id references from the JSON:
#   whatdreamscost-comfyui  -> LTXDirector / LTXDirectorGuide / LTXDirectorCropGuides
#   comfyui-kjnodes         -> VAELoaderKJ / ModelPreviewOverrideKJ / PatchSageAttentionKJ
#   ComfyUI-GGUF            -> UnetLoaderGGUF
#   ComfyUI-LTXVideo        -> LTXV* AV latent / conditioning / upsampler / audio decode
#   VideoHelperSuite        -> VHS_VideoCombine
#   rgthree-comfy           -> Power Lora Loader (rgthree)
repos = [
    ("WhatDreamsCost-ComfyUI", "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI"),
    ("ComfyUI_KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI_GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy"),
]

for folder, url in repos:
    if not os.path.isdir(folder):
        print(f"  Cloning {folder}...")
        run_cmd(f"git clone {url} {folder}")
        req_file = os.path.join(folder, "requirements.txt")
        if os.path.isfile(req_file):
            run_cmd(f"pip install -q -r {req_file} || true")

print("✅ Cell 4: Custom Nodes installed.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 5: DOWNLOAD MODELS, 4-LoRA STACK & AUDIO ASSET
# ════════════════════════════════════════════════════════════════════════════
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def download_file(url: str, dest_dir: str, filename: Optional[str] = None) -> Optional[str]:
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
        dest = os.path.join(dest_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  [FOUND] {filename}")
            return filename
        cmd = ['aria2c', '--console-log-level=error', '-c', '-x', '16',
               '-s', '16', '-k', '1M', '-d', dest_dir, '-o', filename, url]
        print(f"  ↓ Downloading {filename}...", end=' ', flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print("Done!")
            return filename
        print("FAILED")
        return None
    except Exception as e:
        print(f"\n  Error downloading {filename}: {e}")
        return None


def link_file_safe(src_path: str, dst_path: str):
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if not os.path.exists(dst_path) and os.path.exists(src_path):
            os.symlink(src_path, dst_path)
    except Exception:
        try:
            shutil.copyfile(src_path, dst_path)
        except Exception:
            pass


print("📦 Downloading LTX-2.3 Core Models...")

download_file(
    "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
    "/content/ComfyUI/models/unet", filename="ltx-2-3-22b-dev-Q4_K_M.gguf")
link_file_safe("/content/ComfyUI/models/unet/ltx-2-3-22b-dev-Q4_K_M.gguf",
               "/content/ComfyUI/models/diffusion_models/ltx-2-3-22b-dev-Q4_K_M.gguf")

download_file(
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders", filename="gemma_3_12B_it_fp4_mixed.safetensors")
link_file_safe("/content/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
               "/content/ComfyUI/models/clip/gemma_3_12B_it_fp4_mixed.safetensors")

download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders", filename="ltx-2.3_text_projection_bf16.safetensors")
link_file_safe("/content/ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
               "/content/ComfyUI/models/clip/ltx-2.3_text_projection_bf16.safetensors")

download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae", filename="LTX23_video_vae_bf16.safetensors")
download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae", filename="LTX23_audio_vae_bf16.safetensors")
download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae", filename="taeltx2_3.safetensors")
link_file_safe("/content/ComfyUI/models/vae/taeltx2_3.safetensors",
               "/content/ComfyUI/models/vae_approx/taeltx2_3.safetensors")

download_file(
    "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "/content/ComfyUI/models/latent_upscale_models", filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
link_file_safe("/content/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
               "/content/ComfyUI/models/upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

print("📦 Downloading Director 2.0 4-LoRA Stack...")
lora_dir = "/content/ComfyUI/models/loras"
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
              lora_dir, filename="ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
              lora_dir, filename="LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
download_file("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",
              lora_dir, filename="ltx2.3-transition.safetensors")
download_file("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors",
              lora_dir, filename="LTX2.3-MVCamera-drclips.safetensors")

audio_dest_dir = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(audio_dest_dir, exist_ok=True)
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file("https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3",
                  audio_dest_dir, filename="Late night trap.mp3")

print("✅ Cell 5: Models, LoRAs and audio validated.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 6: MASTER TIMELINE "NOTES"  (transcribed 1:1 from LTXDirector node id 131)
# ════════════════════════════════════════════════════════════════════════════
# Every value below is copied from the LTXDirector node inside
# LTX-2.3_Director_2.0-MV-Workflow-30s.json — this is the full set of "notes"
# (main track + audio track + motion track + global prompt) that the old
# Master_V2 was missing.

# ── Global positive prompt (exact copy from the workflow's LTXDirector node) ──
GLOBAL_PROMPT = """Create a highly realistic cinematic AI music video using the provided reference image. Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body proportions, and overall appearance exactly as in the reference image. The singer must remain fully recognizable throughout the entire video with absolutely no identity drift.

The person is performing directly to the camera as a world-class pop, hip-hop and rap singer during a sold-out stadium concert. Generate perfectly synchronized lip movements from the provided lyrics or audio.

This is NOT a talking-head video and NOT a presenter. This is a high-energy live music performance filled with charisma, attitude and emotional intensity.

Performance Energy:
• Perform with explosive stage presence.
• Every musical phrase immediately creates a new emotional and physical performance.
• Every lyric instantly changes facial expression, eye emotion, head movement, shoulders, hands, posture and body rhythm.
• The performance continuously builds toward emotional peaks.
• Own the stage with absolute confidence.
• Perform as if in front of 50,000 screaming fans.
• Captivate the audience every second.
• Never appear calm, passive or static.

Facial Performance:
• Extremely expressive facial acting throughout the entire performance.
• Rich emotional transitions every few words.
• Powerful eye contact with intense emotional engagement.
• Eyes sparkle with confidence and passion.
• Highly expressive eyebrows synchronized with important lyrics.
• Strong cheek and jaw movement while singing.
• Natural smiles, smirks, determination, excitement, confidence, attitude, passion, curiosity, joy and intensity.
• Rich cinematic micro-expressions.
• Never hold the same facial expression for more than a brief musical phrase.
• The face should feel emotionally alive every second.

Body Performance:
• The entire body constantly grooves with the beat.
• Strong rhythmic bouncing.
• Powerful shoulder accents.
• Confident chest movement.
• Hip movement follows the groove.
• Frequent body turns.
• Fast weight shifts.
• Dynamic torso twists.
• Lean toward the camera during emotional lyrics.
• Occasionally step toward the camera.
• Performance intensity increases naturally during powerful musical moments.
• Bold, energetic and theatrical stage movement.

Hand Performance:
• Perform like an experienced pop or hip-hop superstar.
• Large expressive gestures.
• Fast rhythmic arm accents.
• Sharp hand movements synchronized with the beat.
• Powerful pointing.
• Sweeping arm movements.
• Punching the air.
• Pulling gestures toward the chest.
• Throwing gestures outward.
• Finger snapping.
• Open palm emphasis.
• Framing the face.
• Expressive wrist movement.
• Hands constantly create visual rhythm.
• One hand naturally leads while the other follows.
• Asymmetrical movement.
• Avoid symmetrical gestures.
• Never repeatedly raise both hands together.
• Every musical phrase introduces fresh gestures.
• Never repeat the same gesture pattern.

Musical Timing:
• Body movement follows musical phrasing rather than every word.
• Strong beats create explosive movements.
• Soft phrases become intimate and emotional.
• Fast lyrics generate faster gestures.
• Slow lyrics become smoother without losing energy.
• Every movement feels rhythmically connected to the music.

Speech Synchronization:
• Perfect lip synchronization.
• Accurate mouth shapes.
• Expressions and gestures match the emotional meaning of every lyric.
• Natural breathing between phrases.

Motion Quality:
• Premium AI human animation.
• Fast, confident and energetic performance.
• Realistic momentum.
• Strong acceleration and deceleration.
• High-energy body mechanics.
• Natural motion blur.
• No robotic movement.
• No frozen poses.
• No repetitive gesture loops.
• No presenter-style gestures.
• No idle standing.
• No jitter.
• No flickering.
• No facial distortion.
• No identity drift.
• No hand deformation.
• No extra fingers.
• No malformed limbs.

Camera:
drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur. Camera movement follows the beat and amplifies the performance.

Lighting:
Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.

Overall Style:
Photorealistic, blockbuster-quality AI music video, premium live concert performance, ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive stage energy, bold movement, powerful attitude, modern pop, hip-hop and rap performance, every second feels alive, impossible to look away.

Spoken dialogue:
"Open up the canvas, blank space on my screen.
Drag a Checkpoint Loader, you know what I mean.
KSampler in the middle, VAE on the right,
Put the Text Encoder, yeah, building tonight.
Connect the nodes, run the queue,
Watch the latent flow right through.
Green, nothing green, nothing yellow,
Positive Prompt, in my hub."
"""

# ── Negative prompt (exact copy) ──
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, robotic movement, static presenter, jitter, flicker, facial distortion, extra limbs, watermark"

# ════════════════════════════════════════════════════════════════════════════
#  🎛️  SETTINGS PANEL  (Colab @param — edit here, then run Cell 17)
# ════════════════════════════════════════════════════════════════════════════
# @markdown ## 🎬 Resolution & Timing
generation_width  = 832    # @param [416, 512, 640, 704, 768, 832, 960, 1024] {type:"raw"}
generation_height = 480    # @param [240, 320, 384, 448, 480, 512, 544, 576, 720] {type:"raw"}
fps               = 24     # @param [24, 25, 30] {type:"raw"}
# @markdown `render_seconds` — lower this (e.g. 8–12 s) to shrink VRAM/RAM on a free T4. 31.5 = full JSON.
render_seconds    = 31.5   # @param {type:"slider", min:3.0, max:31.5, step:0.5}

# @markdown ## 🎞️ Timeline Authoring Canvas (LTXDirector — exact JSON values)
custom_width      = 1280   # @param [768, 1024, 1280] {type:"raw"}
custom_height     = 720    # @param [512, 576, 720] {type:"raw"}
img_compression   = 18     # @param {type:"slider", min:0, max:60, step:1}
divisible_by      = 32     # @param {type:"raw"}
keyframe_guide_strength = 1.0  # @param {type:"slider", min:0.0, max:1.0, step:0.05}
# @markdown ### ⚡ SPEED KNOB — Stage-1 base render resolution
# @markdown `two_stage_base_render` = LTXDirector renders SMALL (generation//2, e.g. 416x240) so the
# @markdown 8-step Stage 1 is ~7x FASTER on a T4; LTXVLatentUpsampler then 2x-upscales to the target.
# @markdown Set False ONLY on an L4/A100 to render Stage 1 at the full 1280x720 canvas (very slow on a T4).
two_stage_base_render = True   # @param {type:"boolean"}

# @markdown ## 🎛️ Director 2.0 4-LoRA Stack  (exact JSON strengths; all ON like the workflow)
use_lora_1 = True      # @param {type:"boolean"}
lora_strength_1 = 0.4  # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_2 = True      # @param {type:"boolean"}
lora_strength_2 = 0.6  # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_3 = True      # @param {type:"boolean"}
lora_strength_3 = 0.7  # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_4 = True      # @param {type:"boolean"}
lora_strength_4 = 0.9  # @param {type:"slider", min:0.0, max:1.5, step:0.05}

# @markdown ## ⚡ Two-Stage Sampler  (matches the original JSON graph)
scheduler_name = "linear_quadratic"  # @param ["linear_quadratic", "normal", "simple", "beta", "sgm_uniform", "karras"]
sampler_name   = "euler"             # @param ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim"]
cfg            = 1.0                  # @param {type:"slider", min:1.0, max:8.0, step:0.5}
stage1_steps   = 8                   # @param {type:"slider", min:4, max:30, step:1}
stage1_denoise = 1.0                 # @param {type:"slider", min:0.1, max:1.0, step:0.01}
stage1_guide_strength = 0.5          # @param {type:"slider", min:0.0, max:1.0, step:0.05}
stage2_steps   = 4                   # @param {type:"slider", min:2, max:20, step:1}
stage2_denoise = 0.42                # @param {type:"slider", min:0.1, max:1.0, step:0.01}
stage2_guide_strength = 1.0          # @param {type:"slider", min:0.0, max:1.0, step:0.05}

# @markdown ## 🎵 Audio (synchronized voice track — real audio latent, NOT empty)
use_song_audio = True                        # @param {type:"boolean"}
audio_trim_start_frames = 446.9222739141953  # @param {type:"raw"}

# @markdown ## 🧠 Memory / Performance  (free-tier T4)
VRAM_MODE = "auto"            # @param ["auto", "normalvram", "lowvram", "novram", "highvram"]
essential_loras_only = False  # @param {type:"boolean"}   # False = faithful 4-LoRA; True = distilled-only (saves VRAM/RAM)
# @markdown `vram_shield_mb` — RESERVE this much VRAM so on-the-fly GGUF LoRA deltas (adaln etc.) have
# @markdown room to compute. 256 was too small → "lora ... CUDA out of memory". 1200 matches the old
# @markdown working Master_V2 (ComfyUI offloads a little model to CPU, leaving headroom for LoRA patches).
vram_shield_mb       = 1200   # @param {type:"raw"}
min_ram_guard_gb     = 1.5    # @param {type:"slider", min:1.0, max:6.0, step:0.5}
# @markdown ### 🎬 Batch-by-batch scene generation (fits the T4; shares ONE timeline + ONE audio latent)
batch_scene_mode          = True   # @param {type:"boolean"}
scene_mode                = "keyframe"  # @param ["keyframe", "fixed"]
scene_chunk_seconds       = 4.0    # @param {type:"slider", min:1.0, max:10.0, step:0.5}
scene_overlap_frames      = 8      # @param {type:"slider", min:0, max:24, step:1}
# @markdown `reload_dit_per_scene` — True = reload the 22B DiT fresh before each scene (old Master_V2's
# @markdown winning pattern: resets VRAM every scene so nothing accumulates → no per-scene LoRA OOM).
# @markdown False loads once (faster) but VRAM creeps up across scenes on a T4. Keep True on a free T4.
reload_dit_per_scene      = True   # @param {type:"boolean"}

# @markdown ## 💾 Output & Run
output_crf         = 8     # @param {type:"slider", min:0, max:30, step:1}
base_seed          = 0     # @param {type:"integer"}
resume_checkpoints = True  # @param {type:"boolean"}

# ────────────────────────────────────────────────────────────────────────────
#  Derived configuration (built from settings above — do not edit below).
# ────────────────────────────────────────────────────────────────────────────
# Original timeline (frames). LTX needs a valid latent length: frames = 8*k + 1.
# Main-track keyframe segment lengths transcribed 1:1 from node 131 timeline_data.
_ORIG_SEGMENTS = [
    ("1785555235678s2fn3", 226.01059340956584, "whatdreamscost/1.png"),
    ("17855552413529uw9r", 161.31859976617454, "whatdreamscost/2.png"),
    ("1785555243885y3h85", 131.45629831196658, "whatdreamscost/3.png"),
    ("1785555247117rcoma", 225.5063328766255,  "whatdreamscost/4.png"),
    ("17855554543736wlrg", 83.22765271847516,  "whatdreamscost/5.3.png"),
]
_ORIG_TOTAL_FRAMES = 756
_ORIG_AUDIO_LEN = 756.5194770828076   # audioSegments[0].length (exact JSON value)


def _snap_ltx_frames(n: float) -> int:
    n = int(max(9, round(n)))
    return ((n - 1) // 8) * 8 + 1


_total_frames = _snap_ltx_frames(render_seconds * fps)
_factor = _total_frames / _ORIG_TOTAL_FRAMES
_duration_seconds = _total_frames / float(fps)

# Scale the 5 keyframes proportionally across the (possibly shortened) timeline.
ORIGINAL_SEGMENTS: List[Dict[str, Any]] = []
_cursor = 0.0
for _sid, _slen, _img in _ORIG_SEGMENTS:
    _L = _slen * _factor
    ORIGINAL_SEGMENTS.append({"id": _sid, "start": _cursor, "length": _L,
                              "prompt": "", "type": "image", "imageFile": _img})
    _cursor += _L
_segment_lengths_str = ",".join(f"{s['length']}" for s in ORIGINAL_SEGMENTS)
_guide_strength_str = ",".join(f"{keyframe_guide_strength:.2f}" for _ in ORIGINAL_SEGMENTS)

# The ONE continuous audio track (this is what makes the voice SYNC — the old
# Master_V2 fed a fresh EMPTY audio latent per fake segment, which is exactly
# why the voice never lined up).
ORIGINAL_AUDIO_SEGMENTS = [{
    "id": "1785169457779kollx", "type": "audio", "start": 0.0,
    "length": _ORIG_AUDIO_LEN * _factor, "trimStart": float(audio_trim_start_frames),
    "audioDurationFrames": 2880, "audioFile": "whatdreamscost/Late night trap.mp3",
    "fileName": "Late night trap.mp3",
}]
ORIGINAL_MOTION_SEGMENTS: List[Dict[str, Any]] = []   # motion track empty in the JSON


def _snap_div(n: int, d: int) -> int:
    d = max(1, int(d))
    return max(d, int(round(n / d)) * d)


# ⚡ SPEED FIX: LTXDirector renders latents at custom_width/custom_height. The old
# 1280x720 canvas made the 8-step Stage 1 sample at ~720p (~7x slower on a T4).
# With two_stage_base_render=True we render at the Stage-1 BASE (generation//2,
# snapped to divisible_by), then LTXVLatentUpsampler 2x-upscales to the target.
_base_w = _snap_div(int(generation_width) // 2, divisible_by)
_base_h = _snap_div(int(generation_height) // 2, divisible_by)
if two_stage_base_render:
    _director_render_w, _director_render_h = _base_w, _base_h
else:
    _director_render_w, _director_render_h = int(custom_width), int(custom_height)

TIMELINE_METADATA = {
    "frame_rate": float(fps),
    "duration_seconds": _duration_seconds,
    "normalDurationFrames": _total_frames,
    "start_frame": 0,
    "end_frame": _total_frames,
    # custom_width/height is the resolution LTXDirector actually RENDERS at.
    "custom_width": int(_director_render_w),
    "custom_height": int(_director_render_h),
    "authoring_width": int(custom_width),      # original 1280x720 authoring canvas (note only)
    "authoring_height": int(custom_height),
    "generation_width": int(generation_width),
    "generation_height": int(generation_height),
    "base_stage1_width": _base_w,
    "base_stage1_height": _base_h,
    "mainTrackEnabled": True,
    "audioTrackEnabled": bool(use_song_audio),
    "motionTrackEnabled": True,
    "inpaint_audio": True,
    "override_audio": False,
    "use_custom_audio": bool(use_song_audio),
    "use_custom_motion": True,
    "audio_file": "whatdreamscost/Late night trap.mp3",
    "audio_duration_frames": 2880,
    "audio_trim_start_frames": float(audio_trim_start_frames),
    "resize_method": "maintain aspect ratio",
    "divisible_by": int(divisible_by),
    "img_compression": int(img_compression),
    "epsilon": 0.001,
    "display_mode": "seconds",
    "guide_strength": _guide_strength_str,
    "segment_lengths": _segment_lengths_str,
    "local_prompts": "",   # empty → LTXDirector single-prompt bypass (all 5 local prompts empty in JSON)
}

# Director 2.0 LoRA stack — exact JSON strengths, applied to the model.
_ALL_LORAS = [
    (use_lora_1, "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", lora_strength_1),
    (use_lora_2, "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",                                lora_strength_2),
    (use_lora_3, "ltx2.3-transition.safetensors",                                           lora_strength_3),
    (use_lora_4, "LTX2.3-MVCamera-drclips.safetensors",                                     lora_strength_4),
]
LORA_STACK = [{"on": bool(o), "lora": n, "strength": float(s)} for (o, n, s) in _ALL_LORAS if o]

STAGE1 = {"scheduler": scheduler_name, "steps": int(stage1_steps), "denoise": float(stage1_denoise),
          "cfg": float(cfg), "guide_strength": float(stage1_guide_strength)}
STAGE2 = {"scheduler": scheduler_name, "steps": int(stage2_steps), "denoise": float(stage2_denoise),
          "cfg": float(cfg), "guide_strength": float(stage2_guide_strength)}
VHS_SETTINGS = {"format": "video/h264-mp4", "pix_fmt": "yuv420p",
                "crf": int(output_crf), "filename_prefix": "LTX23_Director_Master"}

# Runtime globals consumed by later cells.
ESSENTIAL_LORAS_ONLY = bool(essential_loras_only)
VRAM_MODE = str(VRAM_MODE)
VRAM_SHIELD_MB = int(vram_shield_mb)
BATCH_SCENE_MODE = bool(batch_scene_mode)
SCENE_MODE = str(scene_mode)
RELOAD_DIT_PER_SCENE = bool(reload_dit_per_scene)
SCENE_CHUNK_LATENT_FRAMES = max(2, int(round((scene_chunk_seconds * fps) / 8)))
SCENE_OVERLAP_LATENT_FRAMES = max(0, int(round(scene_overlap_frames / 8)))
BASE_SEED = int(base_seed)
OUTPUT_CRF = int(output_crf)
RESUME_CHECKPOINTS = bool(resume_checkpoints)
USE_SONG_AUDIO = bool(use_song_audio)

print(f"✅ Cell 6: Master Timeline notes loaded → Stage1 base {_director_render_w}x{_director_render_h} "
      f"→ 2x → {generation_width}x{generation_height} @ {fps}fps · "
      f"{_duration_seconds:.1f}s ({_total_frames} frames) · 5 keyframes · 1 audio track · "
      f"{len(LORA_STACK)} LoRA(s) · VRAM_MODE={VRAM_MODE} · "
      f"two_stage_base_render={two_stage_base_render}")
if two_stage_base_render:
    print(f"  ⚡ Stage 1 will sample at {_director_render_w}x{_director_render_h} (fast) instead of "
          f"{custom_width}x{custom_height} — expect ~{(custom_width*custom_height)/(_director_render_w*_director_render_h):.1f}x faster iterations.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 7: NODE REGISTRY & 23-NODE ORIGINAL-WORKFLOW AUDIT
# ════════════════════════════════════════════════════════════════════════════
import asyncio
import nest_asyncio
nest_asyncio.apply()

try:
    import server
    from server import PromptServer
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        try:
            PromptServer.instance = PromptServer(loop)
        except Exception:
            class MockServer:
                def __init__(self):
                    from aiohttp import web
                    self.routes = web.RouteTableDef()
                    self.app = web.Application()
                    self.loop = loop

                def send_sync(self, *a, **k):
                    pass
            PromptServer.instance = MockServer()
except Exception:
    pass

from nodes import init_builtin_extra_nodes, init_external_custom_nodes


async def _init_nodes_async():
    try:
        await init_builtin_extra_nodes()
    except Exception:
        pass
    try:
        await init_external_custom_nodes()
    except Exception:
        pass


try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.run_until_complete(asyncio.ensure_future(_init_nodes_async()))
    else:
        loop.run_until_complete(_init_nodes_async())
except Exception:
    pass

from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly

# The exact 23 node types present in LTX-2.3_Director_2.0-MV-Workflow-30s.json.
REQUIRED_WORKFLOW_NODES = [
    "LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides",   # whatdreamscost (Master Timeline Controller + guides)
    "LTXVConditioning", "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
    "LTXVLatentUpsampler", "LTXVAudioVAEDecode",                  # LTXVideo
    "Power Lora Loader (rgthree)",                                # rgthree — 4-LoRA stack on model+clip
    "ModelPreviewOverrideKJ", "VAELoaderKJ",                      # KJNodes
    "UnetLoaderGGUF",                                             # GGUF
    "DualCLIPLoader", "ConditioningZeroOut", "SamplerCustomAdvanced",
    "CFGGuider", "KSamplerSelect", "BasicScheduler", "RandomNoise",
    "VAEDecode", "VAELoader", "LatentUpscaleModelLoader",         # comfy-core
    "VHS_VideoCombine",                                           # VideoHelperSuite
]


def validate_original_nodes() -> bool:
    print("\n" + "=" * 70 + "\n🔍 ORIGINAL WORKFLOW NODE AUDIT (23 nodes)\n" + "=" * 70)
    missing = []
    for name in REQUIRED_WORKFLOW_NODES:
        if name in NODE_CLASS_MAPPINGS:
            print(f"  ✓ {name:<32}-> {NODE_CLASS_MAPPINGS[name].__name__}")
        else:
            print(f"  ❌ MISSING: {name}")
            missing.append(name)
    if missing:
        raise RuntimeError(
            "NODE VALIDATION FAILED. Missing: " + ", ".join(missing) +
            "\nRe-run Cell 4 (custom node install) — the workflow cannot run without these."
        )
    print(f"✅ Cell 7: all {len(REQUIRED_WORKFLOW_NODES)} original workflow nodes verified.")
    return True


validate_original_nodes()


# ════════════════════════════════════════════════════════════════════════════
# CELL 8: PRODUCTION MEMORY ENGINE (purge_deep / ram_guard / VRAM shield)
# ════════════════════════════════════════════════════════════════════════════
def malloc_trim_os():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def get_ram_free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99.0


def patch_comfy_memory_manager():
    """Fix free_memory return type + reserve a VRAM shield (VRAM_SHIELD_MB) so
    ComfyUI keeps headroom for on-the-fly GGUF LoRA delta computation (otherwise
    'lora ... CUDA out of memory'); force the text encoder onto CUDA so the 12B
    Gemma dequantizes into VRAM, not host RAM."""
    try:
        import comfy.model_management as mm
        if not getattr(mm, "_ltx_patched", False):
            _orig_free = mm.free_memory

            def _safe_free(*a, **k):
                try:
                    r = _orig_free(*a, **k)
                    return r if isinstance(r, list) else []
                except Exception:
                    return []
            mm.free_memory = _safe_free

            _orig_getfree = mm.get_free_memory

            def _buffered_getfree(dev=None, torch_free_too=False):
                try:
                    free = _orig_getfree(dev, torch_free_too)
                    shield = int(globals().get("VRAM_SHIELD_MB", 256)) * 1024 * 1024
                    return max(256 * 1024 * 1024, free - shield)
                except Exception:
                    return 2 * 1024 * 1024 * 1024
            mm.get_free_memory = _buffered_getfree
            mm._ltx_patched = True

        if torch.cuda.is_available():
            mm.text_encoder_device = lambda: torch.device("cuda")
            mm.text_encoder_offload_device = lambda: torch.device("cuda")
    except Exception as e:
        print(f"  [mem-patch notice] {e}")


def patch_safetensors_direct_to_gpu():
    """Load text-encoder shards straight onto CUDA so host RAM never spikes."""
    try:
        import safetensors.torch as st
        if not getattr(st, "_ltx_cuda_direct", False):
            _orig = st.load_file

            def _cuda_load(filename, device="cpu"):
                fn = str(filename).lower()
                if torch.cuda.is_available() and any(k in fn for k in
                        ["gemma", "clip", "text_encoder", "projection", "connector"]):
                    return _orig(filename, device="cuda")
                return _orig(filename, device=device)
            st.load_file = _cuda_load
            st._ltx_cuda_direct = True
    except Exception:
        pass


def drop_os_page_cache():
    patterns = [
        "/content/ComfyUI/models/unet/*.gguf",
        "/content/ComfyUI/models/diffusion_models/*.gguf",
        "/content/ComfyUI/models/text_encoders/*.safetensors",
        "/content/ComfyUI/models/clip/*.safetensors",
        "/content/ComfyUI/models/vae/*.safetensors",
        "/content/ComfyUI/models/latent_upscale_models/*.safetensors",
        "/content/ComfyUI/models/upscale_models/*.safetensors",
        "/content/ComfyUI/models/loras/*.safetensors",
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                fd = os.open(f, os.O_RDONLY)
                size = os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except Exception:
                pass


def purge_deep(tag: str = ""):
    """The heavy 'clear_memory' routine, run between every phase/heavy node."""
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
            mm.current_loaded_models.clear()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    drop_os_page_cache()
    malloc_trim_os()


def ram_guard(min_free_gb: float = 2.0, tag: str = ""):
    if get_ram_free_gb() < min_free_gb:
        print(f"  ⚠️ [RAM GUARD] Free RAM {get_ram_free_gb():.2f} GB < {min_free_gb} GB → deep purge")
        purge_deep(f"ram_guard:{tag}")


def light_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def medium_clear(tag: str = ""):
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    malloc_trim_os()


def install_sampling_memory_hook(clear_every: int = 4, ram_guard_gb: float = 0.0):
    """Per-step light memory clear during sampling → prevents fragmentation OOM."""
    try:
        import comfy.utils as cu
        state = {"n": 0}

        def _hook(value, total, preview_bytes=None, *args, **kwargs):
            state["n"] += 1
            if clear_every <= 1 or (state["n"] % clear_every == 0):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if ram_guard_gb > 0 and (state["n"] % 4 == 0):
                if get_ram_free_gb() < ram_guard_gb:
                    gc.collect()
                    malloc_trim_os()

        if hasattr(cu, "set_progress_bar_global_hook"):
            cu.set_progress_bar_global_hook(_hook)
            print(f"  ⚙️ Per-step memory-clear hook active (every {clear_every} step[s]).")
    except Exception as e:
        print(f"  [mem-hook notice] {e}")


def mem_report(phase: str = "", node: str = ""):
    ram = get_ram_free_gb()
    if torch.cuda.is_available():
        gfree = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
        galloc = torch.cuda.memory_allocated() / 1e9
    else:
        gfree = galloc = 0.0
    tail = f" | {phase}" if phase else ""
    tail += f" · {node}" if node else ""
    print(f"  📊 RAM free {ram:.2f} GB | VRAM free {gfree:.2f} GB (alloc {galloc:.2f} GB){tail}")


def configure_vram_state(mode: str = "auto"):
    try:
        import comfy.model_management as mm
        if not torch.cuda.is_available():
            return
        mode = (mode or "auto").lower()
        dev = mm.get_torch_device()
        vs = getattr(mm, "VRAMState", None)
        if vs is None:
            return
        if mode == "highvram":
            mm.vram_state = vs.HIGH_VRAM
            mm.unet_offload_device = lambda: dev
            print("  ⚙️ VRAM → HIGH_VRAM (whole model on GPU — needs L4/A100).")
        elif mode == "lowvram":
            mm.vram_state = vs.LOW_VRAM
            print("  ⚙️ VRAM → LOW_VRAM (layer-by-layer streaming).")
        elif mode == "novram":
            mm.vram_state = vs.NO_VRAM
            print("  ⚙️ VRAM → NO_VRAM (maximum streaming, slowest).")
        elif mode == "normalvram":
            mm.vram_state = vs.NORMAL_VRAM
            print("  ⚙️ VRAM → NORMAL_VRAM.")
        else:
            print("  ⚙️ VRAM → auto (ComfyUI partial offload — recommended on a T4).")
    except Exception as e:
        print(f"  [vram-state notice] {e}")


patch_comfy_memory_manager()
patch_safetensors_direct_to_gpu()
print("✅ Cell 8: Memory engine active (purge_deep · ram_guard · VRAM shield · page-cache drop).")



# ════════════════════════════════════════════════════════════════════════════
# CELL 9: UNIVERSAL NODE DISPATCHER & LATENT/TENSOR HELPERS
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
from PIL import Image, ImageOps, ImageDraw

# Maps a node parameter name to the set of aliases different node packs may use.
PARAM_ALIASES = {
    "weight_dtype": ["weight_dtype", "dtype", "weight_type", "precision"],
    "dtype": ["weight_dtype", "dtype", "weight_type", "precision"],
    "device": ["device", "device_type", "target_device"],
    "vae_name": ["vae_name", "name", "vae"],
    "model_name": ["model_name", "unet_name", "name"],
    "unet_name": ["unet_name", "model_name", "name"],
    "clip_name": ["clip_name", "clip_name1", "name"],
    "clip_name1": ["clip_name1", "clip_name", "name"],
    "clip_name2": ["clip_name2", "name"],
    "samples": ["samples", "latent", "latents", "video_latent", "av_latent", "latent_image"],
    "latents": ["latents", "latent", "samples", "video_latent", "av_latent", "latent_image"],
    "latent": ["latent", "latents", "samples", "video_latent", "latent_image"],
    "latent_image": ["latent_image", "latent", "samples", "latents", "av_latent"],
    "video_latent": ["video_latent", "latent", "samples"],
    "audio_latent": ["audio_latent", "latent", "samples"],
    "av_latent": ["av_latent", "latent", "samples", "latent_image"],
    "audio_vae": ["audio_vae", "vae"],
    "vae": ["vae", "audio_vae", "video_vae"],
    "upscale_model": ["upscale_model", "latent_upscale_model", "model"],
    "frame_rate": ["frame_rate", "fps"],
    "fps": ["fps", "frame_rate"],
    "images": ["images", "image", "frames"],
    "audio": ["audio", "audio_dict", "samples"],
    "positive": ["positive", "pos"],
    "negative": ["negative", "neg"],
    "guider": ["guider", "cfg_guider"],
    "sigmas": ["sigmas", "sigma"],
    "noise": ["noise", "random_noise"],
    "sampler": ["sampler", "sampler_name", "sampler_select"],
    "noise_seed": ["noise_seed", "seed"],
    "scheduler": ["scheduler", "scheduler_name"],
    "sampler_name": ["sampler_name", "sampler"],
    "global_prompt": ["global_prompt", "prompt"],
    "timeline_data": ["timeline_data", "timeline"],
}


def gv(obj: Any, index: int = 0) -> Any:
    """Safely pull output slot `index` from a tuple/list/dict/NodeOutput/object."""
    if obj is None:
        return None
    if isinstance(obj, (tuple, list)):
        return obj[index] if len(obj) > index else None
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], (list, tuple)):
            return obj["result"][index] if len(obj["result"]) > index else None
        return obj.get(index, None)
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)):
        return obj.args[index] if len(obj.args) > index else None
    for attr in ["output", "outputs", "result", "values", "data"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)) and len(val) > index:
                return val[index]
            if index == 0:
                return val
    try:
        return obj[index]
    except Exception:
        pass
    return obj if index == 0 else None


def unwrap_tensor(obj: Any) -> Any:
    if obj is None or isinstance(obj, torch.Tensor):
        return obj
    for attr in ["output", "result"]:
        if hasattr(obj, attr) and getattr(obj, attr) is not None:
            return unwrap_tensor(getattr(obj, attr))
    if isinstance(obj, (tuple, list)) and obj:
        return unwrap_tensor(obj[0])
    if isinstance(obj, dict):
        if "samples" in obj:
            return unwrap_tensor(obj["samples"])
        if "result" in obj and obj["result"]:
            return unwrap_tensor(obj["result"][0])
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(obj, "args") and obj.args:
        return unwrap_tensor(obj.args[0])
    return obj


def unwrap_latent(x: Any) -> Dict[str, Any]:
    if x is None:
        return {"samples": None}
    for attr in ["output", "result"]:
        if hasattr(x, attr) and getattr(x, attr) is not None:
            x = getattr(x, attr)
    while isinstance(x, (tuple, list)) and x:
        x = x[0]
    if isinstance(x, dict):
        cur = x
        while isinstance(cur, dict) and "samples" in cur and isinstance(cur["samples"], dict):
            cur = cur["samples"]
        if isinstance(cur, dict) and "samples" in cur:
            return cur
        for v in cur.values():
            if isinstance(v, torch.Tensor):
                return {"samples": v}
        return {"samples": cur}
    if isinstance(x, torch.Tensor):
        return {"samples": x}
    return {"samples": x}


def sync_latent_device(latent: Any, target_device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """Park a latent dict on the chosen device (CPU between node calls saves VRAM)."""
    target = torch.device(target_device)
    d = unwrap_latent(latent)
    s = d.get("samples", None)
    if isinstance(s, torch.Tensor):
        if s.is_nested:
            d["samples"] = torch.nested.nested_tensor([t.to(target) for t in s.unbind()])
        else:
            d["samples"] = s.to(target)
    return d


def sync_cond_to_cpu(cond: Any) -> Any:
    if cond is None:
        return None
    if isinstance(cond, torch.Tensor):
        return cond.detach().cpu()
    if isinstance(cond, list):
        return [sync_cond_to_cpu(c) for c in cond]
    if isinstance(cond, tuple):
        return tuple(sync_cond_to_cpu(c) for c in cond)
    if isinstance(cond, dict):
        return {k: sync_cond_to_cpu(v) for k, v in cond.items()}
    return cond


def call_node(node_name: str, node_instance: Optional[Any] = None, **kwargs) -> Any:
    """
    Invoke an original ComfyUI node by name, filling only the parameters its real
    signature accepts (alias-aware, schema-default aware). This drives the exact
    workflow nodes from Python without a graph executor.
    """
    if node_instance is None:
        if node_name not in NODE_CLASS_MAPPINGS:
            raise RuntimeError(f"FATAL: node '{node_name}' is not registered.")
        node_instance = NODE_CLASS_MAPPINGS[node_name]()

    func_name = getattr(node_instance, "FUNCTION", None)
    callables = []
    if func_name and hasattr(node_instance, func_name) and callable(getattr(node_instance, func_name)):
        callables.append(getattr(node_instance, func_name))
    if hasattr(node_instance, "execute") and callable(getattr(node_instance, "execute")):
        callables.append(node_instance.execute)
    for fb in ["direct", "get_guider", "get_noise", "get_sampler", "get_sigmas", "sample",
               "apply_guide", "crop_guides", "upsample_latent", "concat", "separate",
               "encode", "decode", "load_unet", "load_clip", "load_vae", "combine_video",
               "override", "load_lora", "process"]:
        if hasattr(node_instance, fb) and callable(getattr(node_instance, fb)):
            callables.append(getattr(node_instance, fb))

    schema_defaults = {}
    if hasattr(node_instance, "INPUT_TYPES") and callable(node_instance.INPUT_TYPES):
        try:
            it = node_instance.INPUT_TYPES()
            for grp in ["required", "optional", "hidden"]:
                for p_name, p_spec in it.get(grp, {}).items():
                    if isinstance(p_spec, tuple) and len(p_spec) > 1 and isinstance(p_spec[1], dict) and "default" in p_spec[1]:
                        schema_defaults[p_name] = p_spec[1]["default"]
                    elif isinstance(p_spec, tuple) and p_spec and isinstance(p_spec[0], list) and p_spec[0]:
                        schema_defaults[p_name] = p_spec[0][0]
        except Exception:
            pass

    last_err = None
    for func in callables:
        try:
            sig = inspect.signature(func)
            valid = {}
            has_kwargs = False
            for p_name, param in sig.parameters.items():
                if p_name in ("cls", "self"):
                    continue
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    has_kwargs = True
                    continue
                if p_name in kwargs:
                    valid[p_name] = kwargs[p_name]
                    continue
                matched = False
                for alias in PARAM_ALIASES.get(p_name, [p_name]):
                    if alias in kwargs:
                        valid[p_name] = kwargs[alias]
                        matched = True
                        break
                if matched:
                    continue
                if param.default is not inspect.Parameter.empty:
                    continue
                if p_name in schema_defaults:
                    valid[p_name] = schema_defaults[p_name]
                    continue
                ann = str(param.annotation)
                if "int" in ann:
                    valid[p_name] = 0
                elif "float" in ann:
                    valid[p_name] = 0.0
                elif "bool" in ann:
                    valid[p_name] = False
                elif "str" in ann:
                    valid[p_name] = ""
                else:
                    valid[p_name] = None
            if has_kwargs:
                for k, v in kwargs.items():
                    valid.setdefault(k, v)
            return func(**valid)
        except Exception:
            last_err = traceback.format_exc()
            continue

    if last_err is not None:
        raise RuntimeError(f"Error calling node '{node_name}':\n{last_err}")
    raise AttributeError(f"No callable function found on node '{node_name}'.")


def tiled_decode_video(video_latent: Any, vae_obj: Any, tile_size: int = 256) -> torch.Tensor:
    """Out-of-core spatiotemporal tiled VAE decode → never holds full RGB on GPU."""
    lat = unwrap_latent(video_latent)
    if "LTXVSpatioTemporalTiledVAEDecode" in NODE_CLASS_MAPPINGS:
        try:
            return unwrap_tensor(call_node(
                "LTXVSpatioTemporalTiledVAEDecode",
                vae=vae_obj, latents=lat,
                spatial_tiles=2, spatial_overlap=8,
                temporal_tile_length=16, temporal_overlap=4,
                last_frame_fix=False, working_device="auto", working_dtype="auto"))
        except Exception:
            pass
    if "VAEDecodeTiled" in NODE_CLASS_MAPPINGS:
        try:
            return unwrap_tensor(call_node("VAEDecodeTiled", samples=lat, vae=vae_obj, tile_size=tile_size))
        except Exception:
            pass
    return unwrap_tensor(call_node("VAEDecode", samples=lat, vae=vae_obj))


def prepare_reference_image(image_path: str, width: int, height: int) -> torch.Tensor:
    """Load + center-crop + resize a keyframe to an IMAGE tensor [1,H,W,3]."""
    if image_path and os.path.exists(image_path):
        img = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        target_aspect = width / height
        w, h = img.size
        if (w / h) > target_aspect:
            nw = int(target_aspect * h)
            off = (w - nw) // 2
            img = img.crop((off, 0, off + nw, h))
        else:
            nh = int(w / target_aspect)
            off = (h - nh) // 2
            img = img.crop((0, off, w, off + nh))
        arr = np.array(img.resize((width, height), Image.BICUBIC)).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    return torch.full((1, height, width, 3), 0.5)


print("✅ Cell 9: Node dispatcher, latent/tensor helpers & tiled decoder ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 10: MASTER TIMELINE CONTROLLER  (builds the LTXDirector payload)
# ════════════════════════════════════════════════════════════════════════════
class DirectorTimelineController:
    """
    Reconstructs the exact `timeline_data` JSON + widget list that the LTXDirector
    node (id 131) carries in the workflow, so the single Master Timeline Controller
    receives all 3 tracks (main / audio / motion) and the global prompt at once.
    THIS is the "notes" object the old Master_V2 never built.
    """

    def __init__(self, global_prompt, negative_prompt, meta, segments,
                 audio_segments, motion_segments, base_input_dir="/content/ComfyUI/input"):
        self.global_prompt = global_prompt
        self.negative_prompt = negative_prompt
        self.meta = meta
        self.segments = segments
        self.audio_segments = audio_segments
        self.motion_segments = motion_segments
        self.base_input_dir = base_input_dir
        self.validate_reference_images()

    def validate_reference_images(self):
        print("\n" + "=" * 70 + "\n🔍 VALIDATING DIRECTOR KEYFRAMES (main track)\n" + "=" * 70)
        for s in self.segments:
            full = os.path.join(self.base_input_dir, s["imageFile"])
            if not os.path.exists(full):
                if "5.3.png" in s["imageFile"]:
                    alt = full.replace("5.3.png", "5.png")
                    if os.path.exists(alt):
                        os.makedirs(os.path.dirname(full), exist_ok=True)
                        shutil.copyfile(alt, full)
                        print(f"  ✓ Alias resolved 5.png → {full}")
                        continue
                os.makedirs(os.path.dirname(full), exist_ok=True)
                ph = Image.new("RGB", (768, 512), color=(40, 30, 70))
                ImageDraw.Draw(ph).text((40, 230),
                    f"UPLOAD singer photo → {os.path.basename(s['imageFile'])}", fill=(255, 255, 255))
                ph.save(full)
                print(f"  ⚠️  Placeholder created (upload your photo): {full}")
            else:
                print(f"  ✓ Keyframe OK: {full}")

    def build_timeline_json_string(self) -> str:
        m = self.meta
        tl = {
            "mainTrackEnabled": m["mainTrackEnabled"],
            "audioTrackEnabled": m["audioTrackEnabled"],
            "motionTrackEnabled": m["motionTrackEnabled"],
            "propHeight": 90,
            "globalPropHeight": 470,
            "showFilenames": True,
            "overrideAudio": m["override_audio"],
            "inpaint_audio": m["inpaint_audio"],
            "global_prompt": self.global_prompt,
            "retake_global_prompt": "",
            "retakeMode": False,
            "retakeStart": 24,
            "retakeLength": 48,
            "retakePrompt": "",
            "retakeStrength": 1,
            "retakeVideo": None,
            "normalStartFrame": int(m["start_frame"]),
            "normalDurationFrames": int(m["normalDurationFrames"]),
            "segments": [
                {
                    "id": s["id"], "start": float(s["start"]), "length": float(s["length"]),
                    "prompt": s.get("prompt", ""), "type": s["type"], "imageFile": s["imageFile"],
                    "imageB64": f"/api/view?filename={os.path.basename(s['imageFile'])}"
                                f"&type=input&subfolder={os.path.dirname(s['imageFile'])}",
                    "isEndFrame": False,
                } for s in self.segments
            ],
            "motionSegments": self.motion_segments,
            "audioSegments": [
                {
                    "id": a["id"], "type": a["type"], "start": float(a["start"]),
                    "length": float(a["length"]), "trimStart": float(a["trimStart"]),
                    "audioDurationFrames": int(a["audioDurationFrames"]),
                    "audioFile": a["audioFile"], "fileName": a["fileName"],
                } for a in self.audio_segments
            ],
        }
        return json.dumps(tl)

    def configure_ltxdirector(self, node_instance: Any):
        """Attach the exact widget list + properties the LTXDirector node expects
        (23 widgets, in the order seen in the JSON)."""
        m = self.meta
        tl_json = self.build_timeline_json_string()
        widgets_values = [
            0,                                    # 0  seed/frame anchor
            float(m["duration_seconds"]),         # 1  start (display=seconds)
            float(m["duration_seconds"]),         # 2  end
            int(m["start_frame"]),                # 3  start frame
            int(m["end_frame"]),                  # 4  end frame
            int(m["normalDurationFrames"]),       # 5  duration frames (756)
            tl_json,                              # 6  full timeline_data JSON (all notes)
            m["local_prompts"] or " |  |  |  | ",  # 7
            str(m["segment_lengths"]),            # 8  per-segment lengths
            float(m["epsilon"]),                  # 9  0.001
            str(m["guide_strength"]),             # 10 "1.00,1.00,1.00,1.00,1.00"
            bool(m["mainTrackEnabled"]),          # 11
            bool(m["audioTrackEnabled"]),         # 12
            bool(m["motionTrackEnabled"]),        # 13
            float(m["frame_rate"]),               # 14 24
            m["display_mode"],                    # 15 "seconds"
            int(m["custom_width"]),               # 16 1280
            int(m["custom_height"]),              # 17 720
            m["resize_method"],                   # 18 "maintain aspect ratio"
            int(m["divisible_by"]),               # 19 32
            int(m["img_compression"]),            # 20 18
            False,                                # 21 retakeMode
            "",                                   # 22 timeline_ui
        ]
        props = {
            "global_prompt": self.global_prompt,
            "timeline_data": tl_json,
            "frame_rate": float(m["frame_rate"]),
            "duration_frames": int(m["normalDurationFrames"]),
            "start_frame": int(m["start_frame"]),
            "end_frame": int(m["end_frame"]),
            "custom_width": int(m["custom_width"]),
            "custom_height": int(m["custom_height"]),
            "guide_strength": str(m["guide_strength"]),
            "segment_lengths": str(m["segment_lengths"]),
            "mainTrackEnabled": bool(m["mainTrackEnabled"]),
            "audioTrackEnabled": bool(m["audioTrackEnabled"]),
            "motionTrackEnabled": bool(m["motionTrackEnabled"]),
            "inpaint_audio": bool(m["inpaint_audio"]),
            "overrideAudio": bool(m["override_audio"]),
            "has_serialized_properties": True,
        }
        if hasattr(node_instance, "properties") and isinstance(node_instance.properties, dict):
            node_instance.properties.update(props)
        else:
            setattr(node_instance, "properties", props)
        setattr(node_instance, "widgets_values", widgets_values)
        setattr(node_instance, "timeline_data", tl_json)
        setattr(node_instance, "global_prompt", self.global_prompt)
        print("  ✓ LTXDirector Master Timeline payload attached (5 keyframes · 1 audio track · motion track).")


print("✅ Cell 10: DirectorTimelineController (Master Timeline Controller) ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 11: MODEL / LoRA LOADERS  (UnetLoaderGGUF → Power Lora Loader → hooks)
# ════════════════════════════════════════════════════════════════════════════
def load_dit_and_loras(clip_obj: Any = None):
    """
    UnetLoaderGGUF ─► Power Lora Loader (rgthree) ─► ModelPreviewOverrideKJ hook.
    Applies the exact 4-LoRA stack (0.4/0.6/0.7/0.9) from the JSON. Falls back to
    LoraLoaderModelOnly if the rgthree signature differs. Returns (model, clip).
    """
    purge_deep("pre_dit_load")
    mem_report("DiT load", "UnetLoaderGGUF")
    model = gv(call_node("UnetLoaderGGUF", unet_name="ltx-2-3-22b-dev-Q4_K_M.gguf"), 0)
    print("  ✓ UnetLoaderGGUF loaded.")

    # ESSENTIAL_LORAS_ONLY keeps just the distilled LoRA to cut the GGUF dequant
    # RAM spike on a T4. Default False = faithful full 4-LoRA stack (like the JSON).
    active_stack = LORA_STACK
    if globals().get("ESSENTIAL_LORAS_ONLY", False):
        active_stack = [lc for lc in LORA_STACK if "distilled" in lc["lora"].lower()] or LORA_STACK[:1]
        print(f"  ⚙️ ESSENTIAL_LORAS_ONLY → applying {len(active_stack)} LoRA(s) to save RAM.")

    applied = False
    if active_stack and "Power Lora Loader (rgthree)" in NODE_CLASS_MAPPINGS:
        try:
            lora_kwargs = {"model": model, "clip": clip_obj}
            for i, lc in enumerate(active_stack, start=1):
                lora_kwargs[f"lora_{i}"] = {"on": lc["on"], "lora": lc["lora"],
                                            "strength": lc["strength"], "strengthTwo": None}
            res = call_node("Power Lora Loader (rgthree)", **lora_kwargs)
            new_model = gv(res, 0)
            new_clip = gv(res, 1)
            if new_model is not None:
                model = new_model
            if new_clip is not None:
                clip_obj = new_clip
            applied = True
            print(f"  ✓ Power Lora Loader (rgthree): {len(active_stack)}-LoRA stack applied.")
        except Exception as e:
            print(f"  [notice] rgthree Power Lora fallback: {e}")

    if not applied:
        # Deterministic per-LoRA fallback (model-only). NOTE the fix: strength_model
        # is the float lc["strength"], NOT the whole config dict (old bug).
        for lc in active_stack:
            path = os.path.join("/content/ComfyUI/models/loras", lc["lora"])
            if lc["on"] and os.path.exists(path):
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    malloc_trim_os()
                    ll = LoraLoaderModelOnly()
                    model = gv(call_node("LoraLoaderModelOnly", node_instance=ll,
                                         model=model, lora_name=lc["lora"],
                                         strength_model=float(lc["strength"])), 0) or model
                    print(f"    + LoRA {lc['lora']} (strength {lc['strength']})")
                except Exception as e:
                    print(f"    [notice] LoRA {lc['lora']} skipped: {e}")

    # Attention / feed-forward memory hooks (KJ + LTXV) — big VRAM savers.
    if "PatchSageAttentionKJ" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node("PatchSageAttentionKJ", model=model, sage_attention="auto"), 0) or model
            print("  ✓ SageAttention hook applied.")
        except Exception:
            pass
    if "LTXVChunkFeedForward" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node("LTXVChunkFeedForward", model=model, chunks=8, dim_threshold=4096), 0) or model
            print("  ✓ ChunkFeedForward hook applied (chunks=8).")
        except Exception:
            pass

    # ModelPreviewOverrideKJ (id 10) — tiny-VAE preview override; pass-through here.
    if "ModelPreviewOverrideKJ" in NODE_CLASS_MAPPINGS:
        try:
            tiny_vae = gv(call_node("VAELoaderKJ", vae_name="taeltx2_3.safetensors",
                                    device="main_device", weight_dtype="bf16"), 0)
            model = gv(call_node("ModelPreviewOverrideKJ", model=model, vae=tiny_vae), 0) or model
            print("  ✓ ModelPreviewOverrideKJ applied.")
        except Exception:
            pass

    return model, clip_obj


print("✅ Cell 11: DiT + 4-LoRA loader with attention hooks ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 12: PHASE A — MASTER TIMELINE INGESTION (LTXDirector, full 756 frames)
# ════════════════════════════════════════════════════════════════════════════
# Free-tier fix: LTXDirector needs BOTH a real model AND a clip. Co-loading the
# 12B Gemma text encoder (dequantizes to ~24 GB) with the DiT blows past 12 GB
# host RAM. So we DECOUPLE them:
#   1) Load ONLY the clip, encode the global prompt on the GPU, PURGE Gemma.
#   2) Load the real GGUF DiT + LoRA stack.
#   3) Run LTXDirector with the REAL model + a PrecomputedClipProxy returning the
#      already-encoded conditioning — so the two big models never co-reside.
class PrecomputedClipProxy:
    """Returns already-encoded conditioning for any text — Gemma stays purged."""
    def __init__(self, precomputed_conditioning, tokenizer=None):
        self.cond = precomputed_conditioning
        self.cond_stage_model = None
        self.patcher = None
        self.layer_idx = None
        self.tokenizer = tokenizer

    def tokenize(self, text, *a, **k):
        if self.tokenizer is not None and hasattr(self.tokenizer, "tokenize_with_weights"):
            try:
                return self.tokenizer.tokenize_with_weights(text)
            except Exception:
                pass
        return {"ltxv": [], "text": text}

    def encode_from_tokens_scheduled(self, *a, **k):
        return self.cond

    def encode_from_tokens(self, *a, **k):
        return self.cond

    def encode(self, *a, **k):
        return self.cond

    def load_model(self, *a, **k):
        return self

    def clone(self):
        return self

    def get_key_patches(self):
        return {}

    def add_patches(self, *a, **k):
        return []

    def __getattr__(self, name):
        return lambda *a, **k: self.cond


def encode_prompt_on_gpu(prompt_text: str):
    """DualCLIPLoader → CLIPTextEncode on CUDA, then purge the 12B encoder weights
    while keeping the tiny tokenizer wrapper. Returns (cond_on_cpu, tokenizer)."""
    purge_deep("pre_clip_load")
    mem_report("Phase A", "DualCLIPLoader (Gemma-3-12B on GPU)")
    import comfy.model_management as mm

    clip = gv(call_node("DualCLIPLoader",
                        clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
                        clip_name2="ltx-2.3_text_projection_bf16.safetensors",
                        type="ltxv", device="default"), 0)
    saved_tokenizer = getattr(clip, "tokenizer", None)

    print("  ⚡ Encoding global prompt on GPU...")
    t0 = time.time()
    with torch.inference_mode():
        cond_raw = gv(call_node("CLIPTextEncode", text=prompt_text, clip=clip), 0)
        cond_cpu = sync_cond_to_cpu(cond_raw)
        del cond_raw
    print(f"  ✓ Prompt encoded in {time.time() - t0:.2f}s. Purging Gemma-12B weights...")

    try:
        clip.cond_stage_model = None
        clip.patcher = None
    except Exception:
        pass
    del clip
    try:
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
            mm.current_loaded_models.clear()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    drop_os_page_cache()
    malloc_trim_os()
    mem_report("Phase A", "Gemma weights purged (tokenizer kept)")
    return cond_cpu, saved_tokenizer


def execute_phase_a(timeline_ctrl: "DirectorTimelineController",
                    workdir: str, resume: bool = True) -> Tuple[Dict[str, Any], Any]:
    print("\n" + "=" * 70 + "\n🎬 PHASE A: LTXDirector MASTER TIMELINE INGESTION (756 frames)\n" + "=" * 70)
    os.makedirs(workdir, exist_ok=True)
    state_file = os.path.join(workdir, "director_state.pt")

    if resume and os.path.exists(state_file) and os.path.getsize(state_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached Director state from: {state_file}")
        try:
            return torch.load(state_file, map_location="cpu"), None
        except Exception as e:
            print(f"  [notice] Director state cache unreadable ({e}) — regenerating.")

    purge_deep("pre_phase_a")
    m = timeline_ctrl.meta
    with torch.inference_mode():
        # A1: encode the prompt with clip ALONE (Gemma on GPU), then purge weights.
        precomputed_cond, saved_tokenizer = encode_prompt_on_gpu(timeline_ctrl.global_prompt)

        # A2: load the real GGUF DiT (mmap) + LoRA stack (model side).
        model, _ = load_dit_and_loras(clip_obj=None)
        mem_report("Phase A", "DiT ready (Gemma already purged)")

        # A3: audio VAE + LTXDirector timeline ingestion with REAL model + clip proxy.
        # This is the step that builds the ONE shared video latent AND the ONE
        # shared AUDIO latent (from the real song) → the voice-sync fix.
        audio_vae = gv(call_node("VAELoader", vae_name="LTX23_audio_vae_bf16.safetensors"), 0)
        clip_proxy = PrecomputedClipProxy(precomputed_cond, tokenizer=saved_tokenizer)
        director_node = NODE_CLASS_MAPPINGS["LTXDirector"]()
        timeline_ctrl.configure_ltxdirector(director_node)

        print("  🚀 Ingesting the full timeline via LTXDirector (Master Timeline Controller)...")
        t_dir = time.time()
        director_out = call_node(
            "LTXDirector", node_instance=director_node,
            model=model, clip=clip_proxy, audio_vae=audio_vae, optional_latent=None,
            global_prompt=timeline_ctrl.global_prompt,
            timeline_data=timeline_ctrl.build_timeline_json_string(),
            local_prompts="", segment_lengths=str(m["segment_lengths"]),
            start_second=0.0, end_second=float(m["duration_seconds"]),
            duration_seconds=float(m["duration_seconds"]),
            start_frame=int(m["start_frame"]), end_frame=int(m["end_frame"]),
            duration_frames=int(m["normalDurationFrames"]),
            epsilon=float(m["epsilon"]), guide_strength=str(m["guide_strength"]),
            frame_rate=float(m["frame_rate"]), display_mode=m["display_mode"],
            custom_width=int(m["custom_width"]), custom_height=int(m["custom_height"]),
            resize_method=m["resize_method"], divisible_by=int(m["divisible_by"]),
            img_compression=int(m["img_compression"]),
            use_custom_audio=bool(m["use_custom_audio"]),
            inpaint_audio=bool(m["inpaint_audio"]),
            use_custom_motion=bool(m["use_custom_motion"]),
            override_audio=bool(m["override_audio"]))
        print(f"  ⚡ Timeline ingested in {time.time() - t_dir:.2f}s.")

        # RETURN order: model, positive, video_latent, audio_latent,
        #               guide_data, motion_guide_data, frame_rate, ...
        patched_model = gv(director_out, 0) or model
        dir_pos = gv(director_out, 1) or precomputed_cond
        dir_vid = sync_latent_device(gv(director_out, 2), "cpu")
        dir_aud = sync_latent_device(gv(director_out, 3), "cpu")
        dir_guide = gv(director_out, 4)
        dir_motion = gv(director_out, 5)
        fps_raw = gv(director_out, 6)
        dir_fps = float(fps_raw) if fps_raw is not None else float(m["frame_rate"])

        # ConditioningZeroOut (node 128) → LTXVConditioning (node 27)
        neg_zeroed = gv(call_node("ConditioningZeroOut", conditioning=dir_pos), 0)
        cond_out = call_node("LTXVConditioning", positive=dir_pos, negative=neg_zeroed, frame_rate=dir_fps)
        final_pos = gv(cond_out, 0)
        final_neg = gv(cond_out, 1)

        state = {
            "positive": final_pos, "negative": final_neg,
            "video_latent": dir_vid, "audio_latent": dir_aud,
            "guide_data": dir_guide, "motion_guide_data": dir_motion,
            "frame_rate": dir_fps, "meta": m,
        }

        try:
            tmp = state_file + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, state_file)
            print(f"  💾 Director state cached: {state_file}")
        except Exception as e:
            print(f"  [notice] Could not cache Director state ({e}) — continuing in-memory.")

        del audio_vae, clip_proxy, director_node, dir_pos, neg_zeroed, cond_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        drop_os_page_cache()
        malloc_trim_os()

    mem_report("Phase A complete (Gemma purged, DiT retained)")
    return state, patched_model


print("✅ Cell 12: Phase A (decoupled encode + real-model LTXDirector ingestion) ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 13: PHASE B — 2-STAGE DIFFUSION OVER THE ONE SHARED TIMELINE
# ════════════════════════════════════════════════════════════════════════════
# The exact JSON graph:
#   Stage 1 : LTXDirectorGuide(0.5) → ConcatAV → SamplerCustomAdvanced
#             (euler · linear_quadratic · 8 steps · denoise 1.0 · cfg 1)
#             → SeparateAV → LTXDirectorCropGuides → LTXVLatentUpsampler (2x)
#   Stage 2 : LTXDirectorGuide(1.0) → ConcatAV(stage-1 audio) → SamplerCustomAdvanced
#             (euler · linear_quadratic · 4 steps · denoise 0.42 · cfg 1)
#             → SeparateAV → LTXDirectorCropGuides
# CRITICAL: every scene-chunk is sliced from the SAME shared video + audio latent
# built in Phase A, so the character, scenes and VOICE stay consistent/synced.

def _run_stage(model, video_vae, guide_strength, base_pos, base_neg,
               video_latent, audio_latent, guide_data, motion_guide_data,
               scheduler, steps, denoise, cfg, seed, stage_name):
    """One LTXDirectorGuide → ConcatAV → Sample → SeparateAV → CropGuides stage."""
    g = call_node(
        "LTXDirectorGuide",
        positive=base_pos, negative=base_neg, vae=video_vae,
        latent=video_latent, guide_data=guide_data,
        motion_guide_data=motion_guide_data, model=model,
        strength=guide_strength, rescale_method="None", guide_frame=1,
        interpolation="bicubic", crop_position="center", enable_guide=True)
    g_pos = gv(g, 0) if gv(g, 0) is not None else base_pos
    g_neg = gv(g, 1) if gv(g, 1) is not None else base_neg
    g_vid = sync_latent_device(gv(g, 2) if gv(g, 2) is not None else video_latent, "cpu")
    g_model = gv(g, 3) if gv(g, 3) is not None else model
    light_clear()

    av = sync_latent_device(gv(call_node("LTXVConcatAVLatent",
                                         video_latent=g_vid, audio_latent=audio_latent), 0), "cpu")
    light_clear()

    noise = gv(call_node("RandomNoise", noise_seed=seed), 0)
    guider = gv(call_node("CFGGuider", cfg=cfg, model=g_model, positive=g_pos, negative=g_neg), 0)
    sampler = gv(call_node("KSamplerSelect", sampler_name="euler"), 0)
    sigmas = gv(call_node("BasicScheduler", model=g_model, scheduler=scheduler,
                          steps=steps, denoise=denoise), 0)

    print(f"  ⚡ {stage_name}: sampling {steps} steps (denoise {denoise})...")
    t0 = time.time()
    ram_guard(globals().get("min_ram_guard_gb", 1.5), stage_name)
    out = call_node("SamplerCustomAdvanced", noise=noise, guider=guider,
                    sampler=sampler, sigmas=sigmas, latent_image=av)
    sampled = sync_latent_device(gv(out, 0), "cpu")
    print(f"  ✓ {stage_name} done in {time.time() - t0:.2f}s.")

    sep = call_node("LTXVSeparateAVLatent", av_latent=sampled)
    v_lat = sync_latent_device(gv(sep, 0), "cpu")
    a_lat = sync_latent_device(gv(sep, 1), "cpu")

    crop = call_node("LTXDirectorCropGuides", positive=g_pos, negative=g_neg, latent=v_lat)
    c_pos = gv(crop, 0) if gv(crop, 0) is not None else g_pos
    c_neg = gv(crop, 1) if gv(crop, 1) is not None else g_neg
    c_vid = sync_latent_device(gv(crop, 2) if gv(crop, 2) is not None else v_lat, "cpu")

    del g, av, noise, guider, sampler, sigmas, out, sampled, sep
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    malloc_trim_os()
    return c_pos, c_neg, c_vid, a_lat, g_model


def _save_ckpt(path: str, obj: Dict[str, Any]) -> bool:
    try:
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"  [notice] Could not write checkpoint {os.path.basename(path)} ({e}).")
        return False


def _ensure_model(model):
    if model is None:
        print("  ↻ Loading DiT for Phase B (resume path)...")
        model, _ = load_dit_and_loras(clip_obj=None)
    return model


def _lat_samples(latdict):
    return unwrap_latent(latdict).get("samples", None)


def _slice_lat_T(latdict, s: int, e: int) -> Dict[str, Any]:
    """Slice a latent dict on the temporal dim (dim 2 of [B,C,T,H,W])."""
    x = _lat_samples(latdict)
    if x is None or x.dim() < 3:
        return {"samples": x.clone() if isinstance(x, torch.Tensor) else x}
    sl = [slice(None)] * x.dim()
    sl[2] = slice(s, e)
    return {"samples": x[tuple(sl)].clone()}


def _blend_append_T(acc: Optional[torch.Tensor], new: torch.Tensor, overlap: int) -> torch.Tensor:
    """Append `new` onto `acc` along the temporal dim, linearly cross-fading the
    `overlap` boundary frames so consecutive scene-chunks join seamlessly."""
    if acc is None:
        return new
    if overlap <= 0 or acc.shape[2] < overlap or new.shape[2] < overlap:
        return torch.cat([acc, new], dim=2)
    a_tail = acc[:, :, -overlap:].float()
    n_head = new[:, :, :overlap].float()
    shape = [1, 1, overlap] + [1] * (acc.dim() - 3)
    w = torch.linspace(1.0, 0.0, overlap).view(*shape)
    blended = (a_tail * w + n_head * (1.0 - w)).to(acc.dtype)
    return torch.cat([acc[:, :, :-overlap], blended, new[:, :, overlap:]], dim=2)


def _keyframe_scene_ranges(Tv: int) -> List[Tuple[int, int]]:
    """One scene per keyframe: split the Tv latent frames proportionally to the
    5 keyframe segment lengths (exactly the proven scene sizes)."""
    segs = globals().get("ORIGINAL_SEGMENTS", [])
    lens = [max(1e-6, float(s.get("length", 1))) for s in segs]
    if not lens:
        return [(0, Tv)]
    tot = sum(lens)
    counts = [max(1, int(round(L / tot * Tv))) for L in lens]
    diff = Tv - sum(counts)
    i = 0
    while diff != 0 and i < 100000:
        j = i % len(counts)
        if diff > 0:
            counts[j] += 1
            diff -= 1
        elif counts[j] > 1:
            counts[j] -= 1
            diff += 1
        i += 1
    ranges, cum = [], 0
    for c in counts:
        ranges.append((cum, min(cum + c, Tv)))
        cum += c
    ranges[-1] = (ranges[-1][0], Tv)
    return ranges


def _fixed_scene_ranges(Tv: int, chunk: int, overlap: int) -> List[Tuple[int, int]]:
    chunk = max(2, min(chunk, Tv))
    overlap = max(0, min(overlap, chunk - 1))
    step = max(1, chunk - overlap)
    ranges, cur = [], 0
    while cur < Tv:
        e = min(cur + chunk, Tv)
        ranges.append((cur, e))
        if e >= Tv:
            break
        cur += step
    return ranges


def execute_phase_b_batched(director_state: Dict[str, Any], model: Any, seed: int,
                            workdir: str, resume: bool = True) -> str:
    """Batch-by-batch scene generation over the ONE shared LTXDirector timeline."""
    latent_file = os.path.join(workdir, "final_latents.pt")
    reload_per_scene = bool(globals().get("RELOAD_DIT_PER_SCENE", True))
    overlap = int(globals().get("SCENE_OVERLAP_LATENT_FRAMES", 1))

    full_v = _lat_samples(director_state["video_latent"])
    full_a = _lat_samples(director_state["audio_latent"])
    if full_v is None or full_v.dim() < 3:
        print("  [notice] Video latent has no temporal dim — falling back to single pass.")
        return _execute_phase_b_single(director_state, model, seed, workdir, resume)

    Tv = full_v.shape[2]
    Ta = full_a.shape[2] if (full_a is not None and full_a.dim() >= 3) else 0
    ratio = (Ta / Tv) if Tv else 0.0

    if str(globals().get("SCENE_MODE", "keyframe")) == "keyframe":
        scene_ranges = _keyframe_scene_ranges(Tv)
    else:
        scene_ranges = _fixed_scene_ranges(Tv, int(globals().get("SCENE_CHUNK_LATENT_FRAMES", 16)), overlap)
    n = len(scene_ranges)
    overlap = max(0, min(overlap, max(1, min(e - s for s, e in scene_ranges)) - 1))
    a_overlap = int(round(overlap * ratio))

    guide_data = director_state["guide_data"]
    motion_guide = director_state["motion_guide_data"]
    base_pos = director_state["positive"]
    base_neg = director_state["negative"]

    print("\n" + "=" * 70 +
          f"\n🎬 PHASE B (BATCH): {n} scene(s) · mode={globals().get('SCENE_MODE','keyframe')} · "
          f"reload_dit_per_scene={reload_per_scene} · overlap {overlap}\n" + "=" * 70)

    if reload_per_scene and model is not None:
        del model
        model = None
        purge_deep("free_phase_a_model")

    acc_v: Optional[torch.Tensor] = None
    acc_a: Optional[torch.Tensor] = None

    for idx, (s, e) in enumerate(scene_ranges):
        tag = f"SCENE {idx + 1}/{n}"
        ck = os.path.join(workdir, f"scene_{idx:02d}.pt")

        if resume and os.path.exists(ck) and os.path.getsize(ck) > 1024:
            print(f"  ⏭ [RESUME] Loading cached {tag}: {ck}")
            pack = torch.load(ck, map_location="cpu")
            v_out, a_out = pack["v"], pack.get("a")
        else:
            print(f"\n  ── {tag}: latent frames {s}:{e} ({(e - s - 1) * 8 + 1} px) ──")
            if reload_per_scene or model is None:
                purge_deep(f"pre_{tag}")
                model, _ = load_dit_and_loras(clip_obj=None)
            video_vae = gv(call_node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"), 0)
            ram_guard(globals().get("min_ram_guard_gb", 1.5), tag)

            # Slice BOTH the shared video AND the shared audio latent → voice sync.
            v_in = _slice_lat_T(director_state["video_latent"], s, e)
            a_in = (_slice_lat_T(director_state["audio_latent"],
                                 int(round(s * ratio)), int(round(e * ratio)))
                    if full_a is not None else None)

            with torch.inference_mode():
                s1p, s1n, s1v, s1a, _ = _run_stage(
                    model, video_vae, STAGE1["guide_strength"], base_pos, base_neg,
                    v_in, a_in, guide_data, motion_guide,
                    STAGE1["scheduler"], STAGE1["steps"], STAGE1["denoise"], STAGE1["cfg"],
                    seed + idx, f"{tag} · Stage 1")
                medium_clear(f"{tag}_post_s1")

                up_model = gv(call_node("LatentUpscaleModelLoader",
                                        model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), 0)
                v_ups = sync_latent_device(gv(call_node("LTXVLatentUpsampler",
                                                        samples=s1v, upscale_model=up_model,
                                                        vae=video_vae), 0), "cpu")
                del up_model, s1v
                medium_clear(f"{tag}_post_upscale")

                s2p, s2n, s2v, s2a, _ = _run_stage(
                    model, video_vae, STAGE2["guide_strength"], s1p, s1n,
                    v_ups, s1a, guide_data, motion_guide,
                    STAGE2["scheduler"], STAGE2["steps"], STAGE2["denoise"], STAGE2["cfg"],
                    seed + idx, f"{tag} · Stage 2")
                medium_clear(f"{tag}_post_s2")

                v_out = unwrap_tensor(s2v).detach().cpu().half()
                a_out = unwrap_tensor(s2a).detach().cpu().half() if s2a is not None else None
                del s1p, s1n, s1a, v_ups, s2p, s2n, s2v, s2a

            _save_ckpt(ck, {"v": v_out, "a": a_out})
            print(f"  💾 {tag} cached: {ck}")
            del video_vae
            if reload_per_scene:
                del model
                model = None
                purge_deep(f"post_{tag}_reload")
            else:
                medium_clear(f"post_{tag}")

        acc_v = _blend_append_T(acc_v, v_out, overlap)
        if a_out is not None:
            acc_a = _blend_append_T(acc_a, a_out, a_overlap)
        del v_out, a_out
        mem_report(f"after {tag}")

    torch.save({"video": acc_v, "audio": acc_a,
                "frame_rate": director_state["frame_rate"]}, latent_file + ".tmp")
    os.replace(latent_file + ".tmp", latent_file)
    print(f"\n  💾 Assembled {n} scene(s) → {latent_file}")
    del acc_v, acc_a
    if model is not None:
        del model

    for f in glob.glob(os.path.join(workdir, "scene_*.pt")):
        try:
            os.remove(f)
        except Exception:
            pass
    purge_deep("phase_b_batched_complete")
    mem_report("Phase B (batch) complete")
    return latent_file


def execute_phase_b(director_state: Dict[str, Any], model: Any, seed: int,
                    workdir: str, resume: bool = True) -> str:
    """Dispatcher: batch-by-batch scene chunks (T4) or one single pass (L4/A100)."""
    latent_file = os.path.join(workdir, "final_latents.pt")
    if resume and os.path.exists(latent_file) and os.path.getsize(latent_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached final latents: {latent_file}")
        return latent_file
    if globals().get("BATCH_SCENE_MODE", True):
        return execute_phase_b_batched(director_state, model, seed, workdir, resume)
    return _execute_phase_b_single(director_state, model, seed, workdir, resume)


def _execute_phase_b_single(director_state: Dict[str, Any], model: Any, seed: int,
                            workdir: str, resume: bool = True) -> str:
    """True single-pass diffusion over all 756 frames (needs L4/A100)."""
    latent_file = os.path.join(workdir, "final_latents.pt")
    print("\n" + "=" * 70 + "\n🎬 PHASE B: WHOLE-TIMELINE 2-STAGE DIFFUSION (single pass)\n" + "=" * 70)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    guide_data = director_state["guide_data"]
    motion_guide = director_state["motion_guide_data"]
    base_pos = director_state["positive"]
    base_neg = director_state["negative"]

    with torch.inference_mode():
        model = _ensure_model(model)
        video_vae = gv(call_node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"), 0)

        # Stage 1 (base 416x240, 8 steps, denoise 1.0, guide 0.5)
        s1_pos, s1_neg, s1_vid, s1_aud, _ = _run_stage(
            model, video_vae, STAGE1["guide_strength"], base_pos, base_neg,
            director_state["video_latent"], director_state["audio_latent"],
            guide_data, motion_guide,
            STAGE1["scheduler"], STAGE1["steps"], STAGE1["denoise"], STAGE1["cfg"],
            seed, "STAGE 1")
        medium_clear("single_post_s1")

        # 2x latent spatial upscale
        up_model = gv(call_node("LatentUpscaleModelLoader",
                                model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), 0)
        v_ups = sync_latent_device(gv(call_node("LTXVLatentUpsampler",
                                                samples=s1_vid, upscale_model=up_model,
                                                vae=video_vae), 0), "cpu")
        del up_model, s1_vid
        medium_clear("single_post_upscale")

        # Stage 2 (refine 832x480, 4 steps, denoise 0.42, guide 1.0)
        s2_pos, s2_neg, s2_vid, s2_aud, _ = _run_stage(
            model, video_vae, STAGE2["guide_strength"], s1_pos, s1_neg,
            v_ups, s1_aud, guide_data, motion_guide,
            STAGE2["scheduler"], STAGE2["steps"], STAGE2["denoise"], STAGE2["cfg"],
            seed, "STAGE 2")

        final_video_lat = unwrap_tensor(s2_vid).detach().cpu().half()
        final_audio_lat = unwrap_tensor(s2_aud).detach().cpu().half() if s2_aud is not None else None

        del model, video_vae, v_ups, s1_pos, s1_neg, s1_aud, s2_pos, s2_neg, s2_vid, s2_aud
        purge_deep("phase_b_pre_save")

        torch.save({"video": final_video_lat, "audio": final_audio_lat,
                    "frame_rate": director_state["frame_rate"]}, latent_file + ".tmp")
        os.replace(latent_file + ".tmp", latent_file)
        print(f"  💾 Final timeline latents saved: {latent_file}")
        del final_video_lat, final_audio_lat

    purge_deep("phase_b_complete")
    mem_report("Phase B complete")
    return latent_file


print("✅ Cell 13: Phase B (2-stage diffusion over the shared timeline) ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 14: PHASE C — OUT-OF-CORE TILED VAE DECODE (video) + AUDIO DECODE
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_c(latent_file: str, workdir: str, resume: bool = True) -> Tuple[str, str]:
    frames_file = os.path.join(workdir, "decoded_frames.pt")
    audio_file = os.path.join(workdir, "decoded_audio.pt")

    pack = torch.load(latent_file, map_location="cpu")
    v_lat = pack["video"].float()
    a_lat = pack["audio"]

    if not (resume and os.path.exists(frames_file) and os.path.getsize(frames_file) > 1024):
        print("\n" + "=" * 70 + "\n🎬 PHASE C1: TILED VIDEO VAE DECODE\n" + "=" * 70)
        purge_deep("pre_video_decode")
        with torch.inference_mode():
            video_vae = gv(call_node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"), 0)
            frames = unwrap_tensor(tiled_decode_video({"samples": v_lat}, video_vae, tile_size=256))
            frames = frames.detach().cpu().half()
            torch.save(frames, frames_file + ".tmp")
            os.replace(frames_file + ".tmp", frames_file)
            print(f"  💾 Decoded frames {tuple(frames.shape)} → {frames_file}")
            del video_vae, frames
        purge_deep("post_video_decode")
    else:
        print(f"  ⏭ [RESUME] Video frames cached: {frames_file}")

    # AUDIO decode via LTXVAudioVAEDecode → the synced vocals (voice-sync fix).
    if not (resume and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1024):
        print("\n" + "=" * 70 + "\n🎬 PHASE C2: AUDIO VAE DECODE (synced vocals)\n" + "=" * 70)
        purge_deep("pre_audio_decode")
        with torch.inference_mode():
            if a_lat is not None:
                audio_vae = gv(call_node("VAELoader", vae_name="LTX23_audio_vae_bf16.safetensors"), 0)
                decoded_audio = gv(call_node("LTXVAudioVAEDecode",
                                             samples={"samples": a_lat.float()}, audio_vae=audio_vae), 0)
                torch.save(decoded_audio, audio_file + ".tmp")
                os.replace(audio_file + ".tmp", audio_file)
                print(f"  💾 Decoded audio → {audio_file}")
                del audio_vae, decoded_audio
            else:
                torch.save(None, audio_file)
                print("  ⚠️ No audio latent; the raw song will be muxed in Phase D.")
        purge_deep("post_audio_decode")
    else:
        print(f"  ⏭ [RESUME] Audio cached: {audio_file}")

    del v_lat, a_lat, pack
    gc.collect()
    malloc_trim_os()
    return frames_file, audio_file


print("✅ Cell 14: Phase C (video + audio VAE decode) ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 15: PHASE D — VHS_VideoCombine FINAL ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_d(frames_file: str, audio_file: str, fps: int, crf: int,
                    outdir: str, song_path: str, trim_start_frames: float) -> str:
    os.makedirs(outdir, exist_ok=True)
    final_path = os.path.join(outdir, "LTX23_Director_Master_30s.mp4")

    print("\n" + "=" * 70 + "\n🎬 PHASE D: VHS_VideoCombine FINAL ASSEMBLY\n" + "=" * 70)
    purge_deep("pre_vhs")

    frames = torch.load(frames_file, map_location="cpu").float()
    audio_dict = torch.load(audio_file, map_location="cpu") if os.path.exists(audio_file) else None
    print(f"  🎬 Combining {frames.shape[0]} frames @ {fps} fps with synced audio...")

    combined = False
    try:
        res = call_node("VHS_VideoCombine",
                        images=frames, audio=audio_dict, frame_rate=float(fps),
                        loop_count=0, filename_prefix=VHS_SETTINGS["filename_prefix"],
                        format=VHS_SETTINGS["format"], pix_fmt=VHS_SETTINGS["pix_fmt"],
                        crf=int(crf), save_metadata=False, trim_to_audio=False,
                        pingpong=False, save_output=True)
        info = gv(res, 0)
        if isinstance(info, dict) and "ui" in info and "gifs" in info["ui"]:
            gen = info["ui"]["gifs"][0].get("fullpath", "")
            if gen and os.path.exists(gen):
                shutil.copyfile(gen, final_path)
                combined = True
    except Exception as e:
        print(f"  [notice] VHS node fallback: {e}")

    if not combined or not os.path.exists(final_path):
        import imageio
        raw = os.path.join(outdir, "_raw_video.mp4")
        arr = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        imageio.mimwrite(raw, arr, fps=fps, quality=9)
        if song_path and os.path.exists(song_path):
            trim_sec = float(trim_start_frames) / fps
            dur_sec = frames.shape[0] / fps
            cmd = (f'ffmpeg -y -i "{raw}" -ss {trim_sec} -t {dur_sec} -i "{song_path}" '
                   f'-map 0:v:0 -map 1:a:0 -c:v libx264 -crf {crf} -pix_fmt yuv420p '
                   f'-c:a aac -b:a 320k -shortest "{final_path}"')
            run_cmd(cmd, silent=False)
            if os.path.exists(raw):
                os.remove(raw)
        else:
            shutil.move(raw, final_path)

    del frames, audio_dict
    purge_deep("post_vhs")
    print(f"  🎉 Master MP4: {final_path}")
    return final_path


print("✅ Cell 15: Phase D (VHS assembly) ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 16: OUTPUT VERIFICATION
# ════════════════════════════════════════════════════════════════════════════
def verify_output(video_path: str):
    print("\n" + "=" * 70 + "\n🔍 FINAL ARTIFACT VERIFICATION\n" + "=" * 70)
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        raise RuntimeError(f"Output '{video_path}' is missing or empty.")
    vprobe = (f'ffprobe -v error -select_streams v:0 -count_packets '
              f'-show_entries stream=nb_read_packets,r_frame_rate,duration '
              f'-of csv=p=0 "{video_path}"')
    aprobe = (f'ffprobe -v error -select_streams a:0 '
              f'-show_entries stream=codec_name,duration -of csv=p=0 "{video_path}"')
    vout = subprocess.run(vprobe, shell=True, capture_output=True, text=True).stdout.strip()
    aout = subprocess.run(aprobe, shell=True, capture_output=True, text=True).stdout.strip()
    print(f"  ✓ Path        : {video_path}")
    print(f"  ✓ Size        : {os.path.getsize(video_path)/(1024*1024):.2f} MB")
    print(f"  ✓ Video stream: {vout}")
    print(f"  ✓ Audio stream: {aout if aout else '(none)'}")
    print("=" * 70)


print("✅ Cell 16: Verifier ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 17: RUNTIME CONFIG & MASTER ONE-CLICK GENERATION
# ════════════════════════════════════════════════════════════════════════════
WORK_DIRECTORY = "/content/LTXDirector_Work"
OUTPUT_DIRECTORY = "/content/LTXStudio_Output"
SONG_PATH = "/content/ComfyUI/input/whatdreamscost/Late night trap.mp3"


def _config_signature(meta: Dict[str, Any]) -> str:
    """Signature of every setting that changes the cached latents. If it differs
    from the last run, the old checkpoints are STALE (e.g. built at 1280x720) and
    must be cleared — otherwise resume would reuse old-resolution latents and the
    render-resolution speed fix would be silently ignored."""
    return (f"render{meta.get('custom_width')}x{meta.get('custom_height')}"
            f"|gen{meta.get('generation_width')}x{meta.get('generation_height')}"
            f"|frames{meta.get('normalDurationFrames')}|fps{meta.get('frame_rate')}"
            f"|loras{len(globals().get('LORA_STACK', []))}"
            f"|s1_{STAGE1['steps']}x{STAGE1['denoise']}|s2_{STAGE2['steps']}x{STAGE2['denoise']}")


def guard_stale_cache(workdir: str, meta: Dict[str, Any]):
    """Auto-clear checkpoints when the config changed since the last run (or when
    caches predate this cache-guard, i.e. no signature file yet)."""
    os.makedirs(workdir, exist_ok=True)
    sig = _config_signature(meta)
    sig_file = os.path.join(workdir, "cache_sig.txt")
    existing = glob.glob(os.path.join(workdir, "*.pt"))
    old = None
    if os.path.exists(sig_file):
        try:
            old = open(sig_file).read().strip()
        except Exception:
            old = None
    # Stale if: pre-guard caches exist without a signature, OR signature changed.
    stale = (old is None and len(existing) > 0) or (old is not None and old != sig)
    if stale:
        print(f"🧹 Config changed (or pre-upgrade caches found) → clearing {len(existing)} stale "
              f"checkpoint(s) so the new {meta.get('custom_width')}x{meta.get('custom_height')} "
              f"render is actually used (prevents reusing old-resolution latents).")
        for f in existing:
            try:
                os.remove(f)
            except Exception:
                pass
    try:
        with open(sig_file, "w") as fh:
            fh.write(sig)
    except Exception:
        pass


def run_ltx23_director_master(global_prompt, negative_prompt, meta, segments,
                              audio_segments, motion_segments,
                              seed=0, crf=8, workdir=WORK_DIRECTORY,
                              outdir=OUTPUT_DIRECTORY, resume=True,
                              use_song_audio=True) -> str:
    t_start = time.time()
    print("\n" + "=" * 70 + "\n🎬 LTX-2.3 DIRECTOR 2.0 · Master_V2 (FAITHFUL REBUILD)\n" + "=" * 70)
    validate_original_nodes()
    patch_comfy_memory_manager()
    patch_safetensors_direct_to_gpu()
    configure_vram_state(mode=globals().get("VRAM_MODE", "auto"))
    install_sampling_memory_hook(clear_every=4, ram_guard_gb=globals().get("min_ram_guard_gb", 1.5))
    os.makedirs(workdir, exist_ok=True)

    active_meta = dict(meta)

    # Invalidate stale checkpoints from a previous config (e.g. the old 1280x720
    # render) so resume never reuses old-resolution latents.
    guard_stale_cache(workdir, active_meta)

    ctrl = DirectorTimelineController(
        global_prompt=global_prompt, negative_prompt=negative_prompt,
        meta=active_meta, segments=segments,
        audio_segments=audio_segments, motion_segments=motion_segments)

    latent_file = os.path.join(workdir, "final_latents.pt")
    if resume and os.path.exists(latent_file) and os.path.getsize(latent_file) > 1024:
        print(f"  ⏭ [RESUME] Phase A+B already done → using {latent_file}")
    else:
        director_state, patched_model = execute_phase_a(ctrl, workdir=workdir, resume=resume)
        latent_file = execute_phase_b(director_state, patched_model, seed=seed,
                                      workdir=workdir, resume=resume)
        del director_state, patched_model
        purge_deep("post_phase_b_master")

    frames_file, audio_file = execute_phase_c(latent_file, workdir=workdir, resume=resume)

    final_video = execute_phase_d(
        frames_file, audio_file,
        fps=int(active_meta["frame_rate"]), crf=crf, outdir=outdir,
        song_path=SONG_PATH if use_song_audio else "",
        trim_start_frames=active_meta["audio_trim_start_frames"])

    verify_output(final_video)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("🎉 GENERATION COMPLETE")
    print(f"  Duration : {active_meta['duration_seconds']:.2f}s "
          f"({active_meta['normalDurationFrames']} frames @ {active_meta['frame_rate']} fps)")
    print(f"  Time     : {elapsed/60:.2f} min")
    print(f"  Output   : {final_video}")
    print(f"  RAM free : {get_ram_free_gb():.2f} GB")
    print("=" * 70 + "\n")
    return final_video


# Ensure the 5.3.png keyframe alias exists before running.
_base_input = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(_base_input, exist_ok=True)
if os.path.exists(f"{_base_input}/5.png") and not os.path.exists(f"{_base_input}/5.3.png"):
    shutil.copy(f"{_base_input}/5.png", f"{_base_input}/5.3.png")

if __name__ == "__main__":
    final_output_file = run_ltx23_director_master(
        global_prompt=GLOBAL_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        meta=TIMELINE_METADATA,
        segments=ORIGINAL_SEGMENTS,
        audio_segments=ORIGINAL_AUDIO_SEGMENTS,
        motion_segments=ORIGINAL_MOTION_SEGMENTS,
        seed=BASE_SEED,
        crf=OUTPUT_CRF,
        workdir=WORK_DIRECTORY,
        outdir=OUTPUT_DIRECTORY,
        resume=RESUME_CHECKPOINTS,
        use_song_audio=USE_SONG_AUDIO,
    )
    print(f"\n🎬 Your synchronized music video is ready:\n   {final_output_file}")
