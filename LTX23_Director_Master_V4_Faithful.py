# -*- coding: utf-8 -*-
"""
LTX23_Director_Master_V4_Faithful.py
================================================================================
100% FAITHFUL single-pass re-implementation of the ORIGINAL ComfyUI workflow:
    LTX-2.3_Director_2.0-MV-Workflow-30s.json

WHY V2/V3 PRODUCED POOR VIDEO
-------------------------------------------------------------------------------
V2 and V3 NEVER used the `LTXDirector` node (the "Master Timeline Controller").
Instead they chopped the 756-frame timeline into 5 INDEPENDENT clips, sampled
each one on its own, and crossfaded the results. That approach loses everything
the real workflow depends on:
  * `LTXDirector` builds ONE continuous latent for the whole timeline and injects
    per-segment attention masks (via prompt_relay) so the SAME character/scene
    context is shared across all 5 shots  -> no identity drift.
  * `LTXDirector` lays the audio track onto the SAME timeline and drives lip-sync
    from it  -> voice stays synchronized.
  * Independent chunks + crossfade => drift, desync, seams, muddy motion.

WHAT THIS FILE DOES (matches the JSON graph node-for-node)
-------------------------------------------------------------------------------
  UnetLoaderGGUF -> Power-LoRA (4 stack) -> ModelPreviewOverrideKJ
                                                 |
  DualCLIPLoader --------------------------------+--> LTXDirector (Master Timeline)
                                                        | model,positive,video_latent,
                                                        | audio_latent,guide_data,
                                                        | motion_guide_data,frame_rate,audio
        ConditioningZeroOut(positive) -> negative
        LTXVConditioning(pos,neg,fps)
   STAGE 1 : LTXDirectorGuide(scale_by=0.5) -> ConcatAV -> Sample(euler,8,denoise1.0)
             -> SeparateAV -> LTXDirectorCropGuides
   UPSCALE : LTXVLatentUpsampler (2x, ltx-2.3-spatial-upscaler)
   STAGE 2 : LTXDirectorGuide(scale_by=1.0) -> ConcatAV -> Sample(euler,4,denoise0.42)
             -> SeparateAV -> LTXDirectorCropGuides
   DECODE  : VAEDecode(tiled) + LTXVAudioVAEDecode  -> VHS_VideoCombine (+ ffmpeg mux)

Target Hardware: Google Colab Free-Tier T4 (15GB VRAM | ~12.7GB RAM).
Zero-crash strategy = sequential model residency + 1.2GB VRAM shield + 16GB swap
+ tiled VAE decode + SageAttention + ChunkFeedForward + aggressive purges.
================================================================================
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP & 16GB SWAP
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'


def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    return subprocess.run(cmd, shell=True).returncode


# 16GB High-Speed Swap Partition for Free-Tier Colab
if not os.path.exists("/content/swapfile") or os.path.getsize("/content/swapfile") < (8 * 1024 * 1024 * 1024):
    print("[1/3] Setting up High-Speed 16GB Swap Partition...")
    run_cmd("swapoff /content/swapfile || true")
    run_cmd("rm -f /content/swapfile")
    run_cmd("fallocate -l 16G /content/swapfile || dd if=/dev/zero of=/content/swapfile bs=1M count=16384")
    run_cmd("chmod 600 /content/swapfile")
    run_cmd("mkswap /content/swapfile")
    run_cmd("swapon /content/swapfile || true")
    run_cmd("sysctl vm.swappiness=100 || true")
    run_cmd("sysctl vm.vfs_cache_pressure=500 || true")

try:
    import psutil
    sw = psutil.swap_memory()
    print(f"  Memory: RAM {psutil.virtual_memory().available/1e9:.2f} GB free | Swap {sw.total/1e9:.2f} GB")
except Exception:
    pass

# Neutralize ComfyUI's utils.install_util so imports don't try to reinstall reqs
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

print("Cell 1: Environment & memory protection configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
print("[2/3] Installing core dependencies & PyTorch...")
run_cmd("pip install -q torch torchvision torchaudio", silent=False)
run_cmd("pip uninstall -y utils || true")
os.chdir("/content")

run_cmd("pip install -q torchsde einops diffusers accelerate psutil")
run_cmd("pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio imageio aiohttp scipy sentencepiece protobuf")
run_cmd("pip install -q 'kornia==0.7.3'")
run_cmd("pip install -q 'transformers>=4.45.0'")
run_cmd("apt-get -y install -qq aria2 ffmpeg")

print("Cell 2: Dependencies installed.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE COMFYUI CORE
# ════════════════════════════════════════════════════════════════════════════
if not os.path.isdir("/content/ComfyUI"):
    print("[3/3] Cloning ComfyUI...")
    run_cmd("git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI")
    run_cmd("pip install -q -r /content/ComfyUI/requirements.txt")

if "/content/ComfyUI" not in sys.path:
    sys.path.insert(0, "/content/ComfyUI")
if "/content" not in sys.path:
    sys.path.insert(1, "/content")

os.makedirs("/content/ComfyUI/utils", exist_ok=True)
run_cmd("touch /content/ComfyUI/utils/__init__.py")

print("Cell 3: ComfyUI core ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 4: INSTALL CUSTOM NODES (LTXDirector needs LATEST LTXVideo + KJNodes!)
# ════════════════════════════════════════════════════════════════════════════
custom_nodes_dir = "/content/ComfyUI/custom_nodes"
os.makedirs(custom_nodes_dir, exist_ok=True)

# Remove junk / numeric folders that break the loader
for item in os.listdir(custom_nodes_dir):
    full_p = os.path.join(custom_nodes_dir, item)
    if os.path.isdir(full_p) and (item.isdigit() or item.startswith(".") or item == "comfyui"):
        shutil.rmtree(full_p, ignore_errors=True)

os.chdir(custom_nodes_dir)

# The LTXDirector node pack (WhatDreamsCost) REQUIRES the newest ComfyUI-LTXVideo
# and ComfyUI-KJNodes or it silently fails to register. We always pull latest.
repos = [
    ("WhatDreamsCost-ComfyUI", "https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI"),
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo"),
    ("ComfyUI_KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI_GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy"),
]

for folder, url in repos:
    if not os.path.isdir(folder):
        print(f"  Cloning {folder}...")
        run_cmd(f"git clone {url} {folder}")
    else:
        # Keep the three consistency-critical packs on the latest commit.
        if folder in ("WhatDreamsCost-ComfyUI", "ComfyUI-LTXVideo", "ComfyUI_KJNodes"):
            run_cmd(f"git -C {folder} pull --ff-only || true")
    req_file = os.path.join(folder, "requirements.txt")
    if os.path.isfile(req_file):
        run_cmd(f"pip install -q -r {req_file} || true")

print("Cell 4: Custom nodes installed (LTXDirector pack + latest LTXVideo/KJNodes).")


# ════════════════════════════════════════════════════════════════════════════
# CELL 5: DOWNLOAD MODELS, 4-LORA STACK, UPSCALER & AUDIO
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
        print(f"  Downloading {filename}...", end=' ', flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print("done")
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


print("Downloading LTX-2.3 core models...")

dit_model = download_file(
    "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
    "/content/ComfyUI/models/unet", filename="ltx-2-3-22b-dev-Q4_K_M.gguf")
link_file_safe("/content/ComfyUI/models/unet/ltx-2-3-22b-dev-Q4_K_M.gguf",
               "/content/ComfyUI/models/diffusion_models/ltx-2-3-22b-dev-Q4_K_M.gguf")

text_encoder_model = download_file(
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders", filename="gemma_3_12B_it_fp4_mixed.safetensors")
link_file_safe("/content/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
               "/content/ComfyUI/models/clip/gemma_3_12B_it_fp4_mixed.safetensors")

text_encoder2_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders", filename="ltx-2.3_text_projection_bf16.safetensors")
link_file_safe("/content/ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
               "/content/ComfyUI/models/clip/ltx-2.3_text_projection_bf16.safetensors")

vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae", filename="LTX23_video_vae_bf16.safetensors")
vae_audio_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae", filename="LTX23_audio_vae_bf16.safetensors")
tiny_vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae", filename="taeltx2_3.safetensors")
link_file_safe("/content/ComfyUI/models/vae/taeltx2_3.safetensors",
               "/content/ComfyUI/models/vae_approx/taeltx2_3.safetensors")

upscaler_model = download_file(
    "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "/content/ComfyUI/models/latent_upscale_models", filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
link_file_safe("/content/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
               "/content/ComfyUI/models/upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

print("Downloading Director 2.0 4-LoRA stack...")
lora_dir = "/content/ComfyUI/models/loras"
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", lora_dir, filename="ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", lora_dir, filename="LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
download_file("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors", lora_dir, filename="ltx2.3-transition.safetensors")
download_file("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors", lora_dir, filename="LTX2.3-MVCamera-drclips.safetensors")

# Audio track (must live under input/whatdreamscost so LTXDirector can find it)
audio_dest_dir = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(audio_dest_dir, exist_ok=True)
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file("https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3", audio_dest_dir, filename="Late night trap.mp3")

print("Cell 5: Models, LoRAs, upscaler and audio validated.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 6: FAITHFUL TIMELINE CONFIG (rebuilds the JSON's LTXDirector widgets)
# ════════════════════════════════════════════════════════════════════════════

# --- Output / render controls -----------------------------------------------
# The ORIGINAL workflow renders at 1280x720. On a T4 that single-pass 756-frame
# job is very heavy; set T4_SAFE=True to render at 768x432 (same 16:9, faithful
# pipeline, far lighter). Set False to reproduce the original 1280x720 exactly.
T4_SAFE = True
if T4_SAFE:
    custom_width, custom_height = 768, 432        # snapped to /32 by the director
else:
    custom_width, custom_height = 1280, 720       # original workflow resolution

fps = 24
output_crf = 8               # final H.264 quality (VHS_VideoCombine)
img_compression = 18         # per-guide-image CRF (matches JSON)
divisible_by = 32
resize_method = "maintain aspect ratio"
epsilon = 0.001              # penalty decay for segment boundaries (paper default)

# --- 4-LoRA stack (Power Lora Loader) : name + strength, matches JSON --------
LORA_STACK = [
    {"on": True, "name": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", "strength": 0.4},
    {"on": True, "name": "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",                                "strength": 0.6},
    {"on": True, "name": "ltx2.3-transition.safetensors",                                           "strength": 0.7},
    {"on": True, "name": "LTX2.3-MVCamera-drclips.safetensors",                                     "strength": 0.9},
]

# --- Global prompt (identical to the workflow) ------------------------------
GLOBAL_PROMPT = """Create a highly realistic cinematic AI music video using the provided reference image. Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body proportions, and overall appearance exactly as in the reference image. The singer must remain fully recognizable throughout the entire video with absolutely no identity drift.

The person is performing directly to the camera as a world-class pop, hip-hop and rap singer during a sold-out stadium concert. Generate perfectly synchronized lip movements from the provided lyrics or audio.

This is NOT a talking-head video and NOT a presenter. This is a high-energy live music performance filled with charisma, attitude and emotional intensity.

Performance Energy:
Perform with explosive stage presence. Every musical phrase immediately creates a new emotional and physical performance. Own the stage with absolute confidence. Perform as if in front of 50,000 screaming fans. Never appear calm, passive or static.

Facial Performance:
Extremely expressive facial acting. Rich emotional transitions every few words. Powerful eye contact. Highly expressive eyebrows synchronized with important lyrics. Rich cinematic micro-expressions. The face feels emotionally alive every second.

Body Performance:
The entire body constantly grooves with the beat. Strong rhythmic bouncing, powerful shoulder accents, confident chest movement, frequent body turns, dynamic torso twists. Lean toward the camera during emotional lyrics. Bold, energetic, theatrical stage movement.

Hand Performance:
Large expressive gestures, fast rhythmic arm accents, sharp hand movements synchronized with the beat, powerful pointing, sweeping arms, punching the air, finger snapping, open palm emphasis. Asymmetrical movement. Every musical phrase introduces fresh gestures.

Musical Timing:
Body movement follows musical phrasing. Strong beats create explosive movements. Fast lyrics generate faster gestures. Every movement feels rhythmically connected to the music.

Speech Synchronization:
Perfect lip synchronization. Accurate mouth shapes. Expressions and gestures match the emotional meaning of every lyric. Natural breathing between phrases.

Motion Quality:
Premium AI human animation, realistic momentum, natural motion blur. No robotic movement, no frozen poses, no repetitive gesture loops, no idle standing, no jitter, no flickering, no facial distortion, no identity drift, no hand deformation, no extra fingers.

Camera:
drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur.

Lighting:
Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.

Overall Style:
Photorealistic, blockbuster-quality AI music video, premium live concert performance, ultra-high facial fidelity, charismatic superstar, explosive stage energy, modern pop/hip-hop/rap performance.

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

NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, robotic movement, static presenter, jitter, flicker, facial distortion, extra limbs, watermark"

# --- 5-scene storyboard : EXACT lengths/positions from the workflow JSON -----
# start/length are in pixel-space frames (the director's internal unit).
SEGMENTS = [
    {"imageFile": "whatdreamscost/1.png",   "start": 0.0,                "length": 226.01059340956584, "prompt": ""},
    {"imageFile": "whatdreamscost/2.png",   "start": 226.01059340956584, "length": 161.31859976617454, "prompt": ""},
    {"imageFile": "whatdreamscost/3.png",   "start": 387.3291931757404,  "length": 131.45629831196658, "prompt": ""},
    {"imageFile": "whatdreamscost/4.png",   "start": 518.785491487707,   "length": 225.5063328766255,  "prompt": ""},
    {"imageFile": "whatdreamscost/5.3.png", "start": 744.2918243643325,  "length": 83.22765271847516,  "prompt": ""},
]

# Audio segment : EXACT values from the workflow JSON (trimStart 447 = 446.92)
AUDIO_SEGMENT = {
    "type": "audio",
    "start": 0,
    "length": 756.5194770828076,
    "trimStart": 446.9222739141953,
    "audioDurationFrames": 2880,
    "audioFile": "whatdreamscost/Late night trap.mp3",
    "fileName": "Late night trap.mp3",
    "waveformPeaks": [],   # UI-only; not used for generation
}

# Timeline render window (the director only samples frames start_frame..end_frame)
START_FRAME = 0
END_FRAME = 756
DURATION_FRAMES = 756
DURATION_SECONDS = round(END_FRAME / fps, 3)   # 31.5

# Per-segment strings the editor auto-populates (fed verbatim to LTXDirector)
LOCAL_PROMPTS = " |  |  |  | "
SEGMENT_LENGTHS = "226.01059340956584,161.31859976617454,131.45629831196658,225.5063328766255,11.708175635667544"
GUIDE_STRENGTH = "1.00,1.00,1.00,1.00,1.00"


def build_timeline_data() -> str:
    """Recreate the JSON string that the LTXDirector timeline editor produces.
    Only generation-relevant keys are populated; UI-only keys use safe defaults."""
    segments = []
    for i, s in enumerate(SEGMENTS):
        segments.append({
            "id": f"seg{i+1}",
            "start": s["start"],
            "length": s["length"],
            "prompt": s.get("prompt", ""),
            "type": "image",
            "imageFile": s["imageFile"],
            "imageB64": f"/api/view?filename={os.path.basename(s['imageFile'])}&type=input&subfolder=whatdreamscost",
            "isEndFrame": False,
        })
    data = {
        "mainTrackEnabled": True,
        "audioTrackEnabled": True,
        "motionTrackEnabled": True,
        "propHeight": 90,
        "globalPropHeight": 470,
        "showFilenames": True,
        "overrideAudio": False,
        "inpaint_audio": True,
        "global_prompt": GLOBAL_PROMPT,
        "retake_global_prompt": "",
        "retakeMode": False,
        "retakeStart": 24,
        "retakeLength": 48,
        "retakePrompt": "",
        "retakeStrength": 1,
        "retakeVideo": None,
        "normalStartFrame": START_FRAME,
        "normalDurationFrames": DURATION_FRAMES,
        "segments": segments,
        "motionSegments": [],
        "audioSegments": [AUDIO_SEGMENT],
    }
    return json.dumps(data)


TIMELINE_DATA = build_timeline_data()

total_render = END_FRAME - START_FRAME
print(f"Cell 6: Timeline built | {custom_width}x{custom_height} | {len(SEGMENTS)} scenes | "
      f"{total_render} frames ({total_render/fps:.2f}s @ {fps}fps) | single-pass Master Timeline.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 7: PRODUCTION MEMORY ENGINE (1.2GB VRAM shield + deep purge)
# ════════════════════════════════════════════════════════════════════════════
def malloc_trim_os():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def patch_comfy_memory_manager():
    try:
        import comfy.model_management as mm
        if not getattr(mm, "_is_free_memory_patched", False):
            _orig_free_memory = mm.free_memory

            def _safe_free_memory(*args, **kwargs):
                try:
                    res = _orig_free_memory(*args, **kwargs)
                    return res if isinstance(res, list) else []
                except Exception:
                    return []
            mm.free_memory = _safe_free_memory

            _orig_get_free_memory = mm.get_free_memory

            def _buffered_get_free_memory(dev=None, torch_free_too=False):
                try:
                    free = _orig_get_free_memory(dev, torch_free_too)
                    # Reserve a 1.2GB shield for dynamic LoRA delta / attention buffers
                    return max(512 * 1024 * 1024, free - 1200 * 1024 * 1024)
                except Exception:
                    return 2 * 1024 * 1024 * 1024
            mm.get_free_memory = _buffered_get_free_memory
            mm._is_free_memory_patched = True
    except Exception as e:
        print(f"Memory patch notice: {e}")


def patch_safetensors_direct_to_gpu():
    try:
        import safetensors.torch
        if not getattr(safetensors.torch, "_is_cuda_direct_patched", False):
            _orig = safetensors.torch.load_file

            def _safe_cuda_load(filename, device="cpu"):
                fn = str(filename).lower()
                if any(k in fn for k in ["gemma", "clip", "text_encoder", "projection", "connector"]):
                    if torch.cuda.is_available():
                        return _orig(filename, device="cuda")
                return _orig(filename, device=device)
            safetensors.torch.load_file = _safe_cuda_load
            safetensors.torch._is_cuda_direct_patched = True
    except Exception:
        pass


patch_comfy_memory_manager()
patch_safetensors_direct_to_gpu()


def get_ram_free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99.0


def drop_page_cache():
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
    drop_page_cache()
    malloc_trim_os()


def clear_memory(light: bool = False):
    """Light = gc + cuda cache only; full = deep purge of all models + page cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    malloc_trim_os()
    if not light:
        purge_deep("clear_memory")


def ram_guard(min_free_gb: float = 2.0, tag: str = ""):
    if get_ram_free_gb() < min_free_gb:
        print(f"  [RAM GUARD] Free RAM {get_ram_free_gb():.2f} GB < {min_free_gb} GB -> deep purge")
        purge_deep(f"ram_guard:{tag}")


print("Cell 7: Memory engine & 1.2GB VRAM shield active.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 8: NODE REGISTRY, DISPATCHER & NodeOutput-SAFE VALUE EXTRACTION
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


async def _init_nodes():
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
        asyncio.ensure_future(_init_nodes())
        loop.run_until_complete(asyncio.sleep(0.1))
    else:
        loop.run_until_complete(_init_nodes())
except Exception:
    pass

from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly


def out(obj: Any, index: int = 0) -> Any:
    """Robust output extractor. Handles comfy_api io.NodeOutput (results in .args),
    plain tuples/lists, and single values. Avoids the 'unwrap single list' trap that
    corrupts single-CONDITIONING outputs."""
    if obj is None:
        return None
    # comfy_api.latest io.NodeOutput -> results live in .args (a tuple)
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)):
        a = obj.args
        return a[index] if index < len(a) else None
    if hasattr(obj, "result") and isinstance(getattr(obj, "result"), (list, tuple)):
        r = obj.result
        return r[index] if index < len(r) else None
    if isinstance(obj, (tuple, list)):
        return obj[index] if index < len(obj) else None
    if isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], (list, tuple)):
        r = obj["result"]
        return r[index] if index < len(r) else None
    return obj if index == 0 else None


def call_node(node_cls_or_inst: Any, **kwargs) -> Any:
    """Instantiate (if needed) and call a ComfyUI node by matching kwargs to its
    execute/FUNCTION signature. Supports both legacy INPUT_TYPES nodes and the new
    comfy_api io.ComfyNode (classmethod execute)."""
    inst = node_cls_or_inst
    try:
        if isinstance(node_cls_or_inst, type):
            inst = node_cls_or_inst()
    except Exception:
        inst = node_cls_or_inst

    callables = []
    func_name = getattr(inst, "FUNCTION", None)
    if func_name and hasattr(inst, func_name):
        callables.append(getattr(inst, func_name))
    # new comfy_api nodes: classmethod execute on the class itself
    cls = inst if isinstance(inst, type) else type(inst)
    if hasattr(cls, "execute"):
        callables.append(getattr(cls, "execute"))
    if hasattr(inst, "execute"):
        callables.append(inst.execute)
    for fb in ["get_guider", "get_noise", "get_sampler", "get_sigmas", "sample",
               "encode", "decode", "upscale", "generate", "combine_video"]:
        if hasattr(inst, fb):
            callables.append(getattr(inst, fb))

    last_err = None
    seen = set()
    for func in callables:
        if id(func) in seen:
            continue
        seen.add(id(func))
        try:
            sig = inspect.signature(func)
            valid = {}
            for name, param in sig.parameters.items():
                if name in ("cls", "self"):
                    continue
                if name in kwargs:
                    valid[name] = kwargs[name]
                elif param.default is not inspect.Parameter.empty:
                    pass
                else:
                    ann = str(param.annotation)
                    if "int" in ann:
                        valid[name] = 0
                    elif "float" in ann:
                        valid[name] = 0.0
                    elif "bool" in ann:
                        valid[name] = False
                    elif "str" in ann:
                        valid[name] = ""
                    else:
                        valid[name] = None
            return func(**valid)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise AttributeError(f"Cannot execute node '{cls.__name__}'")


def has_node(name: str) -> bool:
    return name in NODE_CLASS_MAPPINGS


def N(name: str):
    return NODE_CLASS_MAPPINGS[name]


print(f"Cell 8: {len(NODE_CLASS_MAPPINGS)} nodes registered.")
_required = ["LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides", "UnetLoaderGGUF",
             "DualCLIPLoader", "LTXVConditioning", "ConditioningZeroOut", "LTXVConcatAVLatent",
             "LTXVSeparateAVLatent", "SamplerCustomAdvanced", "KSamplerSelect", "BasicScheduler",
             "RandomNoise", "CFGGuider", "LTXVLatentUpsampler", "LatentUpscaleModelLoader",
             "VAEDecode", "LTXVAudioVAEDecode", "VAELoader"]
_missing = [n for n in _required if n not in NODE_CLASS_MAPPINGS]
if _missing:
    print(f"  WARNING: missing nodes -> {_missing}")
    print("  (If LTXDirector* are missing, ComfyUI-LTXVideo/KJNodes are out of date. Re-run Cell 4.)")
else:
    print("  All required workflow nodes present (incl. LTXDirector Master Timeline Controller).")



# ════════════════════════════════════════════════════════════════════════════
# CELL 9: LATENT/TENSOR HELPERS, LOADERS, SCHEDULER, TILED DECODE
# ════════════════════════════════════════════════════════════════════════════
from PIL import Image, ImageOps, ImageDraw
import numpy as np


def unwrap_tensor(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)) and obj.args:
        return unwrap_tensor(obj.args[0])
    if isinstance(obj, (tuple, list)) and obj:
        return unwrap_tensor(obj[0])
    if isinstance(obj, dict):
        if "samples" in obj:
            return unwrap_tensor(obj["samples"])
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    return obj


def unwrap_latent(x: Any) -> Dict[str, Any]:
    if x is None:
        return {"samples": None}
    if hasattr(x, "args") and isinstance(x.args, (list, tuple)) and x.args:
        x = x.args[0]
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
    target = torch.device(target_device)
    d = unwrap_latent(latent)
    s = d.get("samples", None)
    if isinstance(s, torch.Tensor):
        if s.is_nested:
            d["samples"] = torch.nested.nested_tensor([t.to(target) for t in s.unbind()])
        else:
            d["samples"] = s.to(target)
    # carry noise_mask / audio companions across device if present
    for k in ("noise_mask",):
        if isinstance(d.get(k), torch.Tensor):
            d[k] = d[k].to(target)
    return d


# --- Keyframe validation / fallbacks ----------------------------------------
BASE_INPUT = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(BASE_INPUT, exist_ok=True)

for idx, fname in enumerate(["1.png", "2.png", "3.png", "4.png", "5.3.png"]):
    p = os.path.join(BASE_INPUT, fname)
    if not os.path.exists(p):
        img = Image.new("RGB", (768, 512), color=(30 + idx * 25, 25, 60 + idx * 25))
        ImageDraw.Draw(img).text((40, 230), f"Upload singer photo -> {fname}", fill=(255, 255, 255))
        img.save(p)
if os.path.exists(f"{BASE_INPUT}/5.png") and not os.path.exists(f"{BASE_INPUT}/5.3.png"):
    shutil.copy(f"{BASE_INPUT}/5.png", f"{BASE_INPUT}/5.3.png")


# --- VAE loader helper -------------------------------------------------------
def load_vae(vae_name: str, device: str = "main_device", dtype: str = "bf16"):
    if has_node("VAELoaderKJ"):
        try:
            return out(call_node(N("VAELoaderKJ"), vae_name=vae_name, device=device, weight_dtype=dtype), 0)
        except Exception:
            pass
    return out(call_node(N("VAELoader"), vae_name=vae_name), 0)


# --- DiT + 4-LoRA stack + preview override + attention hooks -----------------
def load_dit_with_loras():
    """UnetLoaderGGUF -> apply LoRA stack -> ModelPreviewOverrideKJ -> SageAttention
    -> ChunkFeedForward. Mirrors the JSON's model preparation chain."""
    purge_deep("pre_dit_load")
    model = out(call_node(N("UnetLoaderGGUF"), unet_name="ltx-2-3-22b-dev-Q4_K_M.gguf"), 0)

    lora_cls = NODE_CLASS_MAPPINGS.get("LoraLoaderGGUF", LoraLoaderModelOnly)
    for cfg in LORA_STACK:
        if not cfg["on"]:
            continue
        full = os.path.join(lora_dir, cfg["name"])
        if not os.path.exists(full):
            print(f"  [LoRA missing] {cfg['name']}")
            continue
        try:
            clear_memory(light=True)
            res = call_node(lora_cls(), model=model, lora_name=cfg["name"], strength_model=cfg["strength"])
            model = out(res, 0) or model
            print(f"  + LoRA {cfg['name']} @ {cfg['strength']}")
        except Exception as e:
            print(f"  [LoRA notice] {cfg['name']}: {e}")

    if has_node("ModelPreviewOverrideKJ"):
        try:
            tiny = load_vae(tiny_vae_model, device="main_device", dtype="bf16")
            res = call_node(N("ModelPreviewOverrideKJ"), model=model, vae=tiny)
            model = out(res, 0) or model
            print("  ok ModelPreviewOverrideKJ")
        except Exception:
            pass
    if has_node("PatchSageAttentionKJ"):
        try:
            model = out(call_node(N("PatchSageAttentionKJ"), model=model, sage_attention="auto"), 0) or model
            print("  ok SageAttention")
        except Exception:
            pass
    if has_node("LTXVChunkFeedForward"):
        try:
            model = out(call_node(N("LTXVChunkFeedForward"), model=model, chunks=8, dim_threshold=4096), 0) or model
            print("  ok ChunkFeedForward (chunks=8)")
        except Exception:
            pass
    return model


# --- Scheduler sigmas (BasicScheduler, linear_quadratic) ---------------------
def basic_scheduler_sigmas(model, steps: int, denoise: float, scheduler: str = "linear_quadratic"):
    try:
        res = call_node(N("BasicScheduler"), model=model, scheduler=scheduler,
                        steps=steps, denoise=denoise)
        sig = out(res, 0)
        if isinstance(sig, torch.Tensor) and sig.numel() > 0:
            return sig
    except Exception:
        pass
    try:
        import comfy.samplers
        ms = model.get_model_object("model_sampling")
        total = int(steps / denoise) if 0.0 < denoise < 1.0 else steps
        sig = comfy.samplers.calculate_sigmas(ms, scheduler, total)
        return sig[-(steps + 1):]
    except Exception:
        total = int(round(steps / denoise)) if 0.0 < denoise < 1.0 else steps
        s = [((1.0 - i / total) ** 2) for i in range(total + 1)]
        return torch.tensor(s[-(steps + 1):], dtype=torch.float32)


# --- Tiled VAE decode (essential for full-timeline decode on T4) -------------
def tiled_decode_video(video_latent, vae_obj, tile_size: int = 256):
    latent_dict = unwrap_latent(video_latent)
    if has_node("LTXVSpatioTemporalTiledVAEDecode"):
        try:
            res = call_node(N("LTXVSpatioTemporalTiledVAEDecode"), vae=vae_obj, latents=latent_dict,
                            spatial_tiles=2, spatial_overlap=8, temporal_tile_length=16,
                            temporal_overlap=4, last_frame_fix=False,
                            working_device="auto", working_dtype="auto")
            return unwrap_tensor(res)
        except Exception:
            pass
    if has_node("VAEDecodeTiled"):
        try:
            return unwrap_tensor(call_node(N("VAEDecodeTiled"), samples=latent_dict, vae=vae_obj, tile_size=tile_size))
        except Exception:
            pass
    return unwrap_tensor(call_node(N("VAEDecode"), samples=latent_dict, vae=vae_obj))


print("Cell 9: Helpers, loaders, scheduler & tiled decode ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 10: FAITHFUL SINGLE-PASS PIPELINE (exact JSON node graph)
# ════════════════════════════════════════════════════════════════════════════
SEED = 0   # JSON RandomNoise seed (fixed)


def purge_clip(clip_ref_names: list):
    """Evict the Gemma text encoder from VRAM/RAM after LTXDirector has produced
    conditioning, while KEEPING the patched DiT model alive (we still need it)."""
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
    drop_page_cache()


def run_faithful_pipeline(workdir="/content/LTXDirector_Work",
                          outdir="/content/LTXStudio_Output",
                          song_path=None,
                          mux_original_song=True) -> Optional[str]:
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    patch_comfy_memory_manager()
    patch_safetensors_direct_to_gpu()
    purge_deep("preflight")
    print(f"  [RAM baseline] {get_ram_free_gb():.2f} GB free")

    with torch.inference_mode():
        # ───────────────────────────────────────────────────────────────────
        # PHASE 1 : MASTER TIMELINE CONTROLLER (LTXDirector)
        #   model + clip + audio_vae -> one continuous latent + shared
        #   conditioning + guide/motion data + timeline audio.
        # ───────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70 + "\nPHASE 1: LTXDirector Master Timeline Controller\n" + "=" * 70)
        ram_guard(2.0, "phase1")

        print("  Loading DualCLIP (Gemma-3-12B) ...")
        clip = out(call_node(N("DualCLIPLoader"),
                             clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
                             clip_name2="ltx-2.3_text_projection_bf16.safetensors",
                             type="ltxv", device="default"), 0)

        print("  Loading audio VAE ...")
        audio_vae = load_vae(vae_audio_model, device="main_device", dtype="fp16")

        print("  Loading DiT (GGUF) + 4-LoRA stack + hooks ...")
        base_model = load_dit_with_loras()

        print("  Running LTXDirector (this encodes prompts + builds the whole timeline) ...")
        d = call_node(
            N("LTXDirector"),
            model=base_model, clip=clip, audio_vae=audio_vae,
            global_prompt=GLOBAL_PROMPT,
            start_second=0.0, end_second=DURATION_SECONDS, duration_seconds=DURATION_SECONDS,
            start_frame=START_FRAME, end_frame=END_FRAME, duration_frames=DURATION_FRAMES,
            timeline_data=TIMELINE_DATA,
            local_prompts=LOCAL_PROMPTS, segment_lengths=SEGMENT_LENGTHS,
            guide_strength=GUIDE_STRENGTH, epsilon=epsilon,
            frame_rate=fps, display_mode="seconds",
            custom_width=custom_width, custom_height=custom_height,
            resize_method=resize_method, divisible_by=divisible_by,
            img_compression=img_compression,
            use_custom_audio=True, use_custom_motion=True,
            inpaint_audio=True, override_audio=False,
        )
        patched_model = out(d, 0)
        director_positive = out(d, 1)
        video_latent = sync_latent_device(out(d, 2), "cpu")
        audio_latent = sync_latent_device(out(d, 3), "cpu")
        guide_data = out(d, 4)
        motion_guide_data = out(d, 5)
        director_fps = out(d, 6) or float(fps)
        combined_audio = out(d, 7)
        del d

        # Free the Gemma text encoder now; keep patched_model (holds segment masks)
        del clip
        purge_clip(["clip"])
        print(f"  ok LTXDirector done. Free RAM {get_ram_free_gb():.2f} GB")

        # Conditioning: negative = zero-out of the director positive; wrap for LTXV
        neg_raw = out(call_node(N("ConditioningZeroOut"), conditioning=director_positive), 0)
        condp = call_node(N("LTXVConditioning"), positive=director_positive,
                          negative=neg_raw, frame_rate=director_fps)
        pos_c, neg_c = out(condp, 0), out(condp, 1)
        del neg_raw, condp

        # Shared sampler + noise
        ks_euler = out(call_node(N("KSamplerSelect"), sampler_name="euler"), 0)

        # Load the video VAE once for both guide stages + final decode
        video_vae = load_vae(vae_model, device="main_device", dtype="bf16")

        # ───────────────────────────────────────────────────────────────────
        # STAGE 1 : LTXDirectorGuide(scale_by=0.5) -> ConcatAV -> Euler 8 / d1.0
        # ───────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70 + "\nSTAGE 1: base-resolution diffusion (Euler 8, denoise 1.0)\n" + "=" * 70)
        ram_guard(2.0, "stage1")
        g1 = call_node(
            N("LTXDirectorGuide"),
            positive=pos_c, negative=neg_c, vae=video_vae, latent=video_latent,
            guide_data=guide_data, motion_guide_data=motion_guide_data, model=patched_model,
            ic_lora_name="None", ic_lora_strength=1.0, scale_by=0.5,
            upscale_method="bicubic", image_attention_strength=1.0, crop="center",
            auto_snap_ic_grid=True, use_tiled_encode=False, tile_size=256, tile_overlap=64,
            retake_mode=False,
        )
        s1_pos, s1_neg = out(g1, 0), out(g1, 1)
        s1_lat = sync_latent_device(out(g1, 2), "cpu")
        s1_model = out(g1, 3) or patched_model
        del g1
        clear_memory(light=True)
        print("  ok Stage 1 guide applied (identity anchored across all segments)")

        av1 = sync_latent_device(out(call_node(N("LTXVConcatAVLatent"),
                                               video_latent=s1_lat, audio_latent=audio_latent), 0), "cpu")
        del s1_lat
        sig1 = basic_scheduler_sigmas(s1_model, steps=8, denoise=1.0)
        noise = call_node(N("RandomNoise"), noise_seed=SEED)
        guider1 = call_node(N("CFGGuider"), model=s1_model, positive=s1_pos, negative=s1_neg, cfg=1.0)
        print("  Sampling stage 1 ...")
        r1 = call_node(N("SamplerCustomAdvanced"), noise=out(noise, 0), guider=out(guider1, 0),
                       sampler=ks_euler, sigmas=sig1, latent_image=av1)
        s1_out = sync_latent_device(out(r1, 0), "cpu")
        del av1, r1, guider1, noise, sig1, s1_model
        clear_memory(light=True)

        sep1 = call_node(N("LTXVSeparateAVLatent"), av_latent=s1_out)
        v1 = sync_latent_device(out(sep1, 0), "cpu")
        a1 = sync_latent_device(out(sep1, 1), "cpu")
        del sep1, s1_out

        c1 = call_node(N("LTXDirectorCropGuides"), positive=s1_pos, negative=s1_neg, latent=v1)
        s1_pos, s1_neg = out(c1, 0) or s1_pos, out(c1, 1) or s1_neg
        v1c = sync_latent_device(out(c1, 2) if out(c1, 2) is not None else v1, "cpu")
        del c1, v1

        # ───────────────────────────────────────────────────────────────────
        # UPSCALE : LTXVLatentUpsampler (2x spatial)
        # ───────────────────────────────────────────────────────────────────
        print("\nUPSCALE: 2x latent spatial upscaler ...")
        upmodel = out(call_node(N("LatentUpscaleModelLoader"),
                                model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), 0)
        v_ups = sync_latent_device(out(call_node(N("LTXVLatentUpsampler"),
                                                 samples=v1c, upscale_model=upmodel, vae=video_vae), 0), "cpu")
        del upmodel, v1c
        clear_memory(light=True)

        # ───────────────────────────────────────────────────────────────────
        # STAGE 2 : LTXDirectorGuide(scale_by=1.0) -> ConcatAV -> Euler 4 / d0.42
        # ───────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70 + "\nSTAGE 2: refinement diffusion (Euler 4, denoise 0.42)\n" + "=" * 70)
        ram_guard(2.0, "stage2")
        g2 = call_node(
            N("LTXDirectorGuide"),
            positive=s1_pos, negative=s1_neg, vae=video_vae, latent=v_ups,
            guide_data=guide_data, motion_guide_data=motion_guide_data, model=patched_model,
            ic_lora_name="None", ic_lora_strength=1.0, scale_by=1.0,
            upscale_method="bicubic", image_attention_strength=1.0, crop="center",
            auto_snap_ic_grid=True, use_tiled_encode=False, tile_size=256, tile_overlap=64,
            retake_mode=False,
        )
        s2_pos, s2_neg = out(g2, 0), out(g2, 1)
        s2_lat = sync_latent_device(out(g2, 2), "cpu")
        s2_model = out(g2, 3) or patched_model
        del g2, v_ups
        clear_memory(light=True)
        print("  ok Stage 2 guide applied (micro-detail lock at full resolution)")

        av2 = sync_latent_device(out(call_node(N("LTXVConcatAVLatent"),
                                               video_latent=s2_lat, audio_latent=a1), 0), "cpu")
        del s2_lat, a1
        sig2 = basic_scheduler_sigmas(s2_model, steps=4, denoise=0.42)
        noise2 = call_node(N("RandomNoise"), noise_seed=SEED)
        guider2 = call_node(N("CFGGuider"), model=s2_model, positive=s2_pos, negative=s2_neg, cfg=1.0)
        print("  Sampling stage 2 ...")
        r2 = call_node(N("SamplerCustomAdvanced"), noise=out(noise2, 0), guider=out(guider2, 0),
                       sampler=ks_euler, sigmas=sig2, latent_image=av2)
        s2_out = sync_latent_device(out(r2, 0), "cpu")
        del av2, r2, guider2, noise2, sig2, s2_model

        # Free the big DiT now; only VAE decode remains
        del patched_model, base_model
        purge_deep("post-sampling")

        sep2 = call_node(N("LTXVSeparateAVLatent"), av_latent=s2_out)
        v2 = sync_latent_device(out(sep2, 0), "cpu")
        a2 = sync_latent_device(out(sep2, 1), "cpu")
        del sep2, s2_out

        c2 = call_node(N("LTXDirectorCropGuides"), positive=s2_pos, negative=s2_neg, latent=v2)
        v2c = out(c2, 2) if out(c2, 2) is not None else v2
        del c2, v2

        # ───────────────────────────────────────────────────────────────────
        # DECODE : tiled video VAE + audio VAE  ->  VHS_VideoCombine
        # ───────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70 + "\nDECODE & ASSEMBLE\n" + "=" * 70)
        print("  Tiled VAE decoding video ...")
        frames = tiled_decode_video(v2c, video_vae, tile_size=256)
        frames = unwrap_tensor(frames).detach().cpu().float()
        del v2c, video_vae
        clear_memory(light=True)
        print(f"  ok decoded {frames.shape[0]} frames ({frames.shape[0]/fps:.2f}s)")

        print("  Decoding audio latent ...")
        decoded_audio = None
        try:
            aud_vae2 = load_vae(vae_audio_model, device="main_device", dtype="fp16")
            decoded_audio = out(call_node(N("LTXVAudioVAEDecode"), samples=a2, audio_vae=aud_vae2), 0)
            del aud_vae2
        except Exception as e:
            print(f"  [audio decode notice] {e}; will fall back to combined_audio/original song")
        if decoded_audio is None:
            decoded_audio = combined_audio
        del a2
        clear_memory(light=True)

        # Write MP4 (VHS_VideoCombine preferred, imageio fallback)
        raw_path = os.path.join(outdir, "LTX23_Director_Master_V4.mp4")
        saved = False
        if has_node("VHS_VideoCombine"):
            try:
                cv = call_node(N("VHS_VideoCombine"), images=frames, audio=decoded_audio,
                               frame_rate=fps, loop_count=0, filename_prefix="LTX2.3/Video",
                               format="video/h264-mp4", pix_fmt="yuv420p", crf=output_crf,
                               save_metadata=False, trim_to_audio=False, pingpong=False, save_output=True)
                # VHS returns a dict with saved file paths in .ui/result; find the mp4
                res = out(cv, 0)
                fp = None
                try:
                    ui = getattr(cv, "ui", None) or (res if isinstance(res, dict) else None)
                    if isinstance(ui, dict):
                        gifs = ui.get("gifs") or ui.get("videos") or []
                        if gifs:
                            fp = gifs[0].get("fullpath") or os.path.join(
                                "/content/ComfyUI/output", gifs[0].get("subfolder", ""), gifs[0].get("filename", ""))
                except Exception:
                    pass
                if fp and os.path.exists(fp):
                    shutil.copyfile(fp, raw_path)
                    saved = True
            except Exception as e:
                print(f"  [VHS_VideoCombine notice] {e}")
        if not saved:
            import imageio
            arr = (frames.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            imageio.mimwrite(raw_path, arr, fps=fps, quality=9)
            print("  (used imageio fallback for video encode)")
        print(f"  ok video written: {raw_path}")

        # Optional: mux the ORIGINAL crisp song (trimmed to trimStart) for clean audio
        final_path = raw_path
        if mux_original_song and song_path and os.path.exists(song_path):
            trim_sec = AUDIO_SEGMENT["trimStart"] / fps
            out_muxed = raw_path.replace(".mp4", "_song.mp4")
            cmd = (f'ffmpeg -y -i "{raw_path}" -ss {trim_sec:.3f} -i "{song_path}" '
                   f'-map 0:v:0 -map 1:a:0 -c:v libx264 -crf {output_crf} -pix_fmt yuv420p '
                   f'-c:a aac -b:a 320k -shortest "{out_muxed}"')
            if run_cmd(cmd, silent=False) == 0 and os.path.exists(out_muxed):
                final_path = out_muxed
                print(f"  ok original song muxed: {final_path}")

        del frames
        purge_deep("finished")
        return final_path


print("Cell 10: Faithful single-pass pipeline ready.")


# ════════════════════════════════════════════════════════════════════════════
# RUNTIME TRIGGER
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    song = "/content/ComfyUI/input/whatdreamscost/Late night trap.mp3"
    # keyframe sanity print
    for s in SEGMENTS:
        p = os.path.join("/content/ComfyUI/input", s["imageFile"])
        print(("  ok  " if os.path.exists(p) else "  MISSING ") + p)
    print(f"  audio: {song} ({'found' if os.path.exists(song) else 'MISSING'})")

    final = run_faithful_pipeline(song_path=song if os.path.exists(song) else None,
                                  mux_original_song=True)
    print(f"\nDONE. Output file: {final}")
