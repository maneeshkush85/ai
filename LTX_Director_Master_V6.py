# -*- coding: utf-8 -*-
"""
LTX_Director_Master_V6.py  ·  LTX-2.3 / LTX-2.5 selectable Director pipeline

🇮🇳 यह V5 (LTX-2.3) का ही code है, पर अब इसमें एक स्विच जुड़ गया है जिससे आप
LTX-2.3 या LTX-2.5 (LTX-5) models चुन सकते हैं। नीचे CELL 0 में MODEL_FAMILY बदलें।

⚠️  ज़रूरी सच्चाई (पढ़ें):
  • LTX-2.5 का transformer GGUF (Q4) और Gemma-4 text-encoder तो बिना token के मिल
    जाते हैं, पर LTX-2.5 के VAE (video+audio) और spatial upscaler GATED हैं
    (Lightricks/LTX-2.5 पर license accept करके HF_TOKEN देना ज़रूरी है)। बिना token
    के वे download नहीं होंगे।
  • LTX-2.5 नया diffusion decoder इस्तेमाल करता है → उसका latent space 2.3 से अलग
    है, इसलिए 2.3 का VAE 2.5 पर काम नहीं करेगा (garbage आएगा)। इसलिए 2.5 चुनने पर
    असली 2.5 VAE चाहिए ही चाहिए।
  • LTX-2.5 की documented minimum requirement 32GB VRAM + 32GB RAM है। T4 (15GB /
    ~12.2GB) उससे बहुत छोटा है — GGUF Q4 से DiT ~13GB में आ जाता है पर बाकी pipeline
    भारी है, इसलिए T4 पर सिर्फ़ बहुत छोटी/कम-res clip ही fit होगी (auto-cap लगे हैं)।
  • अगर HF_TOKEN न मिला तो script अपने-आप LTX-2.3 पर वापस आ जाती है (ताकि T4 पर
    बिना token के भी चले)।

Set an HF token before running (needed only for LTX-2.5 gated VAE/encoder):
    import os; os.environ["HF_TOKEN"] = "hf_xxx"    # accept Lightricks/LTX-2.5 license first
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 0: MODEL FAMILY SWITCH & MODEL REGISTRY  (choose LTX-2.3 or LTX-2.5)
# 🇮🇳 CELL 0 का काम: सिर्फ़ यहाँ MODEL_FAMILY बदलकर 2.3 या 2.5 चुनें। बाकी पूरा
#   pipeline (memory-safety, phases, decode) वैसा ही रहता है — सिर्फ़ model files
#   बदल जाती हैं। gated 2.5 files के लिए HF_TOKEN चाहिए (ऊपर docstring देखें)।
# ════════════════════════════════════════════════════════════════════════════
import os as _os_early

# @markdown ## 🧬 Model family — "2.5" (LTX-5) या "2.3"
MODEL_FAMILY = _os_early.environ.get("LTX_MODEL_FAMILY", "2.5")   # @param ["2.3", "2.5"]

# 🇮🇳 gated LTX-2.5 VAE/encoder के लिए HuggingFace token (Lightricks/LTX-2.5 पर
# पहले "Agree and access" करें)। env var HF_TOKEN या HUGGINGFACE_TOKEN से भी लिया जाता है।
HF_TOKEN = (_os_early.environ.get("HF_TOKEN")
            or _os_early.environ.get("HUGGINGFACE_TOKEN")
            or _os_early.environ.get("HUGGING_FACE_HUB_TOKEN")
            or "")   # @param {type:"string"}

# ── MODEL REGISTRIES ─────────────────────────────────────────────────────────
# हर entry: (filename, url, gated?)। gated=True मतलब download के लिए HF_TOKEN चाहिए।
# सारे URLs verify किए गए हैं (2025-08)।
_LTX_ROOT = "https://huggingface.co"
MODEL_REGISTRY = {
    "2.3": {
        "label": "LTX-2.3 (22B, T4-friendly, all assets ungated)",
        # DiT (GGUF quant चुनने के लिए Cell 0 env LTX_DIT_GGUF से override कर सकते हैं)
        "dit_gguf":      ("ltx-2-3-22b-dev-Q4_K_M.gguf",
                          f"{_LTX_ROOT}/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf", False),
        # Text encoder: 2.3 uses DualCLIPLoader (gemma + separate projection)
        "encoder_mode":  "dual",
        "text_encoder":  ("gemma_3_12B_it_fp4_mixed.safetensors",
                          f"{_LTX_ROOT}/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", False),
        "text_proj":     ("ltx-2.3_text_projection_bf16.safetensors",
                          f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors", False),
        "clip_type":     "ltxv",
        "video_vae":     ("LTX23_video_vae_bf16.safetensors",
                          f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors", False),
        "audio_vae":     ("LTX23_audio_vae_bf16.safetensors",
                          f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors", False),
        "tiny_vae":      ("taeltx2_3.safetensors",
                          f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors", False),
        "upscaler":      ("ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                          f"{_LTX_ROOT}/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors", False),
        "loras": [
            ("ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
             f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", 0.4, False),
            ("LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
             f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", 0.6, False),
            ("ltx2.3-transition.safetensors",
             f"{_LTX_ROOT}/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors", 0.7, False),
            ("LTX2.3-MVCamera-drclips.safetensors",
             f"{_LTX_ROOT}/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors", 0.9, False),
        ],
    },
    "2.5": {
        "label": "LTX-2.5 / LTX-5 (22B distilled; VAE+upscaler GATED — needs HF_TOKEN)",
        # DiT: distilled Q4_K_M GGUF — ungated community mirror (FenomAI).
        "dit_gguf":      ("LTX-2.5-Distilled-Q4_K_M.gguf",
                          f"{_LTX_ROOT}/FenomAI/LTX-2.5-Distilled-GGUF/resolve/main/LTX-2.5-Distilled-Q4_K_M.gguf", False),
        # Text encoder: 2.5 uses a single CLIPLoader with the projection FUSED in.
        # The correct fused encoder (gemma4-12b-with-proj) is GATED on Lightricks/LTX-2.5.
        "encoder_mode":  "single",
        "text_encoder":  ("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                          f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", True),
        "text_proj":     None,
        "clip_type":     "ltxv",
        # VAEs + upscaler: GATED on Lightricks/LTX-2.5 (no ungated mirror exists).
        "video_vae":     ("ltx-2.5-video-vae-bf16.safetensors",
                          f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors", True),
        "audio_vae":     ("ltx-2.5-audio-vae-bf16.safetensors",
                          f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors", True),
        # tiny preview VAE: 2.3 taesd is fine for preview only (not the real decode).
        "tiny_vae":      ("taeltx2_3.safetensors",
                          f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors", False),
        "upscaler":      ("ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                          f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", True),
        # LoRAs: 2.5 is backward-compatible with 2.3 LoRAs (per Lightricks). We use
        # the two ungated ones from the 2.5 All-In-One workflow at its exact strengths.
        "loras": [
            ("LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
             f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", 0.4, False),
            ("ltx2.3-transition.safetensors",
             f"{_LTX_ROOT}/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors", 0.8, False),
            ("LTX2.3-MVCamera-drclips.safetensors",
             f"{_LTX_ROOT}/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors", 0.9, False),
        ],
    },
}


def _resolve_model_family():
    """🇮🇳 चुना हुआ family तय करता है। अगर 2.5 चुना पर HF_TOKEN नहीं है, तो gated
    VAE/encoder download नहीं होंगे → अपने-आप 2.3 पर safe fallback कर देता है।"""
    fam = str(MODEL_FAMILY).strip()
    if fam not in MODEL_REGISTRY:
        print(f"  ⚠️  MODEL_FAMILY='{fam}' अमान्य → '2.3' इस्तेमाल कर रहे हैं।")
        fam = "2.3"
    if fam == "2.5":
        reg = MODEL_REGISTRY["2.5"]
        needs_token = any(reg[k][-1] for k in ("text_encoder", "video_vae", "audio_vae", "upscaler")
                          if isinstance(reg.get(k), tuple))
        if needs_token and not HF_TOKEN:
            print("  ⚠️  LTX-2.5 चुना गया पर HF_TOKEN नहीं मिला। LTX-2.5 के VAE/encoder GATED हैं")
            print("      (Lightricks/LTX-2.5 पर license accept करके token चाहिए)।")
            print("      → अभी LTX-2.3 पर safe fallback कर रहे हैं ताकि T4 पर बिना token के चले।")
            print("      2.5 चाहिए तो: os.environ['HF_TOKEN']='hf_...' सेट करके दोबारा चलाएँ।")
            fam = "2.3"
    return fam


ACTIVE_FAMILY = _resolve_model_family()
MODELS = MODEL_REGISTRY[ACTIVE_FAMILY]
print(f"🧬 Model family = LTX-{ACTIVE_FAMILY}  ·  {MODELS['label']}")


class TextEncoderOOM(RuntimeError):
    """🇮🇳 सिर्फ़ text-encoder (prompt encode) के समय VRAM खत्म होने पर। यह
    resolution-independent है — इसलिए resolution घटाने वाला OOM-retry ladder इसे
    नहीं पकड़ता (वरना बेकार में 4 बार retry होता है, जैसा LTX-2.5 12B encoder पर हुआ)।"""
    pass


# ⚠️ LTX-2.5 hard wall: Lightricks का अपना कहना है कि LTX-2 / LTX-2.5 का Gemma-4 12B
# text-encoder चलाने के लिए ~24-27GB VRAM चाहिए (16GB पर भी नहीं चलता, FP8 के बाद भी)।
# T4 (15GB) उससे छोटा है — इसलिए 2.5 का encoder T4 पर fit ही नहीं होता। यह चेतावनी
# पहले ही दे देते हैं ताकि पता रहे कि क्यों encode step पर OOM आएगा।
# 🌊 STREAM_ENCODER: LTX-2.5 का 12B encoder VRAM में पूरा नहीं आता। इसे ON करने पर
# encoder को GPU पर पूरा रखने के बजाय layer-by-layer (CPU + disk-mmap-backed) stream
# किया जाता है → peak VRAM बहुत कम (सिर्फ़ active layer)। धीमा पर छोटे GPU पर fit हो
# सकता है। env LTX_STREAM_ENCODER=0/1 से force भी कर सकते हैं। (EXPERIMENTAL — यह तभी
# काम करेगा जब activation memory भी छोटी हो; बहुत छोटे VRAM पर फिर भी OOM हो सकता है।)
STREAM_ENCODER = False
if ACTIVE_FAMILY == "2.5":
    try:
        import torch as _t_early
        _vram_gb_early = (_t_early.cuda.get_device_properties(0).total_memory / 1e9
                          if _t_early.cuda.is_available() else 0.0)
    except Exception:
        _vram_gb_early = 0.0
    # 🇮🇳 streaming अब सिर्फ़ opt-in (default OFF)। कारण: T4 पर host RAM (~11GB) encoder
    # (~12GB) से छोटी है, इसलिए CPU-stream करने पर encoder host RAM में materialize होकर
    # पूरा session crash कर देता है (हमने यह empirically देखा)। इसलिए इसे अपने-आप ON नहीं
    # करते; encode step का preflight वैसे भी सुरक्षित रूप से पहले ही रोक देगा।
    _env_se = _os_early.environ.get("LTX_STREAM_ENCODER")
    if _env_se is not None:
        STREAM_ENCODER = str(_env_se).strip().lower() not in ("0", "false", "no", "")
    if 0 < _vram_gb_early < 24.0:
        print("=" * 70)
        print(f"  ⛔ LTX-2.5 का Gemma-4 12B text-encoder ~24GB+ VRAM माँगता है "
              f"(Lightricks official)। आपके GPU में ~{_vram_gb_early:.1f}GB है।")
        print(f"  यह encoder न इस VRAM में पूरा आता है, न T4 की ~11GB host RAM में — इसलिए")
        print(f"  encode step पर script इसे load करने से पहले ही साफ़ error देकर रुक जाएगी")
        print(f"  (ताकि पिछली बार जैसा FATAL session-crash न हो)।")
        print(f"  ✅ इस GPU पर काम करने के लिए: MODEL_FAMILY='2.3' (T4 पर पूरा चलता है)।")
        print(f"  ✅ LTX-2.5 quality चाहिए तो: L4/A100 (24GB+) runtime इस्तेमाल करें।")
        if STREAM_ENCODER:
            print(f"  🌊 (आपने LTX_STREAM_ENCODER=1 forced किया है — जोखिम पर streaming कोशिश होगी।)")
        print("=" * 70)

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP & MEMORY PROTECTION  (no swap — Colab blocks it)
# 🇮🇳 CELL 1 का काम: Python environment तैयार करना और memory settings लगाना।
#   • CUDA को बताता है कि GPU memory को कैसे बाँटना है (fragmentation कम हो)।
#   • ComfyUI के 'utils' module का conflict रोकने के लिए एक छोटा patch लगाता है।
#   • host RAM कितनी खाली है, यह print करता है।
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

# ── PATH CONSTANTS ───────────────────────────────────────────────────────────
# 🇮🇳 HINDI: सारे रास्ते (paths) यहाँ एक जगह रखे हैं ताकि Colab के अलावा किसी और
# जगह चलाना हो तो सिर्फ़ यहीं बदलना पड़े (हर जगह ढूँढना न पड़े)।
COMFY_ROOT   = os.environ.get("LTX_COMFY_ROOT", "/content/ComfyUI")
CONTENT_ROOT = os.environ.get("LTX_CONTENT_ROOT", "/content")
MODELS_DIR       = os.path.join(COMFY_ROOT, "models")
INPUT_DIR        = os.path.join(COMFY_ROOT, "input")
WHATDREAMS_INPUT = os.path.join(INPUT_DIR, "whatdreamscost")

# ── LIGHTWEIGHT LOGGING ──────────────────────────────────────────────────────
# 🇮🇳 HINDI: एक हल्का logger। LTX_LOG_LEVEL से शोर कम/ज़्यादा कर सकते हैं:
#   • "DEBUG" → सब कुछ (छिपे हुए errors भी दिखते हैं),
#   • "INFO"  → सामान्य (default),
#   • "WARN"  → सिर्फ़ चेतावनियाँ/errors।
# पुराने print() वैसे ही चलते रहेंगे; नया code log() इस्तेमाल करता है।
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
LOG_LEVEL = os.environ.get("LTX_LOG_LEVEL", "INFO").upper()


def log(msg: str, level: str = "INFO"):
    """स्तर (level) के हिसाब से message छापता है।"""
    if _LOG_LEVELS.get(level, 20) >= _LOG_LEVELS.get(LOG_LEVEL, 20):
        prefix = {"DEBUG": "  🔎", "INFO": " ", "WARN": "  ⚠️", "ERROR": "  ❌"}.get(level, " ")
        print(f"{prefix} {msg}")


def _dbg(msg: str):
    """किसी छिपे हुए (silently-caught) error को DEBUG level पर दिखाने के लिए।"""
    log(msg, "DEBUG")


def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    return subprocess.run(cmd, shell=True).returncode


# 🇮🇳 HINDI: Colab free-tier पर swap काम नहीं करता, इसलिए हटा दिया गया।
# RAM बचाने के तरीके अब ये हैं: (1) Gemma text-encoder को encode के बाद तुरंत
# purge करना, (2) VAE decode को छोटे chunks में stream करना, (3) model को VRAM से
# सीधे free करना (host RAM में copy किए बिना), और (4) T4 पर duration/resolution
# auto-cap करना। नीचे सिर्फ़ एक हल्की memory report है।
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"  📊 Memory: Host RAM {vm.available/1e9:.2f} GB free / {vm.total/1e9:.2f} GB "
          f"(no swap on Colab free-tier — RAM safety handled in-pipeline)")
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
# 🇮🇳 CELL 2 का काम: ज़रूरी Python libraries install करना — PyTorch (GPU),
#   diffusers, einops, opencv, ffmpeg, aria2 (तेज़ downloader) आदि। SageAttention
#   optional है (attention तेज़ करता है); न मिले तो भी pipeline चलता रहता है।
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
# 🇮🇳 CELL 3 का काम: असली ComfyUI (GitHub से) download करना और उसे Python के
#   रास्ते (sys.path) में जोड़ना, ताकि हम उसके सारे nodes सीधे code से चला सकें।
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
# 🇮🇳 CELL 4 का काम: वही 6 custom node-packs install करना जो JSON workflow को
#   चाहिए — WhatDreamsCost (Director), KJNodes, GGUF loader, LTXVideo,
#   VideoHelperSuite, और rgthree (Power LoRA Loader)।
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
# 🇮🇳 CELL 5 का काम: सारे AI models download करना —
#   • 22B main model (GGUF Q4), • Gemma-12B text encoder, • video+audio VAE,
#   • 2x upscaler, • 4 LoRAs (0.4/0.6/0.7/0.9 strength), • song (mp3)।
#   एक ही file को कई जगह चाहिए तो symlink बना देता है (जगह बचाने के लिए)।
# ════════════════════════════════════════════════════════════════════════════
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ── DiT QUANT चुनाव ──────────────────────────────────────────────────────────
# 🇮🇳 HINDI: 22B model किस quant में download/load हो, यह यहाँ चुनते हैं।
#   • छोटा quant (Q4_K_M) → कम memory, थोड़ी कम quality — T4 के लिए default।
#   • बड़ा quant (Q6_K / Q8_0) → ज़्यादा memory पर बेहतर quality — L4/A100 के लिए।
# अब quant registry (Cell 0) से आता है। env LTX_DIT_GGUF से filename override भी कर सकते हैं।
DIT_GGUF_NAME = os.environ.get("LTX_DIT_GGUF", MODELS["dit_gguf"][0])
_DIT_GGUF_URL = (MODELS["dit_gguf"][1] if os.environ.get("LTX_DIT_GGUF") is None
                 else MODELS["dit_gguf"][1].replace(MODELS["dit_gguf"][0], DIT_GGUF_NAME))
_DIT_GGUF_GATED = bool(MODELS["dit_gguf"][2])


def download_file(url: str, dest_dir: str, filename: Optional[str] = None,
                  gated: bool = False) -> Optional[str]:
    """🇮🇳 aria2c से file download करता है। gated=True (LTX-2.5 के VAE/encoder) पर
    HF_TOKEN को Authorization header में भेजता है (Lightricks license accept ज़रूरी)।"""
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
        dest = os.path.join(dest_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  [FOUND] {filename}")
            return filename
        if gated and not HF_TOKEN:
            print(f"\n  🔒 SKIP (gated, no HF_TOKEN): {filename}")
            print(f"      Accept the license at the model page, then set HF_TOKEN and re-run.")
            return None
        cmd = ['aria2c', '--console-log-level=error', '-c', '-x', '16',
               '-s', '16', '-k', '1M', '-d', dest_dir, '-o', filename]
        if HF_TOKEN and "huggingface.co" in url:
            # token gated + private दोनों के लिए safe है; ungated पर भी बेअसर।
            cmd += [f'--header=Authorization: Bearer {HF_TOKEN}']
        cmd.append(url)
        tag = " 🔒" if gated else ""
        print(f"  ↓ Downloading{tag} {filename}...", end=' ', flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print("Done!")
            return filename
        print("FAILED")
        _err = (res.stderr or res.stdout or "").strip().splitlines()
        if _err:
            print(f"      aria2c: {_err[-1][:160]}")
        if gated:
            print(f"      (gated file — token invalid या license accept नहीं किया?)")
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


print(f"📦 Downloading LTX-{ACTIVE_FAMILY} Core Models (registry-driven)...")

TE_DIR    = "/content/ComfyUI/models/text_encoders"
CLIP_DIR  = "/content/ComfyUI/models/clip"
VAE_DIR   = "/content/ComfyUI/models/vae"
UP_DIR    = "/content/ComfyUI/models/latent_upscale_models"
lora_dir  = "/content/ComfyUI/models/loras"

_download_failures = []   # 🇮🇳 gated/failed downloads यहाँ जमा होते हैं (नीचे report)।


def _dl(entry, dest_dir, filename=None):
    """registry entry (name,url,...,gated) → download_file, फ़ेल हुआ तो track करें।"""
    name = filename or entry[0]
    url = entry[1]
    gated = bool(entry[-1])
    ok = download_file(url, dest_dir, filename=name, gated=gated)
    if ok is None:
        _download_failures.append((name, gated))
    return ok


# 🇮🇳 22B DiT model (quant registry/env से)।
download_file(_DIT_GGUF_URL, os.path.join(MODELS_DIR, "unet"),
              filename=DIT_GGUF_NAME, gated=_DIT_GGUF_GATED) or _download_failures.append((DIT_GGUF_NAME, _DIT_GGUF_GATED))
link_file_safe(os.path.join(MODELS_DIR, "unet", DIT_GGUF_NAME),
               os.path.join(MODELS_DIR, "diffusion_models", DIT_GGUF_NAME))

# Text encoder (+ optional separate projection for the 2.3 dual-clip path).
_te = MODELS["text_encoder"]
_dl(_te, TE_DIR)
link_file_safe(os.path.join(TE_DIR, _te[0]), os.path.join(CLIP_DIR, _te[0]))
if MODELS.get("text_proj"):
    _tp = MODELS["text_proj"]
    _dl(_tp, TE_DIR)
    link_file_safe(os.path.join(TE_DIR, _tp[0]), os.path.join(CLIP_DIR, _tp[0]))

# VAEs (video + audio) + tiny preview VAE.
_dl(MODELS["video_vae"], VAE_DIR)
_dl(MODELS["audio_vae"], VAE_DIR)
_dl(MODELS["tiny_vae"], VAE_DIR)
link_file_safe(os.path.join(VAE_DIR, MODELS["tiny_vae"][0]),
               os.path.join("/content/ComfyUI/models/vae_approx", MODELS["tiny_vae"][0]))

# Spatial latent upscaler (Stage-2 2x).
_up = MODELS["upscaler"]
_dl(_up, UP_DIR)
link_file_safe(os.path.join(UP_DIR, _up[0]),
               os.path.join("/content/ComfyUI/models/upscale_models", _up[0]))

print(f"📦 Downloading Director LoRA stack ({len(MODELS['loras'])} LoRA[s])...")
for _lora in MODELS["loras"]:
    # LoRA entry = (name, url, strength, gated)
    _dl((_lora[0], _lora[1], _lora[3]), lora_dir)

audio_dest_dir = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(audio_dest_dir, exist_ok=True)
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file("https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3",
                  audio_dest_dir, filename="Late night trap.mp3")

# ── Download failure report ──────────────────────────────────────────────────
# 🇮🇳 अगर कोई ज़रूरी (खासकर gated 2.5) file नहीं मिली तो साफ़ बताएँ और रोकें, ताकि
# आधे-अधूरे models के साथ चलकर बाद में crash/garbage न हो।
_gated_fail = [n for (n, g) in _download_failures if g]
_other_fail = [n for (n, g) in _download_failures if not g]
if _gated_fail:
    print("\n" + "=" * 70)
    print(f"  🔒 {len(_gated_fail)} GATED file(s) download नहीं हुईं (LTX-{ACTIVE_FAMILY}):")
    for n in _gated_fail:
        print(f"       • {n}")
    print("  👉 इन्हें पाने के लिए: https://huggingface.co/Lightricks/LTX-2.5 पर जाकर")
    print("     'Agree and access' करें, फिर token सेट करें:")
    print("        import os; os.environ['HF_TOKEN'] = 'hf_xxx'")
    print("     और Cell 5 दोबारा चलाएँ। (LTX-2.5 के VAE/encoder इनके बिना नहीं चलेंगे।)")
    print("=" * 70)
    if ACTIVE_FAMILY == "2.5":
        raise RuntimeError(
            "LTX-2.5 gated assets missing (need HF_TOKEN + accepted Lightricks/LTX-2.5 "
            "license). Set HF_TOKEN and re-run, or set MODEL_FAMILY='2.3'.")
if _other_fail:
    print(f"  ⚠️  {len(_other_fail)} file(s) download नहीं हुईं: {', '.join(_other_fail)} — "
          f"network जाँचें और दोबारा चलाएँ।")

print(f"✅ Cell 5: LTX-{ACTIVE_FAMILY} models, LoRAs and audio validated.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 6: MASTER TIMELINE "NOTES"  (transcribed 1:1 from LTXDirector node id 131)
# 🇮🇳 CELL 6 सबसे ज़रूरी SETTINGS वाला हिस्सा है:
#   • GLOBAL_PROMPT + NEGATIVE_PROMPT (JSON से हूबहू कॉपी किए हुए)।
#   • Settings panel: resolution, fps, render_seconds, 4 LoRA strengths,
#     2-stage sampler के steps/denoise, audio sync, और T4 memory settings।
#   • नीचे derived values बनती हैं: कुल frames (8k+1 rule), 5 keyframes की
#     timings, एक continuous audio track, और render resolution।
#   👉 आमतौर पर आपको सिर्फ़ यहीं की values बदलनी होती हैं।
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
# @markdown • False (FAITHFUL, needs L4/A100 ~24GB): base = custom//2 (640x360) → 2x → 1280x720 — best face/eye/lip detail.
# @markdown • True (T4-safe): base = generation//2 (416x240) → 2x → 832x480 — much faster/lighter, lower detail.
# @markdown (auto_safe_on_t4 below will auto-force True if your GPU is a small T4.)
two_stage_base_render = True  # @param {type:"boolean"}

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
# @markdown ### 🎬 Diffusion mode (सिर्फ़ FAITHFUL single continuous pass — JSON जैसा)
# ─────────────────────────────────────────────────────────────────────────────
# 🇮🇳 HINDI: यह पूरी video timeline को एक ही बार (single continuous pass) में
# generate करता है — बिल्कुल ComfyUI JSON की तरह। इसी से 3 चीज़ें सही रहती हैं:
#   1) Character हर scene में एक जैसा (consistent identity),
#   2) Scenes आपस में जुड़े रहते हैं (कोई seam/जोड़ नहीं),
#   3) आवाज़ (voice) video के साथ sync रहती है।
# पुराना "batch_scene_mode" (timeline को टुकड़ों में काटना) पूरी तरह हटा दिया गया
# है, क्योंकि वही ऊपर की तीनों problems पैदा करता था।
# ─────────────────────────────────────────────────────────────────────────────
# @markdown `auto_safe_on_t4` — T4 (छोटा GPU, <~20GB) पर faithful single pass को
# @markdown अपने-आप fit कराने के लिए settings adjust करता है (model VRAM में रखता है,
# @markdown Stage-1 base छोटा रखता है, duration/resolution auto-cap करता है).
auto_safe_on_t4           = True   # @param {type:"boolean"}
# @markdown `t4_singlepass_max_seconds` — 22B model T4 की ज़्यादातर memory भर देता
# @markdown है, इसलिए लंबी timeline के लिए जगह कम बचती है। T4 पर single pass को इतने
# @markdown seconds तक auto-cap किया जाता है ताकि यह पूरा FINISH हो (आपको छोटा पर पूरा
# @markdown continuous + सही clip मिलेगा)। पूरी 31.5s चाहिए तो L4/A100 GPU लें।
t4_singlepass_max_seconds = 10.0   # @param {type:"slider", min:3.0, max:31.5, step:0.5}
# @markdown `t4_singlepass_max_height` — Stage-2 refine पूरी generation resolution
# @markdown पर होता है; उसकी activations को VRAM में ~13GB model के साथ fit होना पड़ता
# @markdown है। T4 पर 480p thrashing/crash करता है, इसलिए height को यहाँ तक cap किया
# @markdown जाता है (width अपने-आप aspect ratio के हिसाब से घटती है). जगह हो तो बढ़ाएँ.
t4_singlepass_max_height  = 384    # @param {type:"slider", min:192, max:480, step:32}

# @markdown ## 💾 Output & Run
output_crf         = 8     # @param {type:"slider", min:0, max:30, step:1}
base_seed          = 0     # @param {type:"integer"}
resume_checkpoints = True  # @param {type:"boolean"}

# ── VRAM auto-detect: protect small T4s from the faithful single-pass full-res OOM ──
try:
    _gpu_total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0.0
    _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
except Exception:
    _gpu_total_gb, _gpu_name = 0.0, "unknown"

FULL_QUALITY_HW = _gpu_total_gb >= 20.0   # L4 (24GB) / A100 (40GB) etc.
print(f"  🖥️  GPU: {_gpu_name} ({_gpu_total_gb:.1f} GB VRAM) → "
      f"{'FULL-QUALITY capable' if FULL_QUALITY_HW else 'small GPU (T4-class)'}")

# ── LTX-2.5 T4 tightening ────────────────────────────────────────────────────
# 🇮🇳 HINDI: LTX-2.5 का pipeline (नया diffusion decoder + joint AV) 2.3 से भारी है।
# इसलिए T4 (छोटा GPU) पर 2.5 चुनने पर हम duration/height के defaults और कस देते हैं
# ताकि single pass बिना crash पूरा हो। (आपने खुद values बदली हों तो उन्हें नहीं छेड़ते।)
if ACTIVE_FAMILY == "2.5" and not FULL_QUALITY_HW:
    if abs(float(t4_singlepass_max_seconds) - 10.0) < 1e-6:
        t4_singlepass_max_seconds = 6.0
    if int(t4_singlepass_max_height) == 384:
        t4_singlepass_max_height = 320
    print(f"  🧬 LTX-2.5 on T4: tighter caps → ≤{t4_singlepass_max_seconds}s, "
          f"height ≤{t4_singlepass_max_height}px (2.5 pipeline is heavier than 2.3).")
    print("     फिर भी OOM आए तो render_seconds/height और घटाएँ, या L4/A100 इस्तेमाल करें।")

# ── T4 STRATEGY (character/scene/voice सही रखने की रणनीति) ───────────────────
# 🇮🇳 HINDI: ComfyUI JSON में video अच्छी इसलिए बनती है क्योंकि वह पूरी audio+video
# timeline को एक ही continuous pass में diffuse करती है। इसलिए हम भी हमेशा वही
# single continuous pass चलाते हैं (कोई time-slicing/chunking नहीं)।
#
# ⚠️ ज़रूरी MEMORY बात (एक crash से सीखी): free T4 पर VRAM (~15.6 GB) असल में host
# RAM (~13.6 GB) से बड़ी है। इसलिए 22B model को VRAM में रखना चाहिए (normalvram)।
# "novram" model को HOST RAM से stream करता है — जो छोटी pool है — और sampling के
# दौरान वह भर जाती है → FATAL "session crashed" (host-RAM OOM पूरा kernel मार देता
# है, कोई catchable error नहीं)। normalvram में अगर memory कम पड़े तो सिर्फ़ एक
# *catchable* CUDA OOM आता है, crash नहीं।
#
# चूँकि 22B model T4 की ज़्यादातर memory भर देता है, लंबी timeline की activations के
# लिए जगह कम बचती है — इसलिए T4 पर हम duration और resolution auto-cap करते हैं ताकि
# single pass सच में fit हो और पूरा FINISH हो जाए।
if not FULL_QUALITY_HW:
    # ── छोटा GPU (T4-class) ──
    if auto_safe_on_t4:
        two_stage_base_render = True      # Stage-1 base छोटा (gen//2 → 2x) रखें → T4 में fit
        # Model को VRAM में रखें (यहाँ host RAM से बड़ी pool) → FATAL host-RAM crash से बचाव।
        if str(VRAM_MODE).lower() in ("auto", "novram"):
            VRAM_MODE = "normalvram"
        # (1) Duration cap: continuous pass 22B model के साथ fit हो सके।
        _cap = float(t4_singlepass_max_seconds)
        if render_seconds > _cap:
            print(f"  ✂️  render_seconds {render_seconds}s → {_cap}s cap किया (T4 पर 22B model के")
            print(f"      साथ लंबी timeline fit नहीं होती)। आपको पूरा continuous + सही clip मिलेगा —")
            print(f"      बस छोटा। लंबा चाहिए तो t4_singlepass_max_seconds बढ़ाएँ / L4/A100 इस्तेमाल करें।")
            render_seconds = _cap
        # (2) Resolution cap: Stage-2 activations VRAM में model के साथ fit हों → thrashing रुके।
        _hcap = int(t4_singlepass_max_height)
        if int(generation_height) > _hcap:
            _db = max(1, int(divisible_by))
            _ratio = _hcap / float(int(generation_height))
            _new_h = max(_db, int(round(_hcap / _db)) * _db)
            _new_w = max(_db, int(round(int(generation_width) * _ratio / _db)) * _db)
            print(f"  ✂️  Resolution {generation_width}x{generation_height} → {_new_w}x{_new_h} cap किया "
                  f"(Stage-2 को VRAM में 22B model के साथ जगह मिले, host-RAM thrashing रुके)।")
            generation_width, generation_height = _new_w, _new_h
        print(f"  ✅ FAITHFUL T4 MODE: single continuous pass · {render_seconds}s · "
              f"base {int(generation_width)//2}x{int(generation_height)//2} → 2x → "
              f"{generation_width}x{generation_height} · VRAM={VRAM_MODE}.")
        print("     फिर भी CUDA OOM आए तो: render_seconds और घटाएँ या generation_height घटाएँ —")
        print("     pass CONTINUOUS ही रहेगा इसलिए identity/scenes/voice सही बने रहेंगे।")
    else:
        print("  ⚠️  auto_safe_on_t4=False (छोटा GPU) — जैसा configured है वैसा ही चलेगा "
              "(single pass OOM कर सकता है)।")
elif not two_stage_base_render:
    # ── बड़ा GPU (L4/A100) ── पूरी faithful quality
    print("  ✅ FAITHFUL MODE: single continuous full-timeline pass, base custom//2 → 2x → full canvas.")

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


# LTXDirector renders latents at custom_width/custom_height; the LTXVLatentUpsampler
# then 2x-upscales. So Stage-1 base = (final target) / 2.
#   • two_stage_base_render=True  → base = generation//2 (416x240) → 2x → 832x480  (T4-fast, lower detail)
#   • two_stage_base_render=False → base = CUSTOM//2 (640x360)     → 2x → 1280x720 (FAITHFUL, needs L4/A100)
# The old bug rendered the base at the FULL 1280x720, which the 2x upscaler would
# blow up to 2560x1440. Rendering the base at custom//2 fixes that.
_base_w = _snap_div(int(generation_width) // 2, divisible_by)
_base_h = _snap_div(int(generation_height) // 2, divisible_by)
if two_stage_base_render:
    _director_render_w, _director_render_h = _base_w, _base_h
else:
    _director_render_w = _snap_div(int(custom_width) // 2, divisible_by)
    _director_render_h = _snap_div(int(custom_height) // 2, divisible_by)

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
# 🇮🇳 LoRA stack अब registry (Cell 0) से आता है — 2.3 पर 4 LoRA, 2.5 पर 3 (उसी के
# exact strengths जो 2.5 All-In-One workflow में थे)। ऊपर के use_lora_N / lora_strength_N
# knobs index के हिसाब से लागू होते हैं: use_lora_N=False तो वह LoRA बंद; अगर आपने
# lora_strength_N को उसके default से बदला तो वही strength registry पर override कर देता है।
_USE_TOGGLES  = [use_lora_1, use_lora_2, use_lora_3, use_lora_4]
_STR_TOGGLES  = [lora_strength_1, lora_strength_2, lora_strength_3, lora_strength_4]
_STR_DEFAULTS = [0.4, 0.6, 0.7, 0.9]   # panel defaults (बदले तो override माना जाएगा)
_ALL_LORAS = []
for _i, _lentry in enumerate(MODELS["loras"]):
    _name, _url, _reg_strength = _lentry[0], _lentry[1], _lentry[2]
    _on = _USE_TOGGLES[_i] if _i < len(_USE_TOGGLES) else True
    _strength = _reg_strength
    if _i < len(_STR_TOGGLES) and abs(_STR_TOGGLES[_i] - _STR_DEFAULTS[_i]) > 1e-6:
        _strength = _STR_TOGGLES[_i]     # user ने panel में बदला → override
    _ALL_LORAS.append((_on, _name, _strength))
LORA_STACK = [{"on": bool(o), "lora": n, "strength": float(s)} for (o, n, s) in _ALL_LORAS if o]
print(f"  🎛️  LoRA stack (LTX-{ACTIVE_FAMILY}): "
      + ", ".join(f"{lc['lora'].split('.')[0][:24]}@{lc['strength']}" for lc in LORA_STACK))

STAGE1 = {"scheduler": scheduler_name, "steps": int(stage1_steps), "denoise": float(stage1_denoise),
          "cfg": float(cfg), "guide_strength": float(stage1_guide_strength)}
STAGE2 = {"scheduler": scheduler_name, "steps": int(stage2_steps), "denoise": float(stage2_denoise),
          "cfg": float(cfg), "guide_strength": float(stage2_guide_strength)}
VHS_SETTINGS = {"format": "video/h264-mp4", "pix_fmt": "yuv420p",
                "crf": int(output_crf), "filename_prefix": "LTX23_Director_Master"}

# Runtime globals consumed by later cells.
# 🇮🇳 HINDI: ये वो settings हैं जिन्हें आगे के cells (Phase A/B/C/D) इस्तेमाल करते हैं।
# (scene-chunking वाले सारे globals हटा दिए गए हैं — अब सिर्फ़ single continuous pass।)
ESSENTIAL_LORAS_ONLY = bool(essential_loras_only)
VRAM_MODE = str(VRAM_MODE)
# ── Adaptive VRAM shield ─────────────────────────────────────────────────────
# 🇮🇳 HINDI: VRAM shield = इतनी VRAM खाली छोड़ना ताकि LoRA/adaln के calculations की
# जगह रहे। पर shield बहुत बड़ा हो और single-pass चल रहा हो तो ComfyUI model के कुछ
# हिस्से host RAM में डाल देता है (यही T4 पर thrashing/crash करता है)। इसलिए अगर
# आपने default (1200) नहीं बदला तो हम GPU size के हिसाब से इसे adjust कर देते हैं:
#   • छोटा GPU (T4) → छोटा shield (~640MB) ताकि पूरा model VRAM में रहे,
#   • बड़ा GPU (L4/A100) → 1200MB (LoRA dequant के लिए काफ़ी जगह)।
# आपने खुद कोई value दी हो तो उसी का सम्मान होता है।
if int(vram_shield_mb) == 1200:                       # यानी default पर ही है
    if _gpu_total_gb and _gpu_total_gb < 20.0:
        VRAM_SHIELD_MB = 640                          # T4: model VRAM में रखें
    else:
        VRAM_SHIELD_MB = 1200                         # बड़ा GPU
else:
    VRAM_SHIELD_MB = int(vram_shield_mb)              # user override
print(f"  🛡️  VRAM shield = {VRAM_SHIELD_MB} MB (GPU {_gpu_total_gb:.1f} GB)")
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
# 🇮🇳 CELL 7 का काम: ComfyUI के node-system को चालू करना और जाँचना कि JSON को
#   चाहिए वाले सभी 23 nodes सही से load हुए हैं। कोई node गायब हो तो यहीं error
#   देकर रोक देता है (ताकि आगे चलकर बीच में fail न हो)।
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
# 🇮🇳 CELL 8 का काम: memory साफ़ रखने के tools —
#   • purge_deep()  → GPU+RAM की गहरी सफाई।
#   • free_models_no_cpu_offload() → 22B model को VRAM से सीधे हटाना (host RAM में
#     copy किए बिना) → वही "session crashed" वाला crash रोकता है।
#   • ram_guard(), light/medium_clear(), mem_report(), VRAM shield आदि।
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
            if globals().get("STREAM_ENCODER", False):
                # 🌊 STREAM: encoder GPU पर compute पर offload/park CPU पर → comfy का
                # model_prefetch layers को CPU(disk-mmap)↔GPU stream करता है, पूरा 12B
                # एक साथ VRAM में नहीं आता। (peak VRAM = ~active layer + activations)
                mm.text_encoder_device = lambda: torch.device("cuda")
                mm.text_encoder_offload_device = lambda: torch.device("cpu")
            else:
                # 2.3/normal: encoder पूरा GPU पर (host-RAM spike से बचने के लिए)।
                mm.text_encoder_device = lambda: torch.device("cuda")
                mm.text_encoder_offload_device = lambda: torch.device("cuda")
    except Exception as e:
        print(f"  [mem-patch notice] {e}")


def patch_safetensors_direct_to_gpu():
    """Load text-encoder shards straight onto CUDA so host RAM never spikes.

    🌊 STREAM_ENCODER mode: encoder को CUDA पर सीधे load NA करें — CPU पर mmap रहने दें
    (safetensors डिफ़ॉल्ट रूप से file को memory-map करता है, यानी weights disk-backed
    page-cache में रहते हैं, न पूरे host RAM में न VRAM में)। तब comfy layer-by-layer
    GPU पर stream करता है।"""
    try:
        import safetensors.torch as st
        if not getattr(st, "_ltx_cuda_direct", False):
            _orig = st.load_file

            def _cuda_load(filename, device="cpu"):
                fn = str(filename).lower()
                _is_te = any(k in fn for k in
                             ["gemma", "clip", "text_encoder", "projection", "connector"])
                if (not globals().get("STREAM_ENCODER", False)) and torch.cuda.is_available() and _is_te:
                    return _orig(filename, device="cuda")
                # streaming (or non-TE): CPU/mmap — disk-backed, कम RAM व VRAM।
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


def free_models_no_cpu_offload(tag: str = ""):
    """Free loaded models straight from VRAM WITHOUT moving them to host RAM first.

    ComfyUI's default unload (mm.unload_all_models()) MOVES a model to its CPU
    offload device before releasing it. On a free T4 the 22B model is ~13 GB and
    host RAM is only ~13.6 GB (with ~10 GB free after Phase A), so that move
    OVERFLOWS host RAM and FATALLY crashes the kernel ("session crashed after
    using all available RAM") — this is exactly what happens right after Stage 2.

    By CLEARING ComfyUI's loaded-model list FIRST, unload_all_models() has nothing
    to move to host RAM; torch.cuda.empty_cache() then reclaims the VRAM directly.
    Call this (instead of purge_deep) once the model is no longer needed and the
    caller has already dropped its own reference to it.
    """
    try:
        import comfy.model_management as mm
        if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
            mm.current_loaded_models.clear()      # drop refs → nothing to offload to host RAM
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
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
    """Per-step light memory clear during sampling + progress/ETA print.

    🇮🇳 HINDI: यह हर sampling step पर (1) थोड़ी VRAM साफ़ करता है और (2) अब ETA भी
    बताता है — जैसे 'step 3/8 · 54.2s/it · ETA ~4.5 min'। LTX steps लंबे होते हैं,
    इसलिए यह जानना कि कितना समय बचा है बहुत काम आता है।"""
    try:
        import comfy.utils as cu
        state = {"n": 0, "t0": None, "total": None}

        def _hook(value, total, preview_bytes=None, *args, **kwargs):
            state["n"] += 1
            # ── ETA ──: नया sampling run शुरू होने पर timer reset करो।
            try:
                v = float(value) if value is not None else 0.0
                tot = float(total) if total else 0.0
                if state["t0"] is None or state["total"] != tot or v <= 1.0:
                    state["t0"] = time.time()
                    state["total"] = tot
                elif v > 1.0 and tot > 0.0:
                    elapsed = time.time() - state["t0"]
                    per = elapsed / max(1.0, (v - 1.0))     # औसत समय प्रति step
                    eta = per * max(0.0, tot - v)
                    log(f"step {int(v)}/{int(tot)} · {per:.1f}s/it · ETA ~{eta/60:.1f} min", "INFO")
            except Exception:
                pass
            # ── memory ──
            if clear_every <= 1 or (state["n"] % clear_every == 0):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if ram_guard_gb > 0 and (state["n"] % 4 == 0):
                if get_ram_free_gb() < ram_guard_gb:
                    gc.collect()
                    malloc_trim_os()

        if hasattr(cu, "set_progress_bar_global_hook"):
            cu.set_progress_bar_global_hook(_hook)
            print(f"  ⚙️ Per-step memory-clear + ETA hook active (every {clear_every} step[s]).")
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
# 🇮🇳 CELL 9 का काम: ComfyUI के nodes को Python से चलाने का "engine"।
#   • call_node() → किसी भी node को नाम से बुलाता है और सही parameters भर देता है।
#   • gv()/unwrap_tensor()/unwrap_latent() → nodes के output से सही value निकालते हैं।
#   • sync_latent_device() → latents को CPU पर रखकर VRAM बचाता है।
#   • tiled_decode_video() → VAE decode को टुकड़ों में करता है (RAM safe)।
#   • prepare_reference_image() → keyframe photo को सही size में लाता है।
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
                # 🇮🇳 यहाँ पहुँचना खतरनाक है: यह एक REQUIRED parameter है जिसकी कोई
                # value हमें नहीं मिली (न kwargs में, न schema-default में)। नीचे हम
                # type देखकर एक "अंदाज़ा" (0/0.0/False/""/None) भर देते हैं — इससे node
                # चुपचाप गलत output दे सकता है। इसलिए इसे WARN पर दिखाते हैं ताकि पता चले।
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
                log(f"call_node('{node_name}'): required param '{p_name}' की कोई value नहीं मिली "
                    f"→ fallback '{valid[p_name]!r}' भरा (output गलत हो सकता है — जाँचें)।", "WARN")
            if has_kwargs:
                for k, v in kwargs.items():
                    valid.setdefault(k, v)
            return func(**valid)
        except Exception:
            last_err = traceback.format_exc()
            _dbg(f"call_node('{node_name}'): callable '{getattr(func, '__name__', func)}' fail "
                 f"— अगला try कर रहे हैं।")
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
# 🇮🇳 CELL 10 का काम: वही "timeline_data" JSON और widget list बनाना जो असली
#   LTXDirector node (id 131) में होती है — यानी 5 keyframes + 1 audio track +
#   motion track + global prompt, सब एक साथ। साथ ही keyframe photos check करता है
#   (न मिलें तो placeholder बना देता है ताकि आप अपनी photo डाल सकें)।
# ════════════════════════════════════════════════════════════════════════════
class DirectorTimelineController:
    """
    Reconstructs the exact `timeline_data` JSON + widget list that the LTXDirector
    node (id 131) carries in the workflow, so the single Master Timeline Controller
    receives all 3 tracks (main / audio / motion) and the global prompt at once.
    THIS is the "notes" object the old Master_V2 never built.
    """

    def __init__(self, global_prompt, negative_prompt, meta, segments,
                 audio_segments, motion_segments, base_input_dir=INPUT_DIR):
        self.global_prompt = global_prompt
        self.negative_prompt = negative_prompt
        self.meta = meta
        self.segments = segments
        self.audio_segments = audio_segments
        self.motion_segments = motion_segments
        self.base_input_dir = base_input_dir
        self.validate_reference_images()

    def validate_reference_images(self):
        """🇮🇳 सभी keyframe photos जाँचता है। जो न मिलें उनके लिए placeholder बनाता है
        और गिनती बताता है — अगर कोई placeholder बना तो साफ़ WARNING देता है (मतलब आपने
        वह photo upload नहीं की, और video की quality गिरेगी)।"""
        print("\n" + "=" * 70 + "\n🔍 VALIDATING DIRECTOR KEYFRAMES (main track)\n" + "=" * 70)
        self.n_real_keyframes = 0
        self.n_placeholder_keyframes = 0
        for s in self.segments:
            full = os.path.join(self.base_input_dir, s["imageFile"])
            if not os.path.exists(full):
                if "5.3.png" in s["imageFile"]:
                    alt = full.replace("5.3.png", "5.png")
                    if os.path.exists(alt):
                        os.makedirs(os.path.dirname(full), exist_ok=True)
                        shutil.copyfile(alt, full)
                        print(f"  ✓ Alias resolved 5.png → {full}")
                        self.n_real_keyframes += 1
                        continue
                os.makedirs(os.path.dirname(full), exist_ok=True)
                ph = Image.new("RGB", (768, 512), color=(40, 30, 70))
                ImageDraw.Draw(ph).text((40, 230),
                    f"UPLOAD singer photo → {os.path.basename(s['imageFile'])}", fill=(255, 255, 255))
                ph.save(full)
                self.n_placeholder_keyframes += 1
                print(f"  ⚠️  Placeholder created (upload your photo): {full}")
            else:
                self.n_real_keyframes += 1
                print(f"  ✓ Keyframe OK: {full}")
        total = len(self.segments)
        print(f"  📸 Keyframes: {self.n_real_keyframes}/{total} असली, "
              f"{self.n_placeholder_keyframes} placeholder.")
        if self.n_placeholder_keyframes:
            log(f"{self.n_placeholder_keyframes} keyframe(s) placeholder हैं — असली singer photos "
                f"({', '.join(os.path.basename(s['imageFile']) for s in self.segments)}) "
                f"{self.base_input_dir} में upload करें, वरना video सही नहीं बनेगी।", "WARN")

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
# 🇮🇳 CELL 11 का काम: 22B model load करना और उस पर 4-LoRA stack (0.4/0.6/0.7/0.9)
#   लगाना — बिल्कुल JSON जैसा। फिर memory-बचाने वाले hooks (SageAttention,
#   ChunkFeedForward) और preview override लगाता है। (model, clip) लौटाता है।
# ════════════════════════════════════════════════════════════════════════════
def load_dit_and_loras(clip_obj: Any = None):
    """
    UnetLoaderGGUF ─► Power Lora Loader (rgthree) ─► ModelPreviewOverrideKJ hook.
    Applies the exact 4-LoRA stack (0.4/0.6/0.7/0.9) from the JSON. Falls back to
    LoraLoaderModelOnly if the rgthree signature differs. Returns (model, clip).
    """
    purge_deep("pre_dit_load")
    mem_report("DiT load", "UnetLoaderGGUF")
    # 🇮🇳 quant नाम DIT_GGUF_NAME से आता है (Cell 5 में चुना गया — default Q4_K_M)।
    _gguf = globals().get("DIT_GGUF_NAME", "ltx-2-3-22b-dev-Q4_K_M.gguf")
    model = gv(call_node("UnetLoaderGGUF", unet_name=_gguf), 0)
    print(f"  ✓ UnetLoaderGGUF loaded ({_gguf}).")

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
            path = os.path.join(MODELS_DIR, "loras", lc["lora"])
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
    # 🇮🇳 अब silent 'pass' की जगह DEBUG log — hook fail हो तो पता चले (quality पर असर)।
    if "PatchSageAttentionKJ" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node("PatchSageAttentionKJ", model=model, sage_attention="auto"), 0) or model
            print("  ✓ SageAttention hook applied.")
        except Exception as e:
            _dbg(f"SageAttention hook लगा नहीं ({e}) — बिना इसके भी चलेगा (थोड़ा slower)।")
    if "LTXVChunkFeedForward" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node("LTXVChunkFeedForward", model=model, chunks=8, dim_threshold=4096), 0) or model
            print("  ✓ ChunkFeedForward hook applied (chunks=8).")
        except Exception as e:
            _dbg(f"ChunkFeedForward hook लगा नहीं ({e})।")

    # ModelPreviewOverrideKJ (id 10) — tiny-VAE preview override; pass-through here.
    if "ModelPreviewOverrideKJ" in NODE_CLASS_MAPPINGS:
        try:
            tiny_vae = gv(call_node("VAELoaderKJ", vae_name=MODELS["tiny_vae"][0],
                                    device="main_device", weight_dtype="bf16"), 0)
            model = gv(call_node("ModelPreviewOverrideKJ", model=model, vae=tiny_vae), 0) or model
            print("  ✓ ModelPreviewOverrideKJ applied.")
        except Exception as e:
            _dbg(f"ModelPreviewOverrideKJ लगा नहीं ({e})।")

    # 🇮🇳 torch.compile — सिर्फ़ बड़े GPU (L4/A100) पर, जहाँ compile का एक-बार का
    # overhead sampling की speed-up से वसूल हो जाता है। T4 पर यह फ़ायदेमंद नहीं,
    # इसलिए वहाँ छोड़ देते हैं। LTX_TORCH_COMPILE=0 से बंद भी कर सकते हैं।
    _want_compile = (globals().get("FULL_QUALITY_HW", False)
                     and os.environ.get("LTX_TORCH_COMPILE", "1") != "0")
    if _want_compile and hasattr(torch, "compile"):
        try:
            if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
                model.model.diffusion_model = torch.compile(
                    model.model.diffusion_model, mode="reduce-overhead", dynamic=True)
                print("  ✓ torch.compile applied to the DiT (big-GPU speed-up).")
        except Exception as e:
            _dbg(f"torch.compile लगा नहीं ({e}) — बिना इसके भी चलेगा।")

    # 🇮🇳 LoRA/hooks लगने के बाद थोड़ी हल्की सफाई (VRAM/RAM टुकड़े जमा न हों)।
    medium_clear("post_dit_load")
    return model, clip_obj


print("✅ Cell 11: DiT + 4-LoRA loader with attention hooks ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 12: PHASE A — MASTER TIMELINE INGESTION (LTXDirector, full 756 frames)
# 🇮🇳 CELL 12 (Phase A) का काम: timeline तैयार करना। सबसे बड़ी memory-चाल यहीं है:
#   Gemma-12B text encoder और 22B model एक साथ RAM में नहीं आ सकते, इसलिए —
#     1) पहले सिर्फ़ text encoder load करके prompt encode करते हैं, फिर उसे purge
#        (हटा) देते हैं ताकि RAM खाली हो जाए।
#     2) फिर असली 22B model + 4 LoRA load करते हैं।
#     3) LTXDirector चलाकर एक ही shared video latent + एक ही shared audio latent
#        बनाते हैं (यही voice-sync की जड़ है)। नतीजा director_state.pt में cache।
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
    """(Dual)CLIPLoader → CLIPTextEncode on CUDA, then purge the encoder weights
    while keeping the tiny tokenizer wrapper. Returns (cond_on_cpu, tokenizer).

    🇮🇳 HINDI: encoder का तरीका registry के 'encoder_mode' से तय होता है —
      • LTX-2.3 → "dual": DualCLIPLoader (Gemma + अलग projection file)।
      • LTX-2.5 → "single": CLIPLoader (projection encoder में ही fused है — "with-proj")।
    दोनों में weights CUDA पर load होते हैं (host RAM spike न हो), encode के बाद purge।"""
    purge_deep("pre_clip_load")
    _enc_name = MODELS["text_encoder"][0]
    _clip_type = MODELS.get("clip_type", "ltxv")
    _stream = bool(globals().get("STREAM_ENCODER", False))

    # ── PREFLIGHT: क्या यह encoder इस hardware पर fit होगा? ────────────────────
    # 🇮🇳 सबक (एक fatal session-crash से): 12B encoder को GPU पर पूरा रखो तो VRAM OOM
    # (catchable), पर CPU/stream पर डालो तो वह पूरा HOST RAM में materialize होता है
    # (comfy_kitchen int8 mmap-view नहीं रखता) → host RAM भर जाए तो पूरा kernel मर
    # जाता है (session crash, uncatchable)। इसलिए load करने से पहले ही जाँच लें: अगर
    # न GPU में fit है, न host RAM में — तो load की कोशिश ही मत करो, साफ़ error दो।
    if MODELS.get("encoder_mode", "dual") == "single":   # सिर्फ़ 2.5-style बड़ा fused encoder
        _forced = str(os.environ.get("LTX_FORCE_25_ENCODER", "0")).strip().lower() not in ("0", "false", "no", "")
        _enc_path = os.path.join("/content/ComfyUI/models/text_encoders", _enc_name)
        try:
            _enc_gb = os.path.getsize(_enc_path) / 1e9
        except Exception:
            _enc_gb = 12.0
        _vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0.0
        _ram_free = get_ram_free_gb()
        _gpu_ok = _vram_gb >= (_enc_gb + 3.0)          # पूरा encoder + dequant/activation headroom
        _stream_ok = _stream and (_ram_free >= (_enc_gb + 2.0))   # CPU copy को host RAM में जगह
        print(f"  🔎 Encoder preflight: file ~{_enc_gb:.1f}GB · VRAM ~{_vram_gb:.1f}GB · "
              f"host RAM free ~{_ram_free:.1f}GB · gpu_fit={_gpu_ok} · stream_fit={_stream_ok}")
        if not (_gpu_ok or _stream_ok or _forced):
            raise TextEncoderOOM(
                f"LTX-{ACTIVE_FAMILY} का text-encoder ({_enc_name}, ~{_enc_gb:.1f}GB) इस hardware पर "
                f"fit नहीं होता:\n"
                f"    • GPU पर पूरा रखने के लिए ~{_enc_gb+3.0:.0f}GB VRAM चाहिए, आपके पास ~{_vram_gb:.1f}GB।\n"
                f"    • CPU/stream पर रखने के लिए ~{_enc_gb+2.0:.0f}GB free host RAM चाहिए, आपके पास ~{_ram_free:.1f}GB।\n"
                f"  दोनों में से कोई पूरा नहीं पड़ता — इसलिए load की कोशिश नहीं की (वरना host-RAM भरकर "
                f"पूरा session crash हो जाता, जैसा पिछली बार हुआ)।\n"
                f"  ✅ इस GPU पर: MODEL_FAMILY='2.3' (T4 पर पूरा चलता है)।\n"
                f"  ✅ LTX-2.5 चाहिए तो: L4/A100 (24GB+) runtime।\n"
                f"  (जोखिम उठाकर फिर भी कोशिश करनी हो: os.environ['LTX_FORCE_25_ENCODER']='1' — "
                f"पर यह session crash कर सकता है।)"
            )
        if _stream and not _stream_ok and not _forced:
            # streaming माँगा पर host RAM कम → streaming बंद करो (वरना crash); GPU path पर जाओ
            # (जहाँ कम-से-कम catchable OOM आएगा)। ऊपर _gpu_ok True था इसलिए यहाँ safe है।
            print("  ⚠️ host RAM streaming के लिए कम है → streaming बंद, encoder GPU पर load होगा।")
            _stream = False
    if _stream:
        # 🌊 encode के दौरान LOW_VRAM → comfy encoder layers को CPU/disk से stream करे।
        try:
            import comfy.model_management as mm
            _vs = getattr(mm, "VRAMState", None)
            if _vs is not None:
                mm.vram_state = _vs.LOW_VRAM
            print("  🌊 STREAM_ENCODER: LOW_VRAM में encoder layer-by-layer (CPU/disk-backed) "
                  "stream कर रहे हैं — peak VRAM कम, पर धीमा (experimental)।")
        except Exception:
            pass
    if MODELS.get("encoder_mode", "dual") == "single":
        # LTX-2.5: single fused encoder.
        mem_report("Phase A", f"CLIPLoader ({_enc_name}) on GPU")
        import comfy.model_management as mm
        clip = gv(call_node("CLIPLoader",
                            clip_name=_enc_name, type=_clip_type, device="default"), 0)
    else:
        # LTX-2.3: Gemma + separate projection.
        mem_report("Phase A", f"DualCLIPLoader ({_enc_name} + projection) on GPU")
        import comfy.model_management as mm
        clip = gv(call_node("DualCLIPLoader",
                            clip_name1=_enc_name,
                            clip_name2=(MODELS["text_proj"][0] if MODELS.get("text_proj") else _enc_name),
                            type=_clip_type, device="default"), 0)
    saved_tokenizer = getattr(clip, "tokenizer", None)

    print("  ⚡ Encoding global prompt on GPU...")
    t0 = time.time()
    try:
        with torch.inference_mode():
            cond_raw = gv(call_node("CLIPTextEncode", text=prompt_text, clip=clip), 0)
            cond_cpu = sync_cond_to_cpu(cond_raw)
            del cond_raw
    except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
        if "out of memory" not in str(_e).lower():
            raise
        # 🇮🇳 encoder OOM — resolution घटाने से कोई फ़ायदा नहीं (encoder res-independent है)।
        try:
            del clip
        except Exception:
            pass
        purge_deep("encoder_oom")
        _vg = (torch.cuda.get_device_properties(0).total_memory / 1e9
               if torch.cuda.is_available() else 0.0)
        raise TextEncoderOOM(
            f"Text-encoder ({_enc_name}) VRAM में fit नहीं हुआ (GPU ~{_vg:.1f}GB)।\n"
            f"  LTX-{ACTIVE_FAMILY} का 12B Gemma-4 encoder ~24-27GB VRAM माँगता है "
            f"(Lightricks की official requirement) — T4/16GB पर यह चलता ही नहीं।\n"
            f"  👉 इस GPU पर: MODEL_FAMILY='2.3' इस्तेमाल करें (वह T4 पर पूरा चलता है)।\n"
            f"  👉 LTX-2.5 चाहिए तो: L4/A100 (24GB+) runtime लें।"
        ) from _e
    print(f"  ✓ Prompt encoded in {time.time() - t0:.2f}s. Purging encoder weights...")

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
    if _stream:
        # 🌊 encoder हो गया → DiT के लिए VRAM_MODE (आमतौर पर normalvram) पर वापस।
        try:
            configure_vram_state(str(globals().get("VRAM_MODE", "normalvram")))
        except Exception:
            pass
    mem_report("Phase A", "Encoder weights purged (tokenizer kept)")
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
        audio_vae = gv(call_node("VAELoader", vae_name=MODELS["audio_vae"][0]), 0)
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

        # ── 🇮🇳 KEYFRAME/LATENT जाँच ────────────────────────────────────────
        # यह पक्का करता है कि LTXDirector ने सचमुच सारे keyframes और एक असली audio
        # latent बनाया। अगर audio latent खाली हुआ तो voice नहीं आएगी → साफ़ WARN।
        try:
            _vt = unwrap_latent(dir_vid).get("samples", None)
            _at = unwrap_latent(dir_aud).get("samples", None)
            _vT = int(_vt.shape[2]) if (isinstance(_vt, torch.Tensor) and _vt.dim() >= 3) else 0
            _aT = int(_at.shape[2]) if (isinstance(_at, torch.Tensor) and _at.dim() >= 3) else 0
            _n_kf = len(timeline_ctrl.segments)
            _real_kf = getattr(timeline_ctrl, "n_real_keyframes", _n_kf)
            print(f"  🔍 Director जाँच: video latent frames={_vT}, audio latent frames={_aT}, "
                  f"keyframes(दिए गए)={_n_kf} (असली photos={_real_kf})")
            if _vT <= 0:
                log("Director ने खाली video latent दिया — video नहीं बनेगी। "
                    "settings/keyframes जाँचें।", "ERROR")
            if bool(m.get("use_custom_audio")) and _aT <= 0:
                log("Director ने खाली AUDIO latent दिया — voice नहीं आएगी/lip-sync टूटेगी। "
                    "song file और use_song_audio जाँचें।", "WARN")
            if getattr(timeline_ctrl, "n_placeholder_keyframes", 0) > 0:
                log(f"{timeline_ctrl.n_placeholder_keyframes} keyframe placeholder थे — इसलिए "
                    f"character सही नहीं दिखेगा। असली photos upload करके दोबारा चलाएँ।", "WARN")
            # guide_data से image-guides की संख्या (best-effort) — अगर 5 से कम हो तो चेताएँ।
            _gd = dir_guide
            _n_guides = None
            if isinstance(_gd, dict):
                for _k in ("image_guides", "guides", "keyframes"):
                    if isinstance(_gd.get(_k), (list, tuple)):
                        _n_guides = len(_gd[_k]); break
            elif isinstance(_gd, (list, tuple)):
                _n_guides = len(_gd)
            if _n_guides is not None:
                _dbg(f"guide_data में image-guides ≈ {_n_guides} (keyframes {_n_kf} थे)।")
                if _n_guides < _real_kf:
                    log(f"सिर्फ़ {_n_guides} keyframe guide बने पर आपने {_real_kf} दिए — "
                        f"कुछ keyframes शायद timeline में इस्तेमाल नहीं हुए (render_seconds "
                        f"बहुत छोटा तो नहीं?)।", "WARN")
        except Exception as e:
            _dbg(f"keyframe/latent जाँच skip हुई ({e})।")

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
# 🇮🇳 CELL 13 (Phase B) का काम: असली video generate करना — पूरी timeline एक साथ
#   (single continuous pass), बिल्कुल JSON जैसा:
#     • Stage 1: छोटे resolution पर base video (8 steps, denoise 1.0, guide 0.5)
#     • फिर 2x upscale
#     • Stage 2: refine (4 steps, denoise 0.42, guide 1.0)
#   दोनों stage एक ही seed पर → character consistent। audio+video एक साथ चलते हैं
#   → voice synced। (scene-chunking पूरी तरह हटा दिया गया है।)
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
    """एक diffusion stage: LTXDirectorGuide → ConcatAV → Sample → SeparateAV → CropGuides.

    🇮🇳 HINDI: यह JSON के एक sampling stage की हूबहू नकल है —
      1) LTXDirectorGuide: keyframes का guidance video latent में डालता है।
      2) LTXVConcatAVLatent: video + audio latent को एक साथ जोड़ता है (joint AV)।
      3) RandomNoise + CFGGuider + KSamplerSelect + BasicScheduler + SamplerCustomAdvanced:
         असली denoising (यहीं video "बनती" है)।
      4) LTXVSeparateAVLatent: video और audio latent फिर अलग करता है।
      5) LTXDirectorCropGuides: guide frames हटाता है।
    Stage 1 और Stage 2 दोनों यही function अलग settings के साथ बुलाते हैं।
    """
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


# 🇮🇳 HINDI: यहाँ पहले scene-chunking वाले helper functions थे (_lat_samples,
# _slice_lat_T, _blend_append_T, _keyframe_scene_ranges, _cap_scene_ranges,
# _fixed_scene_ranges) और execute_phase_b_batched()। ये सब timeline को टुकड़ों में
# काटकर अलग-अलग generate करते थे → character बदल जाता, scenes में जोड़ दिखते और
# आवाज़ desync हो जाती। इन सबको पूरी तरह हटा दिया गया है। अब सिर्फ़ नीचे वाला
# _execute_phase_b_single() (पूरी timeline एक साथ) चलता है।


def execute_phase_b(director_state: Dict[str, Any], model: Any, seed: int,
                    workdir: str, resume: bool = True) -> str:
    """Phase B चलाने वाला function — सिर्फ़ FAITHFUL single continuous pass।

    🇮🇳 HINDI: पहले जो 'batch_scene_mode' (timeline को टुकड़ों में काटना) था वह पूरी
    तरह हटा दिया गया है, क्योंकि वही character-drift, scene-seams और voice-desync
    की वजह था। अब हमेशा पूरी timeline एक ही बार में generate होती है — बिल्कुल
    ComfyUI JSON की तरह → consistent character + continuous scenes + synced voice.

    अगर T4 पर सच में memory कम पड़े (CUDA OOM) तो यह साफ़ error देता है और बताता है
    कि render_seconds/resolution घटाएँ — गलत quality वाली chunked video बनाने के
    बजाय। (chunking अब मौजूद ही नहीं है।)
    """
    latent_file = os.path.join(workdir, "final_latents.pt")
    if resume and os.path.exists(latent_file) and os.path.getsize(latent_file) > 1024:
        print(f"  ⏭ [RESUME] पहले से बने final latents मिल गए: {latent_file}")
        return latent_file
    try:
        return _execute_phase_b_single(director_state, model, seed, workdir, resume)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # 🇮🇳 OOM हो तो VRAM साफ़ करके error आगे भेज देते हैं — ऊपर run_...() का
        # "OOM retry ladder" इसे पकड़कर resolution घटाकर अपने-आप दोबारा कोशिश करेगा।
        if "out of memory" in str(e).lower():
            purge_deep("single_pass_oom")
        raise


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

    # 🇮🇳 Stage-1 checkpoint: Stage 1 (~8 min) के बाद उसका upscaled latent disk पर
    # save कर देते हैं। अगर Stage 2 crash करे तो दोबारा चलाने पर Stage 1 फिर से नहीं
    # करना पड़ता — सीधे Stage 2 से शुरू हो जाता है (आपका समय बचता है)।
    stage1_file = os.path.join(workdir, "stage1_latents.pt")

    with torch.inference_mode():
        if resume and os.path.exists(stage1_file) and os.path.getsize(stage1_file) > 1024:
            print(f"  ⏭ [RESUME] Stage 1 (+upscale) पहले से हो चुका → सीधे Stage 2। ({stage1_file})")
            _pk = torch.load(stage1_file, map_location="cpu")
            s1_pos, s1_neg = _pk["s1_pos"], _pk["s1_neg"]
            v_ups = {"samples": _pk["v_ups"]}
            s1_aud = {"samples": _pk["s1_aud"]} if _pk.get("s1_aud") is not None else None
            model = _ensure_model(model)
            video_vae = gv(call_node("VAELoader", vae_name=MODELS["video_vae"][0]), 0)
        else:
            model = _ensure_model(model)
            video_vae = gv(call_node("VAELoader", vae_name=MODELS["video_vae"][0]), 0)

            # Stage 1 (base res, 8 steps, denoise 1.0, guide 0.5)
            s1_pos, s1_neg, s1_vid, s1_aud, _ = _run_stage(
                model, video_vae, STAGE1["guide_strength"], base_pos, base_neg,
                director_state["video_latent"], director_state["audio_latent"],
                guide_data, motion_guide,
                STAGE1["scheduler"], STAGE1["steps"], STAGE1["denoise"], STAGE1["cfg"],
                seed, "STAGE 1")
            medium_clear("single_post_s1")

            # 2x latent spatial upscale
            up_model = gv(call_node("LatentUpscaleModelLoader",
                                    model_name=MODELS["upscaler"][0]), 0)
            v_ups = sync_latent_device(gv(call_node("LTXVLatentUpsampler",
                                                    samples=s1_vid, upscale_model=up_model,
                                                    vae=video_vae), 0), "cpu")
            del up_model, s1_vid
            medium_clear("single_post_upscale")

            # 💾 Stage 1 का नतीजा checkpoint करें (Stage-2 crash पर दोबारा नहीं करना पड़ेगा)।
            _save_ckpt(stage1_file, {
                "s1_pos": sync_cond_to_cpu(s1_pos),
                "s1_neg": sync_cond_to_cpu(s1_neg),
                "v_ups": unwrap_tensor(v_ups).detach().cpu().half(),
                "s1_aud": (unwrap_tensor(s1_aud).detach().cpu().half()
                           if s1_aud is not None else None),
            })
            print(f"  💾 Stage-1 latent checkpointed → {stage1_file}")

        # Stage 2 (refine at full res, 4 steps, denoise 0.42, guide 1.0)
        s2_pos, s2_neg, s2_vid, s2_aud, _ = _run_stage(
            model, video_vae, STAGE2["guide_strength"], s1_pos, s1_neg,
            v_ups, s1_aud, guide_data, motion_guide,
            STAGE2["scheduler"], STAGE2["steps"], STAGE2["denoise"], STAGE2["cfg"],
            seed, "STAGE 2")

        final_video_lat = unwrap_tensor(s2_vid).detach().cpu().half()
        final_audio_lat = unwrap_tensor(s2_aud).detach().cpu().half() if s2_aud is not None else None

        del model, video_vae, v_ups, s1_pos, s1_neg, s1_aud, s2_pos, s2_neg, s2_vid, s2_aud
        # Free the 22B model straight from VRAM. purge_deep()'s unload_all_models()
        # would MOVE it to the CPU offload device (~13 GB) → overflow the 13.6 GB
        # host RAM → FATAL "session crashed" (this is where Stage-2 runs died).
        free_models_no_cpu_offload("phase_b_pre_save")

        torch.save({"video": final_video_lat, "audio": final_audio_lat,
                    "frame_rate": director_state["frame_rate"]}, latent_file + ".tmp")
        os.replace(latent_file + ".tmp", latent_file)
        print(f"  💾 Final timeline latents saved: {latent_file}")
        del final_video_lat, final_audio_lat
        # 🇮🇳 अब Stage-1 का अस्थायी checkpoint बेकार है → हटा दें (disk बचाएँ)।
        try:
            if os.path.exists(stage1_file):
                os.remove(stage1_file)
        except Exception as e:
            _dbg(f"stage1 checkpoint हटा नहीं सके ({e})।")

    purge_deep("phase_b_complete")
    mem_report("Phase B complete")
    return latent_file


print("✅ Cell 13: Phase B (2-stage diffusion over the shared timeline) ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 14: PHASE C — CHUNKED STREAMING VAE DECODE (video) + AUDIO DECODE
# 🇮🇳 CELL 14 (Phase C) का काम: latents को असली video frames + audio में बदलना।
#   पूरी video एक साथ decode करने पर RAM भर जाती है, इसलिए इसे छोटे-छोटे temporal
#   chunks में decode करके हर chunk को सीधे एक छोटी MP4 में लिख देता है, फिर सब
#   MP4 जोड़ देता है। audio अलग से LTXVAudioVAEDecode से decode होता है।
# ════════════════════════════════════════════════════════════════════════════
# RAM-safe: the old code decoded ALL 753 frames in one shot → a 3.85 GB float
# tensor + a 1.9 GB half copy + a re-load in Phase D → OOM-kill on a 13 GB T4.
# Now we decode the video latent in TEMPORAL CHUNKS and stream each chunk's frames
# straight into the MP4 writer as uint8, so the full RGB video is never in RAM.
DECODE_CHUNK_LAT_FRAMES = 8    # latent frames per decode chunk (~57 px frames) — small = RAM-safe on T4
DECODE_CHUNK_OVERLAP = 1       # latent-frame context each side (dropped) to avoid seams


def _save_audio_wav(audio_dict: Any, wav_path: str, fallback_sr: int = 48000) -> bool:
    """Write a ComfyUI AUDIO dict ({'waveform','sample_rate'}) to a WAV file."""
    try:
        if not isinstance(audio_dict, dict) or "waveform" not in audio_dict:
            return False
        wf = audio_dict["waveform"]
        sr = int(audio_dict.get("sample_rate", fallback_sr))
        if not isinstance(wf, torch.Tensor):
            return False
        w = wf.detach().cpu().float()
        while w.dim() > 2:      # [B,C,S] → [C,S]
            w = w[0]
        if w.dim() == 1:
            w = w.unsqueeze(0)
        import torchaudio
        torchaudio.save(wav_path, w, sr)
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 100
    except Exception as e:
        print(f"  [notice] audio wav save failed ({e}).")
        return False


def execute_phase_c(latent_file: str, workdir: str, fps: int, crf: int,
                    resume: bool = True) -> Tuple[str, str]:
    raw_video = os.path.join(workdir, "raw_video_noaudio.mp4")
    audio_file = os.path.join(workdir, "decoded_audio.pt")
    chunk_dir = os.path.join(workdir, "dec_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    need_video = not (resume and os.path.exists(raw_video) and os.path.getsize(raw_video) > 1024)
    need_audio = not (resume and os.path.exists(audio_file))

    pack = torch.load(latent_file, map_location="cpu") if (need_video or need_audio) else None

    # ── C1: PER-CHUNK RESUMABLE VIDEO decode → one small MP4 per chunk → concat ──
    # Each chunk is decoded to its own vchunk_NNN.mp4. A crash only loses the
    # in-progress chunk; re-running SKIPS finished chunks and continues. The full
    # RGB video is never held in RAM (peak ≈ one chunk).
    if need_video:
        print("\n" + "=" * 70 + "\n🎬 PHASE C1: PER-CHUNK RESUMABLE VIDEO DECODE\n" + "=" * 70)
        purge_deep("pre_video_decode")
        import imageio
        v_full = unwrap_latent({"samples": pack["video"]})["samples"]   # [B,C,T,H,W]
        T = int(v_full.shape[2]) if (v_full is not None and v_full.dim() >= 3) else 1
        chunk = max(2, int(globals().get("DECODE_CHUNK_LAT_FRAMES", 8)))
        ov = max(0, int(globals().get("DECODE_CHUNK_OVERLAP", 1)))

        # Deterministic chunk list (same every run → resume maps correctly).
        chunk_ranges, start = [], 0
        while start < T:
            chunk_ranges.append((start, min(start + chunk, T)))
            start += chunk

        video_vae = None
        for ci, (s, e) in enumerate(chunk_ranges):
            seg_mp4 = os.path.join(chunk_dir, f"vchunk_{ci:03d}.mp4")
            if resume and os.path.exists(seg_mp4) and os.path.getsize(seg_mp4) > 512:
                print(f"  ⏭ [RESUME] chunk {ci+1}/{len(chunk_ranges)} already decoded.")
                continue
            if video_vae is None:
                video_vae = gv(call_node("VAELoader", vae_name=MODELS["video_vae"][0]), 0)
            with torch.inference_mode():
                ctx = max(0, s - ov) if s > 0 else 0
                sub = v_full[:, :, ctx:e].float()
                frames = unwrap_tensor(tiled_decode_video({"samples": sub}, video_vae, tile_size=256)).clamp(0, 1)
                drop_px = (s - ctx) * 8 if s > 0 else 0
                fslice = frames[drop_px:] if drop_px > 0 else frames
                arr = (fslice.cpu().numpy() * 255.0).astype(np.uint8)
            tmp_seg = seg_mp4 + ".tmp.mp4"
            w = imageio.get_writer(tmp_seg, fps=int(fps), codec="libx264", format="FFMPEG",
                                   macro_block_size=None,
                                   ffmpeg_params=["-crf", str(int(crf)), "-pix_fmt", "yuv420p"])
            for i in range(arr.shape[0]):
                w.append_data(arr[i])
            w.close()
            os.replace(tmp_seg, seg_mp4)
            print(f"  🎨 chunk {ci+1}/{len(chunk_ranges)} (latent {s}:{e}) → {arr.shape[0]} px frames → {os.path.basename(seg_mp4)}")
            del sub, frames, fslice, arr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            malloc_trim_os()
        if video_vae is not None:
            del video_vae
        del v_full
        purge_deep("post_video_decode")

        # Concatenate all chunk MP4s (same codec/params → lossless -c copy concat).
        seglist = sorted(glob.glob(os.path.join(chunk_dir, "vchunk_*.mp4")))
        listfile = os.path.join(chunk_dir, "concat_list.txt")
        with open(listfile, "w") as fh:
            for sp in seglist:
                fh.write(f"file '{sp}'\n")
        run_cmd(f'ffmpeg -y -f concat -safe 0 -i "{listfile}" -c copy "{raw_video}"', silent=False)
        if not (os.path.exists(raw_video) and os.path.getsize(raw_video) > 1024):
            # Fallback: re-encode concat if stream-copy failed.
            run_cmd(f'ffmpeg -y -f concat -safe 0 -i "{listfile}" -c:v libx264 -crf {int(crf)} '
                    f'-pix_fmt yuv420p "{raw_video}"', silent=False)
        print(f"  💾 Concatenated {len(seglist)} chunks → {raw_video}")
    else:
        print(f"  ⏭ [RESUME] Raw video cached: {raw_video}")

    # ── C2: AUDIO decode via LTXVAudioVAEDecode → the synced vocals ──
    if need_audio:
        print("\n" + "=" * 70 + "\n🎬 PHASE C2: AUDIO VAE DECODE (synced vocals)\n" + "=" * 70)
        purge_deep("pre_audio_decode")
        with torch.inference_mode():
            a_lat = pack["audio"] if pack is not None else None
            if a_lat is not None:
                audio_vae = gv(call_node("VAELoader", vae_name=MODELS["audio_vae"][0]), 0)
                decoded_audio = gv(call_node("LTXVAudioVAEDecode",
                                             samples={"samples": a_lat.float()}, audio_vae=audio_vae), 0)
                torch.save(decoded_audio, audio_file + ".tmp")
                os.replace(audio_file + ".tmp", audio_file)
                print(f"  💾 Decoded synced audio → {audio_file}")
                del audio_vae, decoded_audio
            else:
                torch.save(None, audio_file)
                print("  ⚠️ No audio latent; the raw song will be muxed in Phase D.")
        purge_deep("post_audio_decode")
    else:
        print(f"  ⏭ [RESUME] Audio cached: {audio_file}")

    del pack
    gc.collect()
    malloc_trim_os()
    return raw_video, audio_file


print("✅ Cell 14: Phase C (chunked streaming video decode + audio decode) ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 15: PHASE D — FINAL AUDIO MUX  (video already streamed to MP4 in Phase C)
# 🇮🇳 CELL 15 (Phase D) का काम: video और audio को ffmpeg से जोड़कर final MP4 बनाना।
#   पहले model की बनाई synced vocals लगाता है; न मिलें तो original song (trim करके);
#   वह भी न हो तो सिर्फ़ video (बिना audio)।
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_d(raw_video: str, audio_file: str, fps: int, crf: int,
                    outdir: str, song_path: str, trim_start_frames: float) -> str:
    os.makedirs(outdir, exist_ok=True)
    final_path = os.path.join(outdir, "LTX23_Director_Master_30s.mp4")

    print("\n" + "=" * 70 + "\n🎬 PHASE D: FINAL AUDIO MUX\n" + "=" * 70)
    purge_deep("pre_mux")

    if not (os.path.exists(raw_video) and os.path.getsize(raw_video) > 1024):
        raise RuntimeError(f"Raw video missing: {raw_video}")

    audio_dict = torch.load(audio_file, map_location="cpu") if os.path.exists(audio_file) else None
    muxed = False
    wav_path = os.path.join(outdir, "_synced_audio.wav")

    def _mux_model_vocals() -> bool:
        # The MODEL-generated synced vocals (from LTXVAudioVAEDecode). In the
        # faithful single pass these are continuous and lip-synced (best result).
        if audio_dict is None or not _save_audio_wav(audio_dict, wav_path):
            return False
        print("  🎵 Muxing model-generated synced vocals...")
        cmd = (f'ffmpeg -y -i "{raw_video}" -i "{wav_path}" -map 0:v:0 -map 1:a:0 '
               f'-c:v copy -c:a aac -b:a 320k -shortest "{final_path}"')
        ok = run_cmd(cmd, silent=False) == 0 and os.path.exists(final_path)
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return ok

    def _mux_original_song() -> bool:
        # The CONTINUOUS original song, trimmed to the timeline's trimStart. This
        # never drifts, so it is the safer choice when scene-chunk mode was used.
        if not (song_path and os.path.exists(song_path)):
            return False
        print("  🎵 Muxing original song track (trimmed to timeline)...")
        trim_sec = float(trim_start_frames) / float(fps)
        cmd = (f'ffmpeg -y -i "{raw_video}" -ss {trim_sec} -i "{song_path}" '
               f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -shortest "{final_path}"')
        return run_cmd(cmd, silent=False) == 0 and os.path.exists(final_path)

    # 🇮🇳 HINDI: चूँकि अब हमेशा single continuous pass चलता है, model की बनाई हुई
    # आवाज़ (vocals) पूरी और video के साथ synced होती है — इसलिए पहले वही लगाते हैं।
    # अगर किसी वजह से वह न मिले, तो fallback में original song (trim करके) लगाते हैं।
    for _mux_fn in (_mux_model_vocals, _mux_original_song):
        if not muxed:
            muxed = _mux_fn()

    # Preference 3: no audio — just publish the video.
    if not muxed:
        shutil.copyfile(raw_video, final_path)
        print("  ⚠️ No audio muxed; published video-only.")

    del audio_dict
    purge_deep("post_mux")
    print(f"  🎉 Master MP4: {final_path}")
    return final_path


print("✅ Cell 15: Phase D (final audio mux) ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 16: OUTPUT VERIFICATION
# 🇮🇳 CELL 16 का काम: बनी हुई MP4 को ffprobe से जाँचना — file खाली तो नहीं,
#   video/audio stream, size और frame count सही हैं या नहीं।
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
# 🇮🇳 CELL 17 का काम: सब कुछ एक साथ चलाना (main)। यह Phase A→B→C→D क्रम से चलाता है।
#   • guard_stale_cache() → settings बदलने पर पुराने cache अपने-आप साफ़ करता है।
#   • output यहाँ बनता है: /content/LTXStudio_Output/LTX23_Director_Master_30s.mp4
#   • __main__ ब्लॉक पूरी generation शुरू कर देता है।
# ════════════════════════════════════════════════════════════════════════════
WORK_DIRECTORY = os.path.join(CONTENT_ROOT, "LTXDirector_Work")
OUTPUT_DIRECTORY = os.path.join(CONTENT_ROOT, "LTXStudio_Output")
SONG_PATH = os.path.join(WHATDREAMS_INPUT, "Late night trap.mp3")

# ── नई मदद: config जाँच + OOM पर resolution घटाकर retry ─────────────────────
MAX_OOM_RETRIES = 3   # 🇮🇳 OOM पर कितनी बार resolution घटाकर दोबारा कोशिश करें


def validate_config(meta: Dict[str, Any], segments: List[Dict[str, Any]],
                    use_song_audio: bool, song_path: str) -> bool:
    """🇮🇳 चलाने से पहले settings जाँचता है (गलत resolution/fps/frames/keyframe/song)।
    सिर्फ़ चेतावनी देता है ताकि 8 मिनट बर्बाद होने से पहले पता चल जाए।"""
    print("\n🔍 CONFIG जाँच...")
    problems = []
    db = int(meta.get("divisible_by", 32))
    for k in ("generation_width", "generation_height", "custom_width", "custom_height"):
        v = int(meta.get(k, 0))
        if v <= 0:
            problems.append(f"{k} गलत है ({v})")
        elif v % db:
            problems.append(f"{k}={v} · {db} से divisible नहीं")
    if int(meta.get("normalDurationFrames", 0)) < 9:
        problems.append("timeline बहुत छोटी (frames < 9)")
    if float(meta.get("frame_rate", 0)) <= 0:
        problems.append("frame_rate गलत है")
    if not segments:
        problems.append("कोई keyframe segment नहीं है")
    if use_song_audio and song_path and not os.path.exists(song_path):
        log(f"song file नहीं मिली: {song_path} — voice/lip-sync नहीं आएगा।", "WARN")
    if problems:
        log("Config में ये दिक्कतें दिखीं:", "WARN")
        for p in problems:
            log(f"   • {p}", "WARN")
        return False
    print("  ✓ Config ठीक है (resolution/fps/frames सही)।")
    return True


def _scale_meta_resolution(meta: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """🇮🇳 meta की सभी resolution values को `scale` से घटाता है (2x वाला रिश्ता बना
    रहता है), divisible_by पर snap करके। OOM retry ladder इसका इस्तेमाल करता है।
    सिर्फ़ spatial घटता है — frames/timing वही रहते हैं, इसलिए continuity बरकरार।"""
    db = max(1, int(meta.get("divisible_by", 32)))

    def _snap(n):
        return max(db, int(round(float(n) * scale / db)) * db)

    m = dict(meta)
    for k in ("generation_width", "generation_height", "custom_width", "custom_height",
              "base_stage1_width", "base_stage1_height"):
        if m.get(k):
            m[k] = _snap(m[k])
    return m


def _config_signature(meta: Dict[str, Any]) -> str:
    """Signature of every setting that changes the cached latents. If it differs
    from the last run, the old checkpoints are STALE (e.g. built at 1280x720) and
    must be cleared — otherwise resume would reuse old-resolution latents and the
    render-resolution speed fix would be silently ignored."""
    # PIPELINE_REV: bump whenever the diffusion math changes so old final_latents
    # are auto-regenerated (e.g. the overlap-continuity change → Phase B must rerun).
    # 🇮🇳 HINDI: यह सभी settings का एक "fingerprint" बनाता है। अगर settings बदलें
    # तो fingerprint बदल जाता है और पुराने cached latents अपने-आप delete हो जाते हैं
    # (ताकि गलत/पुराने resolution के latents दोबारा इस्तेमाल न हों)।
    pipeline_rev = "v4-singlepass-only"   # batch mode हटाने के बाद bump किया
    return (f"rev={pipeline_rev}"
            f"|render{meta.get('custom_width')}x{meta.get('custom_height')}"
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
    # 🇮🇳 HINDI: यह MAIN function है — पूरी video एक click में बनाता है।
    # क्रम: (0) nodes जाँचो + memory patches लगाओ + पुराना cache साफ़ करो →
    #       Phase A (timeline) → Phase B (video बनाओ, single pass) →
    #       Phase C (decode) → Phase D (audio जोड़ो) → verify → path लौटाओ।
    t_start = time.time()
    print("\n" + "=" * 70 + f"\n🎬 LTX-{ACTIVE_FAMILY} DIRECTOR 2.0 · Master_V6 (family-selectable)\n" + "=" * 70)
    validate_original_nodes()
    patch_comfy_memory_manager()
    patch_safetensors_direct_to_gpu()
    configure_vram_state(mode=globals().get("VRAM_MODE", "auto"))
    install_sampling_memory_hook(clear_every=4, ram_guard_gb=globals().get("min_ram_guard_gb", 1.5))
    os.makedirs(workdir, exist_ok=True)

    # (0) चलाने से पहले config जाँच (जल्दी पता चल जाए अगर कुछ गलत है)।
    validate_config(meta, segments, use_song_audio, SONG_PATH if use_song_audio else "")

    base_meta = dict(meta)          # मूल resolution (retry में इससे घटाते हैं)
    active_meta = dict(meta)
    latent_file = os.path.join(workdir, "final_latents.pt")

    # ── OOM RETRY LADDER ────────────────────────────────────────────────────
    # 🇮🇳 Phase A+B को एक loop में चलाते हैं। अगर CUDA OOM आए तो resolution ~20%
    # घटाकर (frames/timing वही रखते हुए → continuity बनी रहती है) अपने-आप दोबारा
    # कोशिश करते हैं। MAX_OOM_RETRIES बार के बाद साफ़ guidance देकर रुकते हैं।
    attempt = 0
    while True:
        # resolution बदलने पर guard_stale_cache पुराने latents अपने-आप साफ़ कर देगा।
        guard_stale_cache(workdir, active_meta)

        ctrl = DirectorTimelineController(
            global_prompt=global_prompt, negative_prompt=negative_prompt,
            meta=active_meta, segments=segments,
            audio_segments=audio_segments, motion_segments=motion_segments)

        if resume and os.path.exists(latent_file) and os.path.getsize(latent_file) > 1024:
            print(f"  ⏭ [RESUME] Phase A+B already done → using {latent_file}")
            break
        try:
            director_state, patched_model = execute_phase_a(ctrl, workdir=workdir, resume=resume)
            latent_file = execute_phase_b(director_state, patched_model, seed=seed,
                                          workdir=workdir, resume=resume)
            del director_state, patched_model
            purge_deep("post_phase_b_master")
            break                       # ✅ हो गया
        except TextEncoderOOM as e:
            # 🇮🇳 encoder OOM — resolution retry बेकार है (यही LTX-2.5 12B encoder की
            # असली दिक्कत है)। इसलिए तुरंत साफ़ guidance देकर रुक जाते हैं।
            purge_deep("text_encoder_oom_abort")
            raise
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise               # OOM नहीं है → असली error, आगे भेजो
            attempt += 1
            purge_deep(f"oom_retry_{attempt}")
            if attempt > MAX_OOM_RETRIES:
                raise RuntimeError(
                    "CUDA OOM: resolution घटाने के बाद भी यह GPU पूरी continuous timeline "
                    "नहीं बना पाया।\n"
                    "  👉 render_seconds और घटाएँ, या L4/A100 runtime इस्तेमाल करें।"
                ) from e
            active_meta = _scale_meta_resolution(base_meta, 0.8 ** attempt)
            log(f"OOM #{attempt}: resolution घटाकर दोबारा कोशिश → "
                f"{active_meta['generation_width']}x{active_meta['generation_height']} "
                f"(pass CONTINUOUS ही रहेगा, quality सही)।", "WARN")

    raw_video, audio_file = execute_phase_c(
        latent_file, workdir=workdir,
        fps=int(active_meta["frame_rate"]), crf=crf, resume=resume)

    final_video = execute_phase_d(
        raw_video, audio_file,
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


# 🇮🇳 चलाने से पहले 5.3.png alias बना दें (5.png से) ताकि keyframe मिल जाए।
_base_input = WHATDREAMS_INPUT
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



# ════════════════════════════════════════════════════════════════════════════
# CELL 18: QUALITY SELF-CHECK  (thumbnails + objective diagnostics)
# 🇮🇳 CELL 18 का काम: quality की जाँच — कुछ thumbnails निकालता है, black/frozen
#   frames पकड़ता है, और video-vs-audio की लंबाई मिलाकर lip-sync drift check करता है।
# ════════════════════════════════════════════════════════════════════════════
# You (or the assistant, if you share the JPEGs) can eyeball the thumbnails to
# judge identity / distortion / lighting; the metrics flag obvious failures
# (black or frozen frames, video/audio duration mismatch = lip-sync drift).
def quality_self_check(video_path: str, outdir: str = "/content/LTXStudio_Output",
                       n_thumbs: int = 6) -> Dict[str, Any]:
    print("\n" + "=" * 70 + "\n🔎 QUALITY SELF-CHECK\n" + "=" * 70)
    report: Dict[str, Any] = {"video": video_path}
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        print(f"  ❌ Output missing/empty: {video_path}")
        return report

    def _probe(stream, entries):
        cmd = (f'ffprobe -v error -select_streams {stream} '
               f'-show_entries {entries} -of default=nw=1 "{video_path}"')
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    vinfo = _probe("v:0", "stream=width,height,r_frame_rate,nb_read_packets,duration")
    ainfo = _probe("a:0", "stream=codec_name,duration,sample_rate,channels")
    print("  🎞️  Video:", vinfo.replace("\n", " ") or "(none)")
    print("  🔊 Audio:", ainfo.replace("\n", " ") or "(none — video only!)")

    # Video vs audio duration → lip-sync drift check.
    def _get(info, key):
        for line in info.splitlines():
            if line.startswith(key + "="):
                try:
                    return float(line.split("=", 1)[1])
                except Exception:
                    return None
        return None
    vd, ad = _get(vinfo, "duration"), _get(ainfo, "duration")
    if vd and ad:
        drift = abs(vd - ad)
        flag = "✅ OK" if drift < 0.5 else "⚠️ possible lip-sync drift"
        print(f"  ⏱️  Durations: video {vd:.2f}s | audio {ad:.2f}s | Δ {drift:.2f}s  {flag}")

    # Extract evenly-spaced thumbnails for visual inspection.
    thumb_dir = os.path.join(outdir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    for f in glob.glob(os.path.join(thumb_dir, "*.jpg")):
        try:
            os.remove(f)
        except Exception:
            pass
    dur = vd or 30.0
    thumbs = []
    for i in range(n_thumbs):
        t = max(0.1, dur * (i + 0.5) / n_thumbs)
        out = os.path.join(thumb_dir, f"thumb_{i:02d}.jpg")
        run_cmd(f'ffmpeg -y -ss {t:.2f} -i "{video_path}" -frames:v 1 -q:v 2 "{out}"')
        if os.path.exists(out):
            thumbs.append(out)
    report["thumbnails"] = thumbs
    print(f"  🖼️  Saved {len(thumbs)} thumbnails → {thumb_dir}")

    # Objective black / frozen-frame detection via numpy on the thumbnails.
    try:
        from PIL import Image as _Im
        means, prev, frozen = [], None, 0
        for tpath in thumbs:
            a = np.asarray(_Im.open(tpath).convert("L"), dtype=np.float32)
            means.append(a.mean())
            if prev is not None and np.abs(a - prev).mean() < 2.0:
                frozen += 1
            prev = a
        if means:
            dark = sum(1 for m in means if m < 8)
            print(f"  💡 Brightness (0-255) across thumbs: "
                  f"min {min(means):.0f} / avg {sum(means)/len(means):.0f} / max {max(means):.0f}")
            if dark:
                print(f"  ⚠️ {dark}/{len(means)} thumbnails look almost BLACK — check the decode/keyframes.")
            if frozen:
                print(f"  ⚠️ {frozen} near-identical consecutive thumbnails — motion may be too static.")
            if not dark and not frozen:
                print("  ✅ Frames are bright and changing (no obvious black/frozen failure).")
    except Exception as e:
        print(f"  [notice] frame analysis skipped ({e}).")

    # Try to preview inline if running inside a notebook.
    try:
        from IPython.display import Image as _IPyImage, display
        for t in thumbs:
            display(_IPyImage(filename=t, width=320))
    except Exception:
        pass

    print("=" * 70)
    print(f"  👉 Download the thumbnails from: {thumb_dir}")
    print("     Share one here and I can visually assess identity / distortion / quality.")
    print("=" * 70 + "\n")
    return report


# Auto-run the self-check if a master video already exists.
_final_mp4 = os.path.join(OUTPUT_DIRECTORY, "LTX23_Director_Master_30s.mp4")
if os.path.exists(_final_mp4):
    quality_self_check(_final_mp4, outdir=OUTPUT_DIRECTORY)
else:
    print("ℹ️ Cell 18 ready: run the generation, then call "
          "quality_self_check('/content/LTXStudio_Output/LTX23_Director_Master_30s.mp4').")
