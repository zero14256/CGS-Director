# ============================================================
# CGS-导演台 (CGS-Director) · ComfyUI MiniMax H3 导演工作台
# Author: zero14256
# Repo:   https://github.com/zero14256/comfyui-MinimaxH3-CGS-Director
# License: MIT · 请保留此来源标识
# ============================================================

import json
import os

base = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(base, "CGS-导演台-示例工作流.json")

nodes = []
links = []


def winput(name, type_, link, widget=None):
    d = {"name": name, "type": type_, "link": link}
    if widget:
        d["widget"] = {"name": widget}
    return d


nodes.append({
    "id": 1, "type": "UNETLoader", "pos": [80, 120],
    "inputs": [winput("unet_name", "COMBO", None, "unet_name"),
               winput("weight_dtype", "COMBO", None, "weight_dtype")],
    "widgets_values": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"],
})

nodes.append({
    "id": 2, "type": "CLIPLoader", "pos": [80, 280],
    "inputs": [winput("clip_name", "COMBO", None, "clip_name"),
               winput("type", "COMBO", None, "type"),
               winput("device", "COMBO", None, "device")],
    "widgets_values": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"],
})

nodes.append({
    "id": 3, "type": "VAELoader", "pos": [80, 460],
    "inputs": [winput("vae_name", "COMBO", None, "vae_name")],
    "widgets_values": ["minimax_h3_video_vae_fp16.safetensors"],
})

nodes.append({
    "id": 4, "type": "VAELoader", "pos": [80, 580],
    "inputs": [winput("vae_name", "COMBO", None, "vae_name")],
    "widgets_values": ["minimax_h3_audio_vae_fp32.safetensors"],
})

cgs_widgets = [
    "T2VA文生音视频",
    "",
    "",
    "1080P 横屏 16:9",
    1920,
    1080,
    5.0,
    "无运镜",
    "默认",
    "标准",
    "res_multistep",
    "native",
    25,
    1.0,
    12.0,
    3.0,
    0,
    1.0,
    "生成",
    True,
    True,
    False,
]
cgs_inputs = [
    winput("model", "MODEL", 1),
    winput("clip", "CLIP", 2),
    winput("video_vae", "VAE", 3),
    winput("audio_vae", "VAE", 4),
    winput("mode", "COMBO", None, "mode"),
    winput("prompt", "STRING", None, "prompt"),
    winput("shots_json", "STRING", None, "shots_json"),
    winput("resolution_preset", "COMBO", None, "resolution_preset"),
    winput("width", "INT", None, "width"),
    winput("height", "INT", None, "height"),
    winput("duration", "FLOAT", None, "duration"),
    winput("camera_preset", "COMBO", None, "camera_preset"),
    winput("style_preset", "COMBO", None, "style_preset"),
    winput("quality_preset", "COMBO", None, "quality_preset"),
    winput("sampler_name", "COMBO", None, "sampler_name"),
    winput("scheduler", "COMBO", None, "scheduler"),
    winput("steps", "INT", None, "steps"),
    winput("cfg", "FLOAT", None, "cfg"),
    winput("shift_video", "FLOAT", None, "shift_video"),
    winput("shift_audio", "FLOAT", None, "shift_audio"),
    winput("seed", "INT", None, "seed"),
    winput("denoise", "FLOAT", None, "denoise"),
    winput("audio_mode", "COMBO", None, "audio_mode"),
    winput("enable_long_video", "BOOLEAN", None, "enable_long_video"),
    winput("continuity", "BOOLEAN", None, "continuity"),
    winput("auto_ref_tags", "BOOLEAN", None, "auto_ref_tags"),
]
nodes.append({
    "id": 5, "type": "CGSDirectorCore", "pos": [420, 200],
    "inputs": cgs_inputs,
    "widgets_values": cgs_widgets,
})

nodes.append({
    "id": 6, "type": "CreateVideo", "pos": [760, 220],
    "inputs": [winput("images", "IMAGE", 5),
               winput("fps", "FLOAT", None, "fps"),
               winput("audio", "AUDIO", 6),
               winput("bit_depth", "INT", None, "bit_depth")],
    "widgets_values": [24.0, 8],
})

nodes.append({
    "id": 7, "type": "SaveVideo", "pos": [1040, 220],
    "inputs": [winput("video", "VIDEO", 8),
               winput("filename_prefix", "STRING", None, "filename_prefix"),
               winput("format", "COMBO", None, "format"),
               winput("codec", "COMBO", None, "codec")],
    "widgets_values": ["CGS-导演台出品", "auto", "auto"],
})

nodes.append({
    "id": 8, "type": "PreviewText", "pos": [760, 460],
    "inputs": [winput("text", "STRING", 9)],
    "widgets_values": [""],
})

links = [
    [1, 1, 0, 5, 0, "MODEL"],
    [2, 2, 0, 5, 1, "CLIP"],
    [3, 3, 0, 5, 2, "VAE"],
    [4, 4, 0, 5, 3, "VAE"],
    [5, 5, 0, 6, 0, "IMAGE"],
    [6, 5, 1, 6, 2, "AUDIO"],
    [8, 6, 0, 7, 0, "VIDEO"],
    [9, 5, 4, 8, 0, "STRING"],
]

workflow = {
    "last_node_id": 8,
    "last_link_id": 9,
    "nodes": nodes,
    "links": links,
    "groups": [],
    "config": {},
    "extra": {},
    "_source": "CGS-Director by zero14256 | https://github.com/zero14256/comfyui-MinimaxH3-CGS-Director",
    "version": 0.4,
}

with open(out, "w", encoding="utf-8") as f:
    json.dump(workflow, f, ensure_ascii=False, indent=2)

print("已生成:", out)
