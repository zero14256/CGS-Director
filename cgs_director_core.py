import gc
import json
import re
import cv2
import torch
import torchaudio
import folder_paths
import nodes

from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3AddGuide,
)

from . import cgs_presets as P

MODE_ALIAS = {
    "text": "T2VA文生音视频", "t2v": "T2VA文生音视频",
    "image": "FL2VA首尾帧", "i2v": "FL2VA首尾帧", "fl2v": "FL2VA首尾帧",
    "ref": "Ref2VA参考生成", "r2v": "Ref2VA参考生成", "ref2va": "Ref2VA参考生成",
    "all": "全能参考", "universal": "全能参考", "multi": "全能参考",
}


def _unpack_node_output(out):
    if hasattr(out, "args"):
        a = out.args
        if a:
            return a
    if isinstance(out, (tuple, list)):
        return tuple(out)
    raise RuntimeError("无法解析节点返回值，类型: %r" % type(out))



def _split_batch(tensor):
    if tensor is None:
        return []
    return [tensor[i:i + 1] for i in range(tensor.shape[0])]


def _load_image(name):
    img, _ = nodes.LoadImage().load_image(name)
    return img


def _load_video(name):
    path = folder_paths.get_annotated_filepath(name)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频: %s" % name)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).float() / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError("视频无帧: %s" % name)
    return torch.stack(frames)


def _load_audio(name):
    path = folder_paths.get_annotated_filepath(name)
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    return {"waveform": wav, "sample_rate": int(sr)}


def _load_video_audio(video_name):
    """提取视频文件的音轨为音频参考 dict（ffmpeg → wav → torchaudio）。无音轨时抛错。"""
    import subprocess, tempfile, os
    path = folder_paths.get_annotated_filepath(video_name)
    if not os.path.exists(path):
        raise RuntimeError("视频不存在: %s" % path)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        r = subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "32000", tmp],
                           capture_output=True, timeout=90)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise RuntimeError("ffmpeg 提取音轨失败")
        wav, sr = torchaudio.load(tmp)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        return {"waveform": wav, "sample_rate": int(sr)}
    finally:
        try: os.unlink(tmp)
        except Exception: pass


def _build_pool(items, loader):
    pool = []
    for n in (items or []):
        try:
            pool.append(loader(n))
        except Exception as e:
            print("[CGS-导演台] 素材加载失败 %s: %s" % (n, e))
    return pool


def _build_ref(pool, idx_list, strength_list, prefix, maxn, tag_word):
    d = {}
    tags = []
    used = 0
    for k, raw in enumerate(idx_list):
        try:
            idx = int(raw)
        except Exception:
            continue
        if idx < 0 or idx >= len(pool):
            continue
        st = strength_list[k] if (strength_list and k < len(strength_list)) else 1.0
        if st is not None and st <= 0:
            continue
        d[prefix + str(used)] = pool[idx]
        tags.append("<%s %d>" % (tag_word, used + 1))
        used += 1
        if used >= maxn:
            break
    return d, tags


def _parse_prompt_refs(prompt):
    """Parse @Image N, @Video N, @Audio N tags from prompt text.
    Returns (clean_prompt, image_indices, video_indices, audio_indices).
    Indices are 0-based (tag @Image 1 → index 0)."""
    image_idx = []
    video_idx = []
    audio_idx = []
    def _repl(m):
        kind = m.group(1).lower()
        n = int(m.group(2)) - 1
        if kind == "image":
            image_idx.append(n)
        elif kind == "video":
            video_idx.append(n)
        elif kind == "audio":
            audio_idx.append(n)
        return ""
    clean = re.sub(r"@(Image|Video|Audio)\s+(\d+)", _repl, prompt, flags=re.IGNORECASE)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean, image_idx, video_idx, audio_idx


def _int_list(v):
    out = []
    for x in (v or []):
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


def _parse_shots(shots_json, fallback_prompt, default_mode, duration, camera_preset, style_preset, seed):
    if shots_json and shots_json.strip():
        try:
            data = json.loads(shots_json)
        except Exception as e:
            raise RuntimeError("分镜 JSON 解析失败: %s" % e)
        if isinstance(data, dict) and isinstance(data.get("shots"), list):
            raw = data["shots"]
        elif isinstance(data, list):
            raw = data
        else:
            raw = None
        if raw:
            shots = []
            for i, s in enumerate(raw):
                if not isinstance(s, dict):
                    continue
                smode = s.get("mode", default_mode)
                if isinstance(smode, str):
                    smode = MODE_ALIAS.get(smode.lower(), smode)
                shots.append({
                    "id": s.get("id", "shot_%d" % i),
                    "name": s.get("name", "镜头 %d" % (i + 1)),
                    "enable": bool(s.get("enable", True)),
                    "mode": smode,
                    "prompt": str(s.get("prompt", "")),
                    "ref_image_idx": _int_list(s.get("ref_image_idx")),
                    "ref_video_idx": _int_list(s.get("ref_video_idx")),
                    "ref_audio_idx": _int_list(s.get("ref_audio_idx")),
                    "negative_prompt": str(s.get("negative_prompt", "") or ""),
                    "motion_prompt": str(s.get("motion_prompt", "") or ""),
                    "motion_strength": float(s.get("motion_strength", 0.5) or 0.5),
                    "subtitles": s.get("subtitles", ""),
                    "duration": float(s.get("duration") or duration),
                    "camera": s.get("camera") or camera_preset,
                    "style": s.get("style") or style_preset,
                    "seed": int(s.get("seed") or (42 + i * 1000)),
                })
            return shots
    return [{
        "id": "shot_0", "name": "镜头 1", "enable": True, "mode": default_mode,
        "prompt": fallback_prompt or "", "negative_prompt": "", "motion_prompt": "",
        "motion_strength": 0.5, "subtitles": "", "duration": duration,
        "camera": camera_preset, "style": style_preset, "seed": seed,
    }]


def _motion_adjective(strength):
    if strength >= 0.7:
        return "strong "
    if strength <= 0.3:
        return "subtle "
    return ""



def _process_dialogue_tags(prompt):
    """处理 <d> 台词标签：把台词从画面描述中提取出来，
    用明确英文指令分隔画面描述和台词，避免 MiniMax H3 把画面描述也当台词念。"""
    if not prompt or "<d>" not in prompt:
        return prompt
    d_pattern = r"<d>\[([^\]]+)\]([\s\S]*?)</d>"
    matches = re.findall(d_pattern, prompt)
    if not matches:
        return prompt
    # 从原提示词中移除 <d> 标签，保留画面描述
    clean = re.sub(d_pattern, " ", prompt)
    # 清理标签移除后留下的多余空格和重复标点
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"([，,、。.])\s*[，,、。.]+", r"\1", clean)
    clean = re.sub(r"[，,、。.\s]+$", "", clean).strip()
    clean = re.sub(r"^[，,、。.\s]+", "", clean).strip()
    # 构建台词部分（多个台词用 S1/S2/... 区分，台词末尾自动补句号）
    dlg_parts = []
    for i, (lang, text) in enumerate(matches):
        sid = "S%d" % (i + 1)
        t = text.strip()
        # 台词末尾没有结束标点时自动补句号
        if t and not re.search(r'[。.!?！？…]$', t):
            t = t + "。"
        dlg_parts.append("Character (%s) says: <d>[%s]%s</d>" % (sid, lang, t))
    dlg_text = " ".join(dlg_parts)
    # 用明确英文指令分隔画面描述和台词（模型对英文指令更敏感）
    if clean:
        return "Visual description only, do not speak this part: %s. Dialogue, speak ONLY the text inside <d> tags: %s" % (clean, dlg_text)
    return "Dialogue, speak ONLY the text inside <d> tags: %s" % dlg_text


def _build_prompt(shot, ref_tags):
    parts = []
    style = P.STYLE_PRESETS.get(shot["style"], "")
    if style:
        parts.append(style)
    if shot["prompt"]:
        parts.append(_process_dialogue_tags(shot["prompt"]))
    cam = P.CAMERA_PRESETS.get(shot["camera"], "")
    if cam:
        parts.append(cam)
    if shot["motion_prompt"]:
        parts.append(_motion_adjective(shot["motion_strength"]) + shot["motion_prompt"])
    prompt = " ".join(p for p in parts if p)
    if ref_tags:
        prompt = (prompt + " " + " ".join(ref_tags)) if prompt else " ".join(ref_tags)
    # 有参考音频时，自动添加音色参考说明（告诉模型参考音频只用于音色克隆，不复制原内容）
    aud_idx = shot.get("ref_audio_idx") or []
    if aud_idx:
        audio_refs = []
        for i, _ in enumerate(aud_idx):
            audio_refs.append("<Audio %d>: reference（音色参考，仅用于克隆说话人的音色和语气，不复制原始音频的内容和台词）" % (i + 1))
        prompt = (prompt + " " + " ".join(audio_refs)) if prompt else " ".join(audio_refs)
    return prompt


class CGSDirectorCore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "连接经 MiniMaxH3 SigmaShift 前的 H3 模型(UNETLoader 加载的 fl2va/ref2va 权重)"}),
                "clip": ("CLIP", {"tooltip": "连接 CLIPLoader(type=minimax / qwen3vl) 加载的 H3 文本编码器"}),
                "video_vae": ("VAE", {"tooltip": "连接 H3 视频 VAE(minimax_h3_video_vae)"}),
                "audio_vae": ("VAE", {"tooltip": "连接 H3 音频 VAE(minimax_h3_audio_vae)"}),
                "shots_json": ("STRING", {"multiline": True, "default": "", "tooltip": "可视化时间线写入的分镜+素材池+全局设置 JSON"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("模型", "正向条件", "负向条件", "潜空间", "采样参数JSON", "宽度", "高度", "帧数", "运行报告", "时间线JSON")
    FUNCTION = "execute"
    CATEGORY = "CGS导演台"

    def execute(self, model, clip, video_vae, audio_vae, shots_json):

        if model is None:
            raise RuntimeError("CGS-导演台：未连接模型(model)，请连接 H3 模型加载节点。")
        if clip is None:
            raise RuntimeError("CGS-导演台：未连接 CLIP，请连接 CLIPLoader(type=minimax)。")
        if video_vae is None:
            raise RuntimeError("CGS-导演台：未连接视频 VAE。")

        global_settings = {}
        materials = {}
        if shots_json and shots_json.strip():
            try:
                data = json.loads(shots_json)
                if isinstance(data, dict):
                    global_settings = data.get("global", {}) or {}
                    materials = data.get("materials", {}) or {}
            except Exception:
                pass

        g_resolution = global_settings.get("resolution_preset", "1080P 横屏 16:9") or "1080P 横屏 16:9"
        g_megapixels = global_settings.get("megapixels_preset", "无(使用上方分辨率)") or "无(使用上方分辨率)"
        g_quality = global_settings.get("quality_preset", "标准") or "标准"
        g_seed = int(global_settings.get("seed", 0) or 0)  # 全局种子已弃用，仅作回退默认值
        g_negative_prompt = str(global_settings.get("negative_prompt", "") or "")
        g_mode = "T2VA文生音视频"  # 全局生成模式已移除，固定默认值，每个镜头独立设置模式
        g_width = int(global_settings.get("width", 1920) or 1920)
        g_height = int(global_settings.get("height", 1080) or 1080)
        g_duration = float(global_settings.get("duration", 5.0) if global_settings.get("duration") is not None else 5.0)
        g_camera = global_settings.get("camera_preset", "无运镜") or "无运镜"
        g_style = global_settings.get("style_preset", "默认") or "默认"

        has_ref = bool(materials.get("images") or materials.get("videos") or materials.get("audios"))
        if has_ref and audio_vae is None:
            raise RuntimeError("CGS-导演台：使用了素材池参考但缺少音频 VAE，请连接 minimax_h3_audio_vae。")

        image_pool = _build_pool(materials.get("images"), _load_image)
        video_pool = _build_pool(materials.get("videos"), _load_video)
        audio_pool = _build_pool(materials.get("audios"), _load_audio)

        shots = _parse_shots(shots_json, "", g_mode, g_duration, g_camera, g_style, g_seed)
        
        # 衔接模式：将 seamless 数据转换为常规 shot
        _view_mode = "normal"
        _seamless = None
        try:
            _parsed = json.loads(shots_json) if isinstance(shots_json, str) else shots_json
            if isinstance(_parsed, dict):
                _view_mode = _parsed.get("viewMode", "normal") or "normal"
                _seamless = _parsed.get("seamless")
        except Exception:
            pass
        
        if _view_mode == "seamless" and _seamless and isinstance(_seamless, dict):
            print("[CGS-Seamless] 检测到衔接模式，转换为常规生成...")
            sm_sel = _seamless.get("selection", {"start": 2.0, "duration": 3.0})
            sm_start = float(sm_sel.get("start", 2.0) or 2.0)
            sm_dur = float(sm_sel.get("duration", 3.0) or 3.0)
            sm_end = sm_start + sm_dur
            sm_prompt = str(_seamless.get("prompt", "") or "")
            sm_gen_mode = str(_seamless.get("gen_mode", "fl2va") or "fl2va")
            sm_segments = _seamless.get("segments") or []
            sm_videos = materials.get("videos") or []
            print("[CGS-Seamless] 选区=%.2f-%.2fs 时长=%.2fs 片段数=%d 模式=%s" % (sm_start, sm_end, sm_dur, len(sm_segments), sm_gen_mode))

            def _cgs_extract_frame(video_name, timestamp):
                try:
                    import subprocess, tempfile, os
                    from PIL import Image
                    import numpy as np
                    import folder_paths
                    input_dir = folder_paths.get_input_directory()
                    video_path = os.path.join(input_dir, video_name)
                    if not os.path.exists(video_path):
                        print("[CGS-Seamless] 视频不存在: %s" % video_path); return None
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        tmp_path = f.name
                    subprocess.run(["ffmpeg", "-y", "-ss", str(max(0, timestamp)), "-i", video_path, "-vframes", "1", "-q:v", "2", tmp_path], capture_output=True, timeout=30)
                    if os.path.exists(tmp_path):
                        img = Image.open(tmp_path).convert("RGB")
                        tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                        os.unlink(tmp_path); return tensor
                except Exception as e:
                    print("[CGS-Seamless] 帧提取失败: %s" % str(e))
                return None

            def _cgs_get_vname(vidx):
                if sm_videos and 0 <= vidx < len(sm_videos):
                    return sm_videos[vidx]
                return None

            def _cgs_source_time(seg, timeline_second):
                # 时间轴秒数 -> 原始视频时间 = trim_start + (timeline_second - seg.start)
                return float(seg.get("trim_start", 0) or 0) + (float(timeline_second) - float(seg["start"]))

            # ===== 完全参考 MiniMaxH3-TimelineDirector 的 _pick_gap_guides 逻辑 =====
            CGS_FPS = 24.0
            cgs_epsilon = 1.0 / CGS_FPS
            sm_first_frame = None; sm_last_frame = None; sm_ref_video_idx = None; sm_ref_vidxs = []

            # 解析有效片段（参考插件：过滤有 file 的 clip）
            valid_segs = []
            for seg in sm_segments:
                if not isinstance(seg, dict): continue
                s_start = float(seg.get("start", 0) or 0)
                s_end = float(seg.get("end", 0) or 0)
                s_vidx = int(seg.get("video_idx", 0) or 0)
                s_trim = float(seg.get("trim_start", 0) or 0)
                if s_end > s_start and s_vidx >= 0 and s_vidx < len(sm_videos):
                    valid_segs.append({"start": s_start, "end": s_end, "video_idx": s_vidx, "trim_start": s_trim})

            for _i, _s in enumerate(valid_segs):
                print("[CGS-Seamless] 片段%d: 时间轴=%.3f-%.3fs 时长=%.3fs 视频idx=%d trim=%.3fs" % (_i, _s["start"], _s["end"], _s["end"]-_s["start"], _s["video_idx"], _s["trim_start"]))

            # 参考插件：按 start 排序
            ordered = sorted(valid_segs, key=lambda s: (s["start"], s["end"]))

            # ===== 参考范围由选区决定：覆盖多少参考多少 =====
            _ref_trims = []
            _trim_len = min(1.5, sm_dur)
            left_cands = [c for c in ordered if c["start"] < sm_start]
            right_cands = [c for c in ordered if c["end"] > sm_end]
            print("[CGS-Seamless] 左右候选: 左=%d 右=%d" % (len(left_cands), len(right_cands)))

            if left_cands:
                left = max(left_cands, key=lambda c: c["end"])
                left_tl = min(left["end"], sm_start)  # 衔接点=过渡起点(选区覆盖时用选区起点，否则用左段末尾)
                left_src = _cgs_source_time(left, left_tl)
                vn = _cgs_get_vname(left["video_idx"])
                print("[CGS-Seamless] 左片段(衔接点): 时间轴=%.3fs 原始视频=%.3fs 视频=%s" % (left_tl, left_src, vn))
                sm_first_frame = _cgs_extract_frame(vn, left_src)
                sm_ref_video_idx = left["video_idx"]
                if left["video_idx"] not in sm_ref_vidxs: sm_ref_vidxs.append(left["video_idx"])
                # 参考范围：选区覆盖前段的部分（若覆盖），否则取前段最后 _trim_len
                if left["end"] > sm_start + cgs_epsilon:
                    _rs = _cgs_source_time(left, sm_start)
                    print("[CGS-Seamless] 选区覆盖前段: 从时间轴%.3fs开始" % sm_start)
                else:
                    _rs = _cgs_source_time(left, max(left["start"], left["end"] - _trim_len))
                _re = _cgs_source_time(left, left["end"])
                _ref_trims.append({"video_idx": left["video_idx"], "start": _rs, "end": _re})
                print("[CGS-Seamless] 前段参考: 视频%d %.2f-%.2fs" % (left["video_idx"], _rs, _re))

            if right_cands:
                right = min(right_cands, key=lambda c: c["start"])
                right_tl = max(right["start"], sm_end)  # 衔接点=过渡终点(选区覆盖时用选区终点，否则用右段开头)
                right_src = _cgs_source_time(right, right_tl)
                vn = _cgs_get_vname(right["video_idx"])
                print("[CGS-Seamless] 右片段(衔接点): 时间轴=%.3fs 原始视频=%.3fs 视频=%s" % (right_tl, right_src, vn))
                sm_last_frame = _cgs_extract_frame(vn, right_src)
                if sm_ref_video_idx is None:
                    sm_ref_video_idx = right["video_idx"]
                if right["video_idx"] not in sm_ref_vidxs: sm_ref_vidxs.append(right["video_idx"])
                # 参考范围：选区覆盖后段的部分（若覆盖），否则取后段前 _trim_len
                if right["start"] < sm_end - cgs_epsilon:
                    _re = _cgs_source_time(right, sm_end)
                    print("[CGS-Seamless] 选区覆盖后段: 到时间轴%.3fs" % sm_end)
                else:
                    _re = _cgs_source_time(right, min(right["end"], right["start"] + _trim_len))
                _rs = _cgs_source_time(right, right["start"])
                _ref_trims.append({"video_idx": right["video_idx"], "start": _rs, "end": _re})
                print("[CGS-Seamless] 后段参考: 视频%d %.2f-%.2fs" % (right["video_idx"], _rs, _re))

            print("[CGS-Seamless] 首尾帧: 首帧=%s 尾帧=%s" % ("OK" if sm_first_frame is not None else "FAIL", "OK" if sm_last_frame is not None else "FAIL"))

            # 回退：未找到左右片段时，从单个视频提取选区首尾帧
            if (sm_first_frame is None or sm_last_frame is None) and sm_videos:
                sm_ref_video_idx = 0
                vn = _cgs_get_vname(0)
                print("[CGS-Seamless] 回退: 从视频=%s 提取选区首尾帧(%.2fs, %.2fs)" % (vn, sm_start, sm_end))
                if sm_first_frame is None: sm_first_frame = _cgs_extract_frame(vn, sm_start)
                if sm_last_frame is None: sm_last_frame = _cgs_extract_frame(vn, sm_end)

            # 构建参考视频索引列表（去重，左先右后）和 @Video 标签
            cgs_ref_list = list(sm_ref_vidxs)
            cgs_ref_tags = " ".join(["@Video %d" % (i + 1) for i in range(len(cgs_ref_list))])
            print("[CGS-Seamless] 参考视频索引=%s 标签=%s" % (cgs_ref_list, cgs_ref_tags))

            # 将 _ref_trims 按 cgs_ref_list 顺序重排，保证与 ref_video_N 一一对应
            _trim_by_vidx = {t["video_idx"]: t for t in _ref_trims}
            _ref_trims = [_trim_by_vidx[_vi] for _vi in cgs_ref_list if _vi in _trim_by_vidx]

            _anchor_first = _seamless.get("anchor_first", True)
            if _anchor_first is None: _anchor_first = True
            _anchor_last = _seamless.get("anchor_last", True)
            if _anchor_last is None: _anchor_last = True
            if sm_gen_mode == "fl2va" and sm_first_frame is not None and sm_last_frame is not None:
                sm_shot = {"id":"seamless_shot_0","name":"衔接生成","enable":True,"mode":"FL2VA首尾帧","prompt":(cgs_ref_tags + " " + sm_prompt).strip(),"negative_prompt":g_negative_prompt,"motion_prompt":"","motion_strength":0.5,"duration":sm_dur,"camera":g_camera,"style":g_style,"seed":g_seed,"ref_image_idx":[],"ref_video_idx":cgs_ref_list,"ref_audio_idx":[],"_first_frame":sm_first_frame,"_last_frame":sm_last_frame,"_ref_trims":_ref_trims}
            else:
                sm_shot = {"id":"seamless_shot_0","name":"衔接生成","enable":True,"mode":"全能参考","prompt":(cgs_ref_tags + " " + sm_prompt).strip(),"negative_prompt":g_negative_prompt,"motion_prompt":"","motion_strength":0.5,"duration":sm_dur,"camera":g_camera,"style":g_style,"seed":g_seed,"ref_image_idx":[],"ref_video_idx":cgs_ref_list,"ref_audio_idx":[],"_ref_trims":_ref_trims}
                if sm_first_frame is not None: sm_shot["_first_frame"] = sm_first_frame
                if sm_last_frame is not None: sm_shot["_last_frame"] = sm_last_frame
            sm_shot["_anchor_first"] = _anchor_first
            sm_shot["_anchor_last"] = _anchor_last
            # ===== 音频参考：开关开启且参考视频有音轨 → 用视频音轨；否则 → 音频素材 =====
            _audio_from_video = _seamless.get("audio_from_video", True)
            if _audio_from_video is None: _audio_from_video = True
            _seam_audio_pool = []
            _seam_audio_idx = []
            if _audio_from_video and cgs_ref_list:
                for _vi in cgs_ref_list:
                    _vn = _cgs_get_vname(_vi)
                    if not _vn: continue
                    try:
                        _seam_audio_pool.append(_load_video_audio(_vn))
                        _seam_audio_idx.append(len(_seam_audio_pool) - 1)
                    except Exception as _e:
                        print("[CGS-Seamless] 视频音轨提取失败 %s: %s" % (_vn, _e))
            if not _seam_audio_idx:
                try:
                    _aui = _seamless.get("audio_idx")
                    if _aui is not None and 0 <= int(_aui) < len(materials.get("audios") or []):
                        _seam_audio_idx = [int(_aui)]
                except Exception:
                    pass
            sm_shot["ref_audio_idx"] = _seam_audio_idx
            sm_shot["_audio_pool"] = _seam_audio_pool if _seam_audio_pool else None
            print("[CGS-Seamless] 音频参考: 视频音轨=%d个 音频素材idx=%s" % (len(_seam_audio_pool), _seam_audio_idx))
            shots = [sm_shot]
            print("[CGS-Seamless] 衔接模式 shot 构建完成，时长=%.1fs，模式=%s，参考视频=%d个" % (sm_dur, sm_gen_mode, len(cgs_ref_list)))
        
        active = [s for s in shots if s["enable"]]
        print("[CGS-DEBUG] 全部镜头enable状态:", [(s.get("name","?"), s.get("enable", True)) for s in shots])
        print("[CGS-DEBUG] 启用镜头数=%d / 总镜头数=%d" % (len(active), len(shots)))
        if not active:
            raise RuntimeError("CGS-导演台：没有启用(enable=true)的镜头可生成。")

        print("[CGS-DEBUG] 分辨率预设=%s megapixels=%s 自定义宽高=%sx%s" % (g_resolution, g_megapixels, g_width, g_height))
        # 优先用 megapixels 预设，提取失败才回退分辨率预设
        mp_res = None
        if g_megapixels and g_megapixels != "无(使用上方分辨率)":
            mp_res = P.megapixels_from_string(g_megapixels)
        if mp_res:
            w, h = mp_res
        else:
            w, h = P.resolution_from_preset(g_resolution, g_width, g_height)
        print("[CGS-DEBUG] 最终分辨率=%sx%s (来源=%s)" % (w, h, "megapixels" if mp_res else "分辨率预设"))

        last_positive = None
        last_negative = None
        last_latent = None
        report_lines = []

        for idx, shot in enumerate(shots):
            if not shot["enable"]:
                report_lines.append("跳过: 镜头 %d (%s) 已禁用" % (idx + 1, shot["name"]))
                continue

            sdur = shot["duration"] or 5.0
            if sdur > 15.0:
                report_lines.append("警告: 镜头 %d 时长 %.1fs 超出训练范围(约15s)" % (idx + 1, sdur))
            frames = P.duration_to_frames(sdur)

            # 优先读前端剥离后的结构化索引；手写 JSON(无索引) 回退从 @标签解析
            img_idx = shot.get("ref_image_idx") or []
            vid_idx = shot.get("ref_video_idx") or []
            aud_idx = shot.get("ref_audio_idx") or []
            if img_idx or vid_idx or aud_idx:
                shot["prompt"] = re.sub(r"@(Image|Video|Audio|图片|视频|音频)\s+\d+", "", shot["prompt"], flags=re.IGNORECASE).strip()
            else:
                clean_prompt, img_idx, vid_idx, aud_idx = _parse_prompt_refs(shot["prompt"])
                shot["prompt"] = clean_prompt

            ref_images, t_im = _build_ref(image_pool, img_idx, [1.0] * len(img_idx), "ref_image_", 9, "Picture")
            ref_videos, t_vi = _build_ref(video_pool, vid_idx, None, "ref_video_", 3, "Video")
            _apool = shot.get("_audio_pool")
            if _apool is None:
                _apool = audio_pool
            ref_audios, t_au = _build_ref(_apool, aud_idx, None, "ref_audio_", 3, "Audio")
            all_tags = t_im + t_vi + t_au

            # ===== 参考视频处理：时间裁剪(选区决定)+帧采样+降分辨率 =====
            if ref_videos:
                _opt_scale = float(global_settings.get("opt_scale", 0.5) if global_settings.get("opt_scale") is not None else 0.5)
                _opt_max_frames = int(global_settings.get("opt_frames", 24) if global_settings.get("opt_frames") is not None else 24)
                _opt_w = max(128, int(w * _opt_scale))
                _opt_h = max(128, int(h * _opt_scale))
                _ref_trims = shot.get("_ref_trims") or []
                _opt_keys = sorted(ref_videos.keys())
                for _vi, _key in enumerate(_opt_keys):
                    _v = ref_videos[_key]
                    # 方案3：时间裁剪
                    if _ref_trims and _vi < len(_ref_trims):
                        _tr = _ref_trims[_vi]
                        _sf = max(0, int(_tr["start"] * CGS_FPS))
                        _ef = int(_tr["end"] * CGS_FPS)
                        if _ef > _sf and _ef <= _v.shape[0]:
                            _v = _v[_sf:_ef]
                    # 方案1：帧均匀采样
                    _t = _v.shape[0]
                    if _t > _opt_max_frames:
                        _idx = torch.linspace(0, _t - 1, _opt_max_frames).long()
                        _v = _v[_idx]
                    # 方案2：降分辨率
                    if _v.shape[1] != _opt_h or _v.shape[2] != _opt_w:
                        _vt = _v.permute(0, 3, 1, 2).float()
                        _vt = torch.nn.functional.interpolate(_vt, size=(_opt_h, _opt_w), mode='bilinear', align_corners=False)
                        _v = _vt.permute(0, 2, 3, 1)
                    ref_videos[_key] = _v
                print("[CGS-优化] 参考视频: %d段, 每段最多%d帧, 分辨率%dx%d" % (len(ref_videos), _opt_max_frames, _opt_w, _opt_h))

            final_prompt = _build_prompt(shot, all_tags)

            neg = []
            neg_parts = []
            if g_negative_prompt:
                neg_parts.append(g_negative_prompt)
            if shot["negative_prompt"]:
                neg_parts.append(shot["negative_prompt"])
            neg_text = ", ".join(neg_parts)
            if neg_text:
                try:
                    neg = clip.encode_from_tokens_scheduled(clip.tokenize(neg_text))
                except Exception:
                    neg = []

            shot_mode = shot.get("mode", "T2VA文生音视频")
            first_frame = None
            last_frame = None

            if shot_mode == "FL2VA首尾帧":
                # 衔接模式：优先使用直接提取的首尾帧
                if shot.get("_first_frame") is not None:
                    first_frame = shot["_first_frame"]
                if shot.get("_last_frame") is not None:
                    last_frame = shot["_last_frame"]
                # 常规模式：从参考图片中获取
                if first_frame is None or last_frame is None:
                    ref_vals = list(ref_images.values()) if ref_images else []
                    if len(ref_vals) >= 2:
                        if first_frame is None: first_frame = ref_vals[0]
                        if last_frame is None: last_frame = ref_vals[1]
                        ref_images = {}
                    elif len(ref_vals) == 1:
                        if first_frame is None: first_frame = ref_vals[0]
                        ref_images = {}
                out = MiniMaxH3ImageToVideo.execute(
                    clip, video_vae, final_prompt, w, h, frames,
                    first_frame=first_frame, last_frame=last_frame,
                )
            elif shot_mode in ("Ref2VA参考生成", "全能参考"):
                # 注意：ReferenceToVideo.execute 签名与 ImageToVideo 不同——
                # (clip, prompt, width, height, length, ref_image_size, vae, audio_vae, ...)，
                # vae/audio_vae 在末尾，须用关键字传，否则参数错位会把 prompt 当 height。
                out = MiniMaxH3ReferenceToVideo.execute(
                    clip, final_prompt, w, h, frames, "match",
                    vae=video_vae, audio_vae=audio_vae,
                    ref_images=ref_images or None,
                    ref_videos=ref_videos or None,
                    ref_video_audios=None,
                    ref_audios=ref_audios or None,
                )
            else:
                out = MiniMaxH3ImageToVideo.execute(
                    clip, video_vae, final_prompt, w, h, frames,
                    first_frame=first_frame, last_frame=None,
                )
            positive, latent = _unpack_node_output(out)

            # ===== 全能参考模式：AddGuide 锚定衔接首/末帧 =====
            # 解决"选区第一帧/末帧 != 生成首/末帧"：ReferenceToVideo 只提供运动参考、不锚定首尾画面；
            # AddGuide 在 conditioning 上追加 minimax_keyframes，把选区首帧画面钉在过渡段第0帧、
            # 选区末帧画面钉在过渡段最后1帧，保证首尾与素材无缝衔接。
            _do_first = shot_mode in ("Ref2VA参考生成", "全能参考") and shot.get("_anchor_first", True) and shot.get("_first_frame") is not None
            _do_last = shot_mode in ("Ref2VA参考生成", "全能参考") and shot.get("_anchor_last", True) and shot.get("_last_frame") is not None
            if _do_first or _do_last:
                try:
                    if _do_first:
                        positive = MiniMaxH3AddGuide.execute(
                            positive=positive,
                            latent=latent,
                            frame_idx=0,
                            image=shot["_first_frame"],
                            vae=video_vae,
                        ).args[0]
                        report_lines.append("镜头 %d: 首帧已锚定选区第一帧(AddGuide)" % (idx + 1))
                    if _do_last:
                        positive = MiniMaxH3AddGuide.execute(
                            positive=positive,
                            latent=latent,
                            frame_idx=-1,
                            image=shot["_last_frame"],
                            vae=video_vae,
                        ).args[0]
                        report_lines.append("镜头 %d: 末帧已锚定选区末帧(AddGuide)" % (idx + 1))
                except Exception as _ae:
                    report_lines.append("镜头 %d: 锚定跳过: %s" % (idx + 1, str(_ae)))

            last_positive = positive
            last_negative = neg
            last_latent = latent

            ref_desc = []
            if ref_images: ref_desc.append("图%d" % len(ref_images))
            if ref_videos: ref_desc.append("视频%d" % len(ref_videos))
            if ref_audios: ref_desc.append("音频%d" % len(ref_audios))
            ref_str = (" 参考[" + ",".join(ref_desc) + "]") if ref_desc else "无参考"
            conn = " 首帧衔接" if first_frame is not None else ""
            sub = " 含字幕(需后期烧录)" if shot.get("subtitles") else ""
            report_lines.append("镜头 %d(%s): 模式=%s 分辨率=%dx%d 帧=%d%s%s%s" % (
                idx + 1, shot["name"], shot["mode"], w, h, frames, ref_str, conn, sub))

        total_frames = last_latent["samples"].shape[2] if last_latent and "samples" in last_latent else 0
        # 采样种子跟随第一个启用镜头的设置，不再使用全局种子
        _sampler_seed = int(active[0]["seed"]) if active else g_seed
        sampler_params = json.dumps({"seed": _sampler_seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "karras", "shift_video": 12.0, "shift_audio": 8.0})
        report_lines.append("总计: %d 个启用镜头, %d 帧, 约 %.1fs" % (
            len(active), total_frames, total_frames / P.FPS))
        return (model, last_positive, last_negative, last_latent, sampler_params, w, h, total_frames, "\n".join(report_lines), shots_json)

