# -*- coding: utf-8 -*-
# ============================================================
# CGS-导演台 (CGS-Director) · ComfyUI MiniMax H3 导演工作台
# Author: zero14256
# Repo:   https://github.com/zero14256/comfyui-MinimaxH3-CGS-Director
# License: MIT · 请保留此来源标识
# ============================================================

"""
CGS-SeamlessCompose 成片合成节点
输入：过渡段解码帧(VAEDecode输出) + CGS导演台的 shots_json(含 seamless 时间轴 + 素材池)
输出：按时间轴裁剪的 [上一段素材 + 过渡段 + 下一段素材] 拼接而成的完整成片帧

拼接质量优化：
- color_match  ：过渡段做 LAB 均值/对比度匹配，向左右段色调靠拢，消除光影/色调跳变
- junction_blend：拼接点交叉淡化帧数(0=关闭)，柔化衔接处的顿挫
"""
import json
import math
import os
import cv2
import torch
import torch.nn.functional as F
import folder_paths

from . import cgs_presets as P

CGS_FPS = float(P.FPS)


def _src_time(seg, tl_second):
    """时间轴秒数 -> 原始视频时间 = trim_start + (timeline_second - seg.start)"""
    return float(seg.get("trim_start", 0) or 0) + (float(tl_second) - float(seg["start"]))


def _extract_clip(path, t_start, t_end, fps, th, tw):
    """从素材视频按时间范围抽帧，对齐目标帧率/分辨率。

    关键改进：用【相邻帧线性插值】在源帧率与目标帧率之间做平滑重采样，
    替代原先的 round() 取帧——避免 30/25fps 素材映射到 24fps 时产生
    不规则的丢帧/重复，从而消除拼接段内部的运动跳变。
    返回 list[tensor H,W,C] 0-1
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开素材视频: %s" % path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = max(1, int(round(max(0.0, t_end - t_start) * fps)))
    frames = []
    _cache = {}

    def _norm(ft):
        if ft.shape[0] != th or ft.shape[1] != tw:
            ft = F.interpolate(ft.permute(2, 0, 1).unsqueeze(0), size=(th, tw),
                               mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
        return ft

    def _read(idx):
        idx = max(0, int(idx))
        if idx in _cache:
            return _cache[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            _cache[idx] = None
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ft = torch.from_numpy(frame).float() / 255.0
        _cache[idx] = ft
        return ft

    for i in range(n):
        t = t_start + i / fps
        pos = max(0.0, t * src_fps)
        idx0 = int(math.floor(pos))
        frac = pos - idx0
        f0 = _read(idx0)
        if f0 is None:
            break
        if frac < 1e-6:
            frames.append(_norm(f0))
        else:
            f1 = _read(idx0 + 1)
            if f1 is None:
                frames.append(_norm(f0))
            else:
                frames.append(_norm(f0 * (1.0 - frac) + f1 * frac))
    cap.release()
    return frames


# ================= 色彩匹配（LAB 空间均值/对比度对齐） =================
def _rgb2lab(t):
    """t: H,W,3 RGB(0-1) -> H,W,3 LAB"""
    def lin(c):
        return torch.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    r, g, b = lin(t[..., 0]), lin(t[..., 1]), lin(t[..., 2])
    x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.95047
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.08883
    d = 6.0 / 29.0

    def f(t):
        return torch.where(t > d ** 3, t ** (1.0 / 3.0), t / (3.0 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f(x), f(y), f(z)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-1)


def _lab2rgb(t):
    """H,W,3 LAB -> H,W,3 RGB(0-1)"""
    L, a, b = t[..., 0], t[..., 1], t[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    d = 6.0 / 29.0

    def finv(t):
        return torch.where(t > d, t ** 3, 3.0 * d * d * (t - 4.0 / 29.0))
    x = finv(fx) * 0.95047
    y = finv(fy)
    z = finv(fz) * 1.08883
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bb = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    def lin_inv(c):
        return torch.where(c > 0.0031308, 1.055 * c ** (1.0 / 2.4) - 0.055, 12.92 * c)
    return torch.stack([lin_inv(r), lin_inv(g), lin_inv(bb)], dim=-1).clamp(0, 1)


def _match_color_series(frames, ref_start, ref_end, strength=0.7):
    """对过渡段每帧做 LAB 均值/标准差对齐：前段向 ref_start(左段末帧)、后段向 ref_end(右段首帧)、中间线性插值。
    任一锚点缺失时退化为仅用可用锚点。返回新帧列表(不修改原列表)。"""
    n = len(frames)
    if n == 0 or strength <= 0:
        return frames
    refs = []
    if ref_start is not None:
        refs.append(_rgb2lab(ref_start))
    if ref_end is not None:
        refs.append(_rgb2lab(ref_end))
    if not refs:
        return frames
    if len(refs) == 1:
        m = refs[0].mean(dim=(0, 1))
        s = refs[0].std(dim=(0, 1)) + 1e-6
        ms, ss, me, se = m, s, m, s
    else:
        ms = refs[0].mean(dim=(0, 1)); ss = refs[0].std(dim=(0, 1)) + 1e-6
        me = refs[1].mean(dim=(0, 1)); se = refs[1].std(dim=(0, 1)) + 1e-6
    out = []
    for i, fr in enumerate(frames):
        w = i / max(1, n - 1)
        m = ms * (1 - w) + me * w
        s = ss * (1 - w) + se * w
        lab = _rgb2lab(fr)
        mu = lab.mean(dim=(0, 1), keepdim=True)
        sd = lab.std(dim=(0, 1), keepdim=True) + 1e-6
        adj = (lab - mu) / sd * s + m
        adj = lab + (adj - lab) * strength
        out.append(_lab2rgb(adj))
    return out


def _blend_junction(a_frames, b_frames, k):
    """拼接点交叉淡化：a 尾部 k 帧向 b 淡出、b 头部 k 帧从 a 淡入（帧数不变，内容渐变）"""
    k = max(1, min(k, len(a_frames), len(b_frames)))
    for i in range(k):
        w = (i + 1) / (k + 1)  # b 的权重 0.33...0.9
        wb = w * 0.6
        ai = a_frames[-(i + 1)]
        bi = b_frames[i]
        am = ai * (1 - wb) + bi * wb
        bm = ai * wb + bi * (1 - wb)
        a_frames[-(i + 1)] = am
        b_frames[i] = bm
    return a_frames, b_frames


class CGSSeamlessCompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transition_frames": ("IMAGE", {"tooltip": "过渡段解码帧(VAEDecode 输出)，其分辨率/帧率作为成片基准"}),
                "shots_json": ("STRING", {"multiline": True, "default": "", "tooltip": "CGS导演台输出的同一份 shots_json(含衔接模式时间轴+素材池)"}),
            },
            "optional": {
                "fps": ("INT", {"default": int(CGS_FPS), "min": 1, "max": 120, "tooltip": "成片帧率(默认24，与生成一致)"}),
                "color_match": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "过渡段色彩匹配强度：向左右段色调靠拢，消除光影/色调跳变(0=关闭)"}),
                "junction_blend": ("INT", {"default": 0, "min": 0, "max": 6, "step": 1,
                                           "tooltip": "拼接点交叉淡化帧数：>0 柔化衔接顿挫(0=关闭)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("成片帧", "总帧数", "拼接报告")
    FUNCTION = "compose"
    CATEGORY = "CGS导演台"

    def compose(self, transition_frames, shots_json, fps=24, color_match=0.7, junction_blend=0):
        report = []
        if transition_frames is None:
            raise RuntimeError("CGS-成片合成：未连接过渡段帧(IMAGE)，请连接 VAEDecode 输出。")

        th, tw = transition_frames.shape[1], transition_frames.shape[2]
        fps = int(fps) or int(CGS_FPS)
        color_match = max(0.0, min(1.0, float(color_match or 0.0)))
        junction_blend = max(0, int(junction_blend or 0))
        trans_list = [transition_frames[i] for i in range(transition_frames.shape[0])]

        # ===== 解析 JSON =====
        data = {}
        try:
            data = json.loads(shots_json) if isinstance(shots_json, str) else (shots_json or {})
        except Exception as e:
            raise RuntimeError("CGS-成片合成：shots_json 解析失败: %s" % e)
        if not isinstance(data, dict):
            raise RuntimeError("CGS-成片合成：shots_json 不是对象。")
        materials = data.get("materials") or {}
        sm = data.get("seamless") or {}
        sm_segments = sm.get("segments") or []
        sm_sel = sm.get("selection") or {}
        sm_start = float(sm_sel.get("start", 2.0) or 2.0)
        sm_dur = float(sm_sel.get("duration", 3.0) or 3.0)
        sm_end = sm_start + sm_dur

        report.append("成片: 分辨率 %dx%d 帧率 %d fps" % (tw, th, fps))
        report.append("选区: %.2f-%.2fs 时长 %.2fs" % (sm_start, sm_end, sm_dur))
        report.append("过渡段帧: %d 帧 (%.2fs)" % (len(trans_list), len(trans_list) / fps))
        if color_match > 0:
            report.append("色彩匹配: 强度 %.2f" % color_match)
        if junction_blend > 0:
            report.append("衔接平滑: 交叉淡化 %d 帧" % junction_blend)

        # ===== 素材名解析 =====
        vnames = materials.get("videos") or []
        def _vpath(vidx):
            if 0 <= vidx < len(vnames):
                try:
                    return folder_paths.get_annotated_filepath(vnames[vidx])
                except Exception:
                    return os.path.join(folder_paths.get_input_directory(), str(vnames[vidx]))
            return None

        # ===== 有效片段(过滤有素材的) =====
        valid_segs = []
        for seg in sm_segments:
            if not isinstance(seg, dict):
                continue
            s_start = float(seg.get("start", 0) or 0)
            s_end = float(seg.get("end", 0) or 0)
            s_vidx = int(seg.get("video_idx", 0) or 0)
            if s_end > s_start and _vpath(s_vidx):
                valid_segs.append(seg)
        ordered = sorted(valid_segs, key=lambda s: (float(s["start"]), float(s["end"])))

        left = None
        right = None
        left_cands = [c for c in ordered if float(c["start"]) < sm_start]
        right_cands = [c for c in ordered if float(c["end"]) > sm_end]
        if left_cands:
            left = max(left_cands, key=lambda c: float(c["end"]))
        if right_cands:
            right = min(right_cands, key=lambda c: float(c["start"]))

        lf = []
        rf = []

        # ===== 上一段：时间轴 [left.start, selection.start] =====
        if left is not None:
            lv = _vpath(int(left["video_idx"]))
            t_l_start = _src_time(left, float(left["start"]))
            left_end_tl = min(float(left["end"]), sm_start)   # 与后端衔接点 left_tl 严格一致
            t_l_end = _src_time(left, left_end_tl)
            if t_l_end > t_l_start + 1e-6:
                try:
                    lf = _extract_clip(lv, t_l_start, t_l_end, fps, th, tw)
                    report.append("上一段: %s 源 %.2f-%.2fs → %d帧(衔接点对齐) " % (
                        vnames[int(left["video_idx"])], t_l_start, t_l_end, len(lf)))
                except Exception as e:
                    report.append("上一段裁剪失败: %s" % str(e))
            else:
                report.append("上一段时长过短，跳过")
        else:
            report.append("选区左侧无片段，成片从过渡段开始")

        # ===== 下一段：时间轴 [selection.end, right.end] =====
        if right is not None:
            rv = _vpath(int(right["video_idx"]))
            right_start_tl = max(float(right["start"]), sm_end)  # 与后端衔接点 right_tl 严格一致
            t_r_start = _src_time(right, right_start_tl)
            t_r_end = _src_time(right, float(right["end"]))
            if t_r_end > t_r_start + 1e-6:
                try:
                    rf = _extract_clip(rv, t_r_start, t_r_end, fps, th, tw)
                    report.append("下一段: %s 源 %.2f-%.2fs → %d帧(衔接点对齐) " % (
                        vnames[int(right["video_idx"])], t_r_start, t_r_end, len(rf)))
                except Exception as e:
                    report.append("下一段裁剪失败: %s" % str(e))
            else:
                report.append("下一段时长过短，跳过")
        else:
            report.append("选区右侧无片段，成片到过渡段结束")

        # ===== 过渡段色彩匹配：向左右段色调靠拢 =====
        if color_match > 0 and trans_list:
            ref_start = lf[-1] if lf else None
            ref_end = rf[0] if rf else None
            if ref_start is not None or ref_end is not None:
                trans_list = _match_color_series(trans_list, ref_start, ref_end, color_match)
                report.append("色彩匹配: 参照 %s" % ("左段末帧" if ref_start is not None else "") +
                              ("+" if ref_start is not None and ref_end is not None else "") +
                              ("右段首帧" if ref_end is not None else ""))

        # ===== 拼接点交叉淡化（柔化顿挫） =====
        if junction_blend > 0:
            if lf and trans_list:
                _blend_junction(lf, trans_list, junction_blend)
                report.append("衔接平滑: 左段↔过渡 淡化 %d 帧" % min(junction_blend, len(lf), len(trans_list)))
            if trans_list and rf:
                _blend_junction(trans_list, rf, junction_blend)
                report.append("衔接平滑: 过渡↔右段 淡化 %d 帧" % min(junction_blend, len(trans_list), len(rf)))

        all_frames = lf + trans_list + rf
        if not all_frames:
            raise RuntimeError("CGS-成片合成：无任何可用帧。")

        out = torch.stack(all_frames)
        total = out.shape[0]
        report.append("成片: 共 %d 帧 ≈ %.2fs" % (total, total / fps))
        return (out, total, "\n".join(report))
