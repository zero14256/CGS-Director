# -*- coding: utf-8 -*-
"""
CGS-Director 生成历史记录后端（独立新增，不改动 cgs_director_core.py / cgs_timeline.js）

API:
  POST /cgs_history/save   保存一条生成记录（JSON + 缩略图）
  GET  /cgs_history/list   列出历史记录
  GET  /cgs_history/thumb  读取缩略图
  POST /cgs_history/delete 删除单条记录（JSON + 缩略图）
  POST /cgs_history/clear  清空整个历史目录
"""
import json
import os
import shutil
import subprocess

from aiohttp import web
from server import PromptServer
from folder_paths import get_output_directory, get_temp_directory, get_input_directory

try:
    from PIL import Image
except Exception:
    Image = None

DEFAULT_PATH = "D:/CGS_History"
routes = PromptServer.instance.routes

_FFMPEG = None
# 按插件安装位置自动推导 ffmpeg 候选（兼容任意 ComfyUI 安装目录，不再硬编码本机路径）
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_COMFYUI_ROOT = os.path.dirname(os.path.dirname(_PLUGIN_DIR))  # custom_nodes/<插件>/.. = ComfyUI 根目录
_FFMPEG_CANDIDATES = [
    os.path.join(_COMFYUI_ROOT, "python_embeded", "Scripts", "ffmpeg.exe"),
    os.path.join(_COMFYUI_ROOT, "models", "Apt_File", "ffmpeg.exe"),
    "ffmpeg",
]


def _find_ffmpeg():
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG
    for c in _FFMPEG_CANDIDATES:
        if not c:
            continue
        p = shutil.which(c) if not os.path.isabs(c) else c
        if p and os.path.isfile(p):
            _FFMPEG = p
            return p
    _FFMPEG = ""
    return None


def _thumb_from_video(src, tpath):
    ff = _find_ffmpeg()
    if not ff:
        return False
    try:
        subprocess.run(
            [ff, "-y", "-i", src, "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "4", tpath],
            capture_output=True, timeout=60, check=True,
        )
        return os.path.isfile(tpath)
    except Exception:
        return False


def _norm(p):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p or DEFAULT_PATH)))


def _safe_name(name):
    """仅允许简单文件名，防路径穿越"""
    name = os.path.basename(str(name or ""))
    return name if name and name not in (".", "..") else ""


def _resolve_dir(typ):
    """按 ComfyUI 资源类型解析文件所在目录（output/temp/input）"""
    return {
        "temp": get_temp_directory(),
        "input": get_input_directory(),
    }.get(typ or "output", get_output_directory())


@routes.post("/cgs_history/save")
async def cgs_history_save(request):
    body = await request.json()
    base = _norm(body.get("path"))
    entry = body.get("entry") or {}
    images = body.get("images") or []
    eid = _safe_name(entry.get("id") or "rec")
    if not eid:
        return web.json_response({"ok": False, "error": "bad id"})
    try:
        os.makedirs(base, exist_ok=True)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})

    thumbs = []
    if Image is not None:
        for i, im in enumerate(images[:24]):  # 最多 24 张
            filename = _safe_name(im.get("filename"))
            if not filename:
                continue
            src = os.path.join(_resolve_dir(im.get("type")), im.get("subfolder") or "", filename)
            if not os.path.isfile(src):
                continue
            tname = "%s_%d.jpg" % (eid, i)
            try:
                img = Image.open(src)
                img.thumbnail((320, 320))
                img.convert("RGB").save(os.path.join(base, tname), "JPEG", quality=80)
                thumbs.append(tname)
            except Exception:
                # 图片打不开 → 按视频处理（mp4/webm/avi…），ffmpeg 抽第一帧
                if _thumb_from_video(src, os.path.join(base, tname)):
                    thumbs.append(tname)

    entry["images"] = thumbs
    entry["src_images"] = [im.get("filename") for im in images if im.get("filename")]
    try:
        with open(os.path.join(base, eid + ".json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})
    return web.json_response({"ok": True, "thumbs": len(thumbs)})


@routes.get("/cgs_history/list")
async def cgs_history_list(request):
    base = _norm(request.query.get("path"))
    if not os.path.isdir(base):
        return web.json_response({"ok": True, "items": []})
    items = []
    for fn in os.listdir(base):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(base, fn), encoding="utf-8") as f:
                d = json.load(f)
            d["_file"] = fn
            items.append(d)
        except Exception:
            pass
    items.sort(key=lambda x: str(x.get("id", "")), reverse=True)
    return web.json_response({"ok": True, "items": items})


@routes.get("/cgs_history/thumb")
async def cgs_history_thumb(request):
    base = _norm(request.query.get("path"))
    name = _safe_name(request.query.get("name"))
    if not name:
        return web.Response(status=404)
    p = os.path.join(base, name)
    if not os.path.isfile(p):
        return web.Response(status=404)
    return web.FileResponse(p)


@routes.post("/cgs_history/delete")
async def cgs_history_delete(request):
    body = await request.json()
    base = _norm(body.get("path"))
    name = _safe_name(body.get("name"))
    removed = []
    if name:
        p = os.path.join(base, name + ".json")
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed.append(name + ".json")
            except Exception:
                pass
        # 缩略图：<id>_N.jpg
        if os.path.isdir(base):
            for fn in os.listdir(base):
                if fn.startswith(name + "_") and fn.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(base, fn))
                        removed.append(fn)
                    except Exception:
                        pass
    return web.json_response({"ok": True, "removed": removed})


@routes.post("/cgs_history/clear")
async def cgs_history_clear(request):
    body = await request.json()
    base = _norm(body.get("path"))
    n = 0
    if os.path.isdir(base):
        for fn in os.listdir(base):
            if fn.endswith(".json") or (fn.endswith(".jpg") and "_" in fn):
                try:
                    os.remove(os.path.join(base, fn))
                    n += 1
                except Exception:
                    pass
    return web.json_response({"ok": True, "removed": n})
