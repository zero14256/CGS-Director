RESOLUTION_PRESETS = {
    "1080P 横屏 16:9": (1920, 1080),
    "2K 横屏 16:9": (2048, 1152),
    "720P 横屏 16:9": (1280, 720),
    "1080P 竖屏 9:16": (1080, 1920),
    "720P 竖屏 9:16": (720, 1280),
    "方形 1:1": (1024, 1024),
    "2K 方形 1:1": (1440, 1440),
    "2K 竖屏 9:16": (1152, 2048),
}

RESOLUTION_PRESETS_ORDER = [
    "1080P 横屏 16:9",
    "2K 横屏 16:9",
    "720P 横屏 16:9",
    "1080P 竖屏 9:16",
    "720P 竖屏 9:16",
    "方形 1:1",
    "2K 方形 1:1",
    "2K 竖屏 9:16",
    "自定义(使用下方宽高)",
]

MEGAPIXELS_PRESETS = {
    "0.2MP 16:9 (608x352)":   (608, 352),
    "0.3MP 16:9 (736x416)":   (736, 416),
    "0.4MP 16:9 (864x480)":   (864, 480),
    "0.5MP 16:9 (960x544)":   (960, 544),
    "0.6MP 16:9 (1056x608)":  (1056, 608),
    "0.7MP 16:9 (1152x640)":  (1152, 640),
    "0.8MP 16:9 (1216x672)":  (1216, 672),
    "0.9MP 16:9 (1280x736)":  (1280, 736),
    "0.98MP 16:9 (1344x768)": (1344, 768),
    "1.0MP 16:9 (1376x768)":  (1376, 768),
    "1.2MP 16:9 (1504x832)":  (1504, 832),
    "1.5MP 16:9 (1664x928)":  (1664, 928),
    "1.8MP 16:9 (1824x1024)": (1824, 1024),
    "2.0MP 16:9 (1920x1088)": (1920, 1088),
}

MEGAPIXELS_PRESETS_ORDER = list(MEGAPIXELS_PRESETS.keys())

CANVAS_MULTIPLE = 32
FPS = 24
TRAINED_MAX_FRAMES = 362

CAMERA_PRESETS = {
    "无运镜": "",
    "推镜头(向前推进)": "The camera slowly pushes forward toward the subject.",
    "拉镜头(向后拉远)": "The camera slowly pulls back, revealing more of the scene.",
    "摇镜头(水平横摇)": "The camera pans horizontally across the scene.",
    "移镜头(横向平移)": "The camera trucks laterally to the side, following the action.",
    "升镜头(向上)": "The camera tilts upward to reveal the higher space.",
    "降镜头(向下)": "The camera tilts downward toward the ground.",
    "缩放(放大主体)": "The camera zooms in gradually on the subject.",
    "环绕(环绕主体)": "The camera orbits slowly around the subject in a circular motion.",
    "跟随(跟拍主体)": "The camera follows the moving subject steadily.",
    "手持(轻微晃动)": "Handheld camera with subtle natural shake.",
    "航拍(高空俯瞰)": "Aerial shot, high above looking down.",
}

CAMERA_PRESETS_ORDER = list(CAMERA_PRESETS.keys())

STYLE_PRESETS = {
    "默认": "",
    "电影感": "cinematic lighting, film grain, shallow depth of field, dramatic composition, ",
    "写实": "photorealistic, natural lighting, high detail, ",
    "二次元": "anime style, clean lineart, vibrant colors, ",
    "短剧": "drama series style, expressive characters, ",
    "微表情": "subtle facial micro-expressions, detailed eye movement, orbicularis oculi contraction, crow's feet at eye corners, eyelid micro-movement, cheek elevation, asymmetrical mouth corner twitch, lip compression and release, brow furrowing and release, nostril micro-flare, forehead transient wrinkles, pupil dilation, gaze aversion and sidelong glancing, natural blinking rhythm, lip sync detail, emotional nuance, lifelike character performance, expressive face, detailed facial muscle movement, micro-expression leakage, ",
    "短视频": "trendy short-video aesthetic, punchy framing, ",
    "赛博朋克": "cyberpunk, neon lights, high contrast, ",
    "水墨风": "Chinese ink wash painting style, soft brush strokes, ",
}

STYLE_PRESETS_ORDER = list(STYLE_PRESETS.keys())

QUALITY_PRESETS = {
    "标准": {"steps": 25, "shift_video": 12.0, "shift_audio": 3.0, "cfg": 1.0,
             "sampler": "res_multistep", "scheduler": "native"},
    "高画质": {"steps": 40, "shift_video": 14.0, "shift_audio": 3.5, "cfg": 1.0,
               "sampler": "res_multistep", "scheduler": "native"},
    "极速/低显存": {"steps": 8, "shift_video": 10.0, "shift_audio": 2.5, "cfg": 1.0,
                   "sampler": "res_multistep", "scheduler": "native"},
}

QUALITY_PRESETS_ORDER = list(QUALITY_PRESETS.keys())

REFERENCE_GROUPS = {
    "role": "角色参考",
    "scene": "场景参考",
    "style": "风格参考",
    "prop": "道具参考",
    "camera": "运镜参考",
}

REFERENCE_GROUP_ORDER = ["role", "scene", "style", "prop", "camera"]


def snap_dim(v):
    v = int(round(float(v)))
    m = CANVAS_MULTIPLE
    return max(m, (v // m) * m)


def duration_to_frames(seconds):
    n = max(5, int(round(max(0.1, float(seconds)) * FPS)))
    while n % 17 != 5:
        n += 1
    return n


import re as _re

def megapixels_from_string(s):
    """从任意格式的 megapixels 字符串中提取 (width, height)。
    支持：'0.5MP 16:9 (960x544)'、'0.5MP 960×544'、'960x544' 等。"""
    if not s:
        return None
    # 优先精确匹配字典
    if s in MEGAPIXELS_PRESETS:
        return MEGAPIXELS_PRESETS[s]
    # 从字符串中提取数字（支持 x × X 分隔）
    m = _re.search(r"(\d{3,5})\s*[x×X]\s*(\d{3,5})", str(s))
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if 100 <= w <= 8192 and 100 <= h <= 8192:
            return (w, h)
    return None


def resolution_from_preset(preset, width, height):
    if preset not in ("自定义(使用下方宽高)", "自定义") and preset in RESOLUTION_PRESETS:
        w, h = RESOLUTION_PRESETS[preset]
    else:
        w, h = width, height
    return snap_dim(w), snap_dim(h)
