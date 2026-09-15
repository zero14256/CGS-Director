# ============================================================
# CGS-导演台 (CGS-Director) · ComfyUI MiniMax H3 导演工作台
# Author: zero14256
# Repo:   https://github.com/zero14256/comfyui-MinimaxH3-CGS-Director
# License: MIT · 请保留此来源标识
# ============================================================

from .cgs_director_core import CGSDirectorCore
from .cgs_seamless_compose import CGSSeamlessCompose
from . import cgs_history  # noqa: F401  注册 /cgs_history/* 生成历史路由

NODE_CLASS_MAPPINGS = {
    "CGSDirectorCore": CGSDirectorCore,
    "CGSSeamlessCompose": CGSSeamlessCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CGSDirectorCore": "CGS-导演台·核心",
    "CGSSeamlessCompose": "CGS-成片合成",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
