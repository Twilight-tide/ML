"""强制修正临时目录。

背景（两个问题，先后遇到）：

1) 最初：VS Code 的 Jupyter 扩展启动 kernel 时，TEMP/TMP 被解析成了
   系统级的 C:\\WINDOWS\\TEMP，该目录对普通用户半受限，导致 ipykernel 报
   FileNotFoundError: No usable temporary directory found。
   处理：把 TEMP 指回用户级临时目录。

2) 后来（2026-11-17）：本机对「工作区外的写入」有限制 ——
   同一个 C:\\Users\\Lenovo\\AppData\\Local\\Temp，
   系统 Python 能 mkdtemp，而本 venv 的 Python 报
   PermissionError: [WinError 5] 拒绝访问。
   实测：venv 的 Python 只能写项目目录内部（工作区内 OK）。
   而 pip 装包第一步就是 tempfile.mkdtemp()，于是永久卡住、无任何输出。

   处理：把临时目录指到**项目内部**的 .tmp/（已在 .gitignore 里排除）。

此文件由 site 模块在每次 Python 启动时自动导入，早于 pip / ipykernel / 用户代码。
"""

import os

# 由本文件位置反推项目根：site-packages -> Lib -> .venv -> 项目根
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_USER_TMP = os.path.join(_PROJECT_ROOT, ".tmp")

try:
    os.makedirs(_USER_TMP, exist_ok=True)
except Exception:
    # 建不出来就退回系统默认，让 Python 自己去找
    _USER_TMP = None

if _USER_TMP and os.path.isdir(_USER_TMP):
    os.environ["TEMP"] = _USER_TMP
    os.environ["TMP"] = _USER_TMP
    try:
        import tempfile

        tempfile.tempdir = _USER_TMP
    except Exception:
        # 这里绝不能抛异常，否则所有 Python 进程都会启动失败
        pass

# pip 的 HTTP 缓存默认写 %LOCALAPPDATA%\pip\Cache，同样在工作区外，
# 会在 cachecontrol 写缓存时卡死（faulthandler 栈显示卡在
# pip/_internal/network/cache.py:85 _write_to_file → NamedTemporaryFile）。
# 指到项目内，之后 pip 不用再带 --no-cache-dir。
if _USER_TMP:
    _PIP_CACHE = os.path.join(_USER_TMP, "pip-cache")
    try:
        os.makedirs(_PIP_CACHE, exist_ok=True)
        os.environ.setdefault("PIP_CACHE_DIR", _PIP_CACHE)
    except Exception:
        pass
