# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build for BioCarbon Screen (one-folder).

    pyinstaller --noconfirm BioCarbon_Screen.spec

Produces  dist/BioCarbon Screen/BioCarbon Screen.exe  plus its _internal/ libs.
Distribute the whole "BioCarbon Screen" folder (zip it to share).
"""
import os
import shutil
from PyInstaller.utils.hooks import (
    collect_all, collect_submodules, collect_data_files, collect_dynamic_libs,
)

APP_NAME = "BioCarbon Screen"

datas = []
binaries = []
hiddenimports = []

# imgui_bundle and xgboost ship native libs but their submodule trees have
# import-time side effects (imgui_bundle demos call exit(1); xgboost.testing
# needs 'hypothesis'), which abort collect_all's submodule scan. Grab their
# binaries + data WITHOUT importing submodules, and name the code imports.
for pkg in ("imgui_bundle", "xgboost"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception as exc:  # pragma: no cover
        print(f"[spec] collect_dynamic_libs({pkg}) skipped: {exc}")
    try:
        datas += collect_data_files(
            pkg, excludes=["**/demos_*/**", "**/doc/**", "**/*.pyi"]
        )
    except Exception as exc:  # pragma: no cover
        print(f"[spec] collect_data_files({pkg}) skipped: {exc}")

hiddenimports += [
    "xgboost", "xgboost.core", "xgboost.sklearn", "xgboost.compat",
    "xgboost.training", "xgboost.callback", "xgboost.libpath", "xgboost.data",
    "xgboost.plotting",
]

# These collect cleanly (no import-time side effects): native libs + data + code.
for pkg in ("lightgbm", "catboost", "shap"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # pragma: no cover
        print(f"[spec] collect_all({pkg}) skipped: {exc}")

# scikit-learn / scipy have many Cython submodules imported by string.
for pkg in ("sklearn", "scipy"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception as exc:  # pragma: no cover
        print(f"[spec] collect_submodules({pkg}) skipped: {exc}")

hiddenimports += [
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
    "sklearn.tree._utils",
]

# This app's own sibling modules (all imported at top level, listed for safety).
hiddenimports += [
    "charts", "screening", "report", "units", "slideshow", "validation",
    "chemistry_features", "cost_model", "planner", "research_gap", "pareto",
    "constraint_engine", "sustainability", "tradeoffs", "portfolio",
    "intelligence", "latent", "bayesopt",
]

a = Analysis(
    ["app_imgui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6", "PyQt5", "PyQt6", "tkinter", "PIL.ImageQt",
        "IPython", "notebook", "jupyter", "pytest", "_pytest",
        # Deep-learning / ONNX stacks that collect_all(shap) drags in for its
        # DeepExplainer/ONNX backends. This app only uses shap's TreeExplainer
        # (forest models), which never imports these — excluding them removes
        # ~4 GB (torch alone is 3.6 GB) with no loss of function.
        "torch", "torchvision", "torchaudio",
        "tensorflow", "keras",
        "onnx", "onnxruntime", "skl2onnx",
        "av",  # PyAV — only reachable via torchvision
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed app — no console flashes on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# --- post-build: drop side files NEXT TO the .exe (not inside _internal) ------
# At runtime the app chdir()s to the .exe folder and points the asset loader
# there, so fonts/ and a starter model.joblib must live at the folder root.
app_root = os.path.join(DISTPATH, APP_NAME)
if os.path.isdir(app_root):
    try:
        import imgui_bundle
        fonts_src = os.path.join(os.path.dirname(imgui_bundle.__file__),
                                 "assets", "fonts")
        shutil.copytree(fonts_src, os.path.join(app_root, "fonts"),
                        dirs_exist_ok=True)
        print("[spec] bundled Roboto fonts next to the exe")
    except Exception as exc:  # pragma: no cover
        print(f"[spec] font copy skipped: {exc}")
    if os.path.isfile("model.joblib"):
        try:
            shutil.copy("model.joblib", app_root)
            print("[spec] bundled starter model.joblib next to the exe")
        except Exception as exc:  # pragma: no cover
            print(f"[spec] model copy skipped: {exc}")
