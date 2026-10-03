#!/usr/bin/env bash
# Make the pip-installed `bpy` wheel find the X11/OpenGL libraries installed in the conda env
# (xorg-*, libgl, libxkbcommon from environment.yml). pip wheels only search their own folder and
# the system library path, so on headless machines without those libraries `import bpy` fails with
# "libX11.so.6: cannot open shared object file". Adding the env's lib/ to the rpath fixes that
# without touching LD_LIBRARY_PATH. Safe to re-run; re-run after reinstalling bpy.
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the conda env first (conda activate mosaic-datagen)." >&2
    exit 1
fi
command -v patchelf >/dev/null || { echo "patchelf not found; install it with: conda install -c conda-forge patchelf" >&2; exit 1; }

# Locate site-packages/bpy without importing it: the import fails before the patch, and after a
# successful import Blender swaps sys.modules['bpy'] for an inner module, so bpy.__file__ is misleading.
BPY_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('bpy').origin))")
echo "bpy package: $BPY_DIR"

# <env>/lib relative to the object being patched
patchelf --set-rpath '$ORIGIN/lib:$ORIGIN/../../..' "$BPY_DIR/__init__.so"
for lib in libMaterialXRenderHw.so.1 libusd_ms.so; do
    [[ -f "$BPY_DIR/lib/$lib" ]] && patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../..' "$BPY_DIR/lib/$lib"
done

python -c "import bpy; print('bpy', bpy.app.version_string, 'imports OK')"
