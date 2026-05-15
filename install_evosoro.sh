#!/bin/bash
# ============================================================
# Evosoro Python 3 installer for Xenobot exercise
# Works on macOS and Linux. Run from your project root.
# ============================================================
set -e

echo "==> Step 1: Creating Python virtual environment..."
python3 -m venv xenobot_env
source xenobot_env/bin/activate

echo "==> Step 2: Installing Python dependencies..."
pip install --upgrade pip -q
pip install numpy scipy networkx deap pyvista matplotlib pytest tqdm -q
echo "✅ Python packages installed"

echo "==> Step 3: Cloning evosoro from GitHub..."
git clone --quiet https://github.com/skriegman/evosoro.git _evosoro_src
echo "✅ evosoro cloned"

echo "==> Step 4: Patching evosoro for Python 3..."
python3 - << 'PYEOF'
import re, os, glob

files = glob.glob('_evosoro_src/evosoro/**/*.py', recursive=True)
files += glob.glob('_evosoro_src/evosoro/*.py')

MODULES = [
    'read_write_voxelyze','evaluation','algorithms','checkpointing',
    'data_analysis','mutation','selection','utils',
    'base','networks','softbot','tools'
]

for fpath in files:
    txt = open(fpath).read()

    # Fix print statements: print "x" → print("x")
    txt = re.sub(r'\bprint (\"[^\"\\n]*\")', r'print(\1)', txt)
    txt = re.sub(r"\bprint ('([^'\\n]*)')", r'print(\1)', txt)
    txt = re.sub(r'\bprint ("([^"\\n]*)",\s*([^\n]+))', r'print(\2, \3)', txt)

    # Fix bare relative imports
    for m in MODULES:
        txt = re.sub(r'^from ' + m + r' import', 'from .' + m + ' import', txt, flags=re.MULTILINE)

    open(fpath, 'w').write(txt)

print("✅ Python 3 patches applied")
PYEOF

echo "==> Step 5: Copying pre-compiled voxelyze binary..."
VOXBIN="_evosoro_src/evosoro/_voxcad/voxelyzeMain/voxelyze"
if [ -f "$VOXBIN" ]; then
    cp "$VOXBIN" xenobot_env/bin/voxelyze
    chmod +x xenobot_env/bin/voxelyze
    echo "✅ voxelyze binary installed to xenobot_env/bin/"
else
    echo "⚠️  Pre-compiled binary not found — you may need to compile from source"
    echo "   cd _evosoro_src/evosoro/_voxcad/Voxelyze && make && make installusr"
fi

echo "==> Step 6: Installing evosoro into venv..."
pip install -e _evosoro_src -q 2>/dev/null || {
    # No setup.py — add to path via .pth file instead
    SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    echo "_evosoro_src" > "$SITE/evosoro.pth"
    echo "✅ evosoro added via .pth file"
}

echo "==> Step 7: Verifying installation..."
python3 - << 'PYEOF'
import sys
sys.path.insert(0, '_evosoro_src')
from evosoro.base import Sim, Env
from evosoro.tools.read_write_voxelyze import write_voxelyze_file
import subprocess, shutil
vox = shutil.which('voxelyze')
print("✅ evosoro imports OK")
print("✅ voxelyze binary:", vox if vox else "NOT FOUND (compile manually)")
PYEOF

echo ""
echo "============================================"
echo "✅ DONE. To activate your environment:"
echo "   source xenobot_env/bin/activate"
echo ""
echo "To verify voxelyze works:"
echo "   voxelyze   # should print 'Input file required. Quitting.'"
echo "============================================"
