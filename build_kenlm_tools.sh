#!/usr/bin/env bash
# Builds the kenlm Python package and the lmplz/build_binary CLI tools from
# source, working around two issues hit on this machine (Python 3.13 /
# recent Boost):
#
#   1. kenlm's PyPI wheel ships a pre-generated python/kenlm.cpp that
#      predates Python 3.13's C API changes (_PyLong_AsByteArray signature,
#      _PyGen_SetStopIterationValue removal) and fails to compile. Fix:
#      regenerate kenlm.cpp from the .pyx source with a current Cython.
#   2. KenLM's CMakeLists.txt requires Boost's "system" component, which
#      modern Boost (1.81+) no longer ships as a separate compiled library
#      (it's header-only now and has no CMake config). Fix: drop "system"
#      from the required COMPONENTS list.
#
# Requires (Debian/Kali): sudo apt install libboost-program-options-dev \
#   libboost-thread-dev libboost-iostreams-dev libboost-test-dev libboost-dev
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(mktemp -d)/kenlm_src"

pip install --upgrade cython

echo "Fetching kenlm source..."
curl -sL https://github.com/kpu/kenlm/archive/master.zip -o /tmp/kenlm.zip
unzip -q /tmp/kenlm.zip -d "$(dirname "$SRC_DIR")"
mv "$(dirname "$SRC_DIR")/kenlm-master" "$SRC_DIR"

echo "Regenerating python/kenlm.cpp with current Cython..."
cython --cplus -3 "$SRC_DIR/python/kenlm.pyx" -o "$SRC_DIR/python/kenlm.cpp"

echo "Patching CMakeLists.txt (drop obsolete Boost 'system' component)..."
sed -i '/^find_package(Boost 1.41.0 REQUIRED COMPONENTS$/,/^)$/{/^  system$/d}' \
  "$SRC_DIR/CMakeLists.txt"

echo "Installing kenlm Python package..."
pip install "$SRC_DIR"

echo "Building lmplz + build_binary CLI tools..."
mkdir -p "$SRC_DIR/build"
cd "$SRC_DIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)" lmplz build_binary

mkdir -p "$PROJECT_DIR/kenlm_bin"
cp bin/lmplz bin/build_binary "$PROJECT_DIR/kenlm_bin/"

echo "Done. kenlm Python module installed; CLI tools in $PROJECT_DIR/kenlm_bin/"
