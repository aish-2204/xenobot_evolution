#!/bin/bash
# ================================================================
# build_voxelyze_mac.sh
# Compiles the CPU-only voxelyze binary on Apple Silicon Mac.
# No CUDA, no GPU, no OpenGL needed.
#
# Run from your project root:
#   chmod +x build_voxelyze_mac.sh
#   ./build_voxelyze_mac.sh
#
# On success: ./voxelyze binary appears in current directory.
# ================================================================
set -e

echo "==> Checking dependencies..."
if ! command -v git &>/dev/null; then echo "❌ git not found. Install Xcode CLI: xcode-select --install"; exit 1; fi
if ! command -v c++ &>/dev/null; then echo "❌ c++ not found. Install Xcode CLI: xcode-select --install"; exit 1; fi
echo "✅ git and c++ found"

echo ""
echo "==> Cloning evosoro (contains CPU voxelyze source)..."
if [ -d "_evosoro_src" ]; then
    echo "   _evosoro_src already exists, skipping clone"
else
    git clone --quiet https://github.com/skriegman/evosoro.git _evosoro_src
    echo "✅ Cloned"
fi

SRC="_evosoro_src/evosoro/_voxcad/Voxelyze"
MAIN_SRC="_evosoro_src/evosoro/_voxcad/voxelyzeMain"
BUILD="_voxelyze_build"

echo ""
echo "==> Setting up build directory..."
mkdir -p "$BUILD"

echo ""
echo "==> Writing headless main.cpp (no OpenGL, no Qt)..."
# The original main.cpp uses VXS_SimGLView (needs Qt/OpenGL).
# We write a clean headless version that runs the same simulation loop.
cat > "$BUILD/main_headless.cpp" << 'MAIN_EOF'
/**
 * voxelyze_headless - CPU soft-body simulator, no GL dependencies.
 * Usage: voxelyze -f <input.vxa> [-o <output.xml>] [-p]
 *
 * Output XML contains <NormFinalDist> — the fitness score (CoM displacement
 * normalised by body characteristic length).
 */
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

// Core simulation headers (no GL)
#include "VX_Object.h"
#include "VX_Environment.h"
#include "VX_SimGA.h"

int main(int argc, char* argv[])
{
    char* inputFile  = nullptr;
    char* outputFile = nullptr;
    bool  printScreen = false;

    if (argc < 3) {
        std::cout << "\nInput file required. Quitting.\n";
        std::cout << "Usage: voxelyze -f <input.vxa> [-o <output.xml>] [-p]\n";
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            inputFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
        else if (strcmp(argv[i], "-p") == 0)
            printScreen = true;
    }

    if (!inputFile) {
        std::cout << "No input file specified. Use -f <file.vxa>\n";
        return 0;
    }

    // Wire up the simulation objects
    CVX_Object      Object;
    CVX_Environment Environment;
    CVX_SimGA       Simulator;
    Environment.pObj = &Object;
    Simulator.pEnv   = &Environment;

    // Override output file if specified on command line
    if (outputFile)
        Simulator.FitnessFileName = std::string(outputFile);

    if (!Simulator.LoadVXAFile(inputFile)) {
        std::cerr << "Problem importing VXA file: " << inputFile << "\n";
        return 0;
    }

    std::string msg;
    Simulator.Import(&Environment, 0, &msg);

    double time = 0.0;
    long   step = 0;
    Simulator.pEnv->UpdateCurTemp(time);

    if (printScreen)
        std::cout << "Starting simulation...\n";

    while (!Simulator.StopConditionMet()) {
        if (printScreen && step % 500 == 0)
            std::cout << "  t=" << time
                      << "  CoM=" << Simulator.GetCM().Length() << "\n";
        Simulator.TimeStep(&msg);
        step++;
        time += Simulator.dt;
        Simulator.pEnv->UpdateCurTemp(time);
    }

    if (printScreen)
        std::cout << "Simulation ended at t=" << time
                  << "  CoM displacement=" << Simulator.GetCM().Length() << "\n";

    // Write results XML (contains <NormFinalDist>)
    Simulator.SaveResultFile(Simulator.FitnessFileName);

    if (printScreen)
        std::cout << "Results saved to: " << Simulator.FitnessFileName << "\n";

    return 1;
}
MAIN_EOF
echo "✅ main_headless.cpp written"

echo ""
echo "==> Collecting source files..."

# All CPU .cpp sources from Voxelyze/ — exclude GL/Mesh/Benchmark files
CPP_SOURCES=""
for f in \
    VX_Object.cpp \
    VX_Environment.cpp \
    VX_Sim.cpp \
    VX_SimGA.cpp \
    VX_Source.cpp \
    VX_Bond.cpp \
    VX_Voxel.cpp \
    VX_FRegion.cpp \
    VX_FEA.cpp \
    VXS_Voxel.cpp \
    VXS_Bond.cpp \
    VXS_BondInternal.cpp \
    VXS_BondCollision.cpp; do
    if [ -f "$SRC/$f" ]; then
        CPP_SOURCES="$CPP_SOURCES $SRC/$f"
    else
        echo "   ⚠️  Missing $f (may be OK)"
    fi
done

# Utils folder
UTILS_DIR="$SRC/Utils"
if [ -d "$UTILS_DIR" ]; then
    for f in "$UTILS_DIR"/*.cpp; do
        [ -f "$f" ] && CPP_SOURCES="$CPP_SOURCES $f"
    done
fi

echo "✅ Sources collected"

echo ""
echo "==> Compiling voxelyze (this takes ~30 seconds)..."

INCLUDES="-I$SRC -I$SRC/Utils -I$MAIN_SRC"
FLAGS="-O2 -std=c++14 -arch arm64 -Wno-deprecated -Wno-unused-variable -Wno-sign-compare"
SDK="-isysroot $(xcrun --show-sdk-path)"

# Compile each source to object file
OBJS=""
cd "$BUILD"
for src in $CPP_SOURCES; do
    obj=$(basename "${src%.cpp}.o")
    echo -n "   Compiling $(basename $src)..."
    c++ $FLAGS $SDK $INCLUDES -c "../$src" -o "$obj" 2>/dev/null && echo " ✓" || {
        echo " ✗ (trying without strict flags)"
        c++ -O1 -std=c++14 -arch arm64 $SDK $INCLUDES \
            -Wno-everything -c "../$src" -o "$obj"
    }
    OBJS="$OBJS $obj"
done

# Compile headless main
echo -n "   Compiling main_headless.cpp..."
c++ $FLAGS $SDK $INCLUDES -c "main_headless.cpp" -o "main_headless.o" && echo " ✓"
OBJS="$OBJS main_headless.o"

echo ""
echo -n "==> Linking voxelyze binary..."
c++ $FLAGS $SDK -o "../voxelyze" $OBJS && echo " ✅"
cd ..

echo ""
echo "==> Verifying binary..."
file ./voxelyze
./voxelyze 2>&1 | head -3

echo ""
echo "================================================================"
echo "✅ SUCCESS! voxelyze binary compiled."
echo ""
echo "Next steps:"
echo "  1. Add to PATH:"
echo "     echo 'export PATH=\$PATH:'$(pwd) >> ~/.zshrc && source ~/.zshrc"
echo ""
echo "  2. Set env var for your Python fitness function:"
echo "     echo 'export VOXCRAFT_BIN='$(pwd)/voxelyze >> ~/.zshrc"
echo "     source ~/.zshrc"
echo ""
echo "  3. Test it:"
echo "     python src/milestone1/fitness.py"
echo "================================================================"
