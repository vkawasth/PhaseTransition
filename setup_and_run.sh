#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_and_run.sh
# One-shot setup and run for all three plot scripts.
# Run this from the directory containing your simulation output files.
# ─────────────────────────────────────────────────────────────────────────────

set -e   # stop on first error

echo "=== Step 1: Check Julia is installed ==="
julia --version || { echo "Julia not found. Install from https://julialang.org/downloads/"; exit 1; }

echo ""
echo "=== Step 2: Install required Julia packages ==="
julia -e '
import Pkg
# GR backend avoids Qt6Base_jll version conflicts
pkgs = ["Plots", "GR", "DelimitedFiles", "Statistics", "LinearAlgebra"]
for p in pkgs
    if !haskey(Pkg.dependencies(), p)
        Pkg.add(p)
        println("Installed: ", p)
    else
        println("Already installed: ", p)
    end
end
# Pin GR to avoid Qt dependency resolution
ENV["GKSwstype"] = "100"   # headless GR (no display needed)
Pkg.precompile()
println("All packages ready.")
'

echo ""
echo "=== Step 3: Check input files ==="
MISSING=0
for f in chambers.tsv; do
    if [ ! -f "$f" ]; then
        echo "  MISSING (required): $f"
        MISSING=1
    else
        ROWS=$(wc -l < "$f")
        COLS=$(head -1 "$f" | tr '\t' '\n' | wc -l)
        echo "  OK: $f  ($ROWS rows, $COLS columns)"
    fi
done

for f in b_eff_snapshots.tsv concentrations.tsv; do
    if [ ! -f "$f" ]; then
        echo "  MISSING (optional, fallback will be used): $f"
    else
        ROWS=$(wc -l < "$f")
        echo "  OK (optional): $f  ($ROWS rows)"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: chambers.tsv is required. See 'chambers.tsv format' below."
    echo ""
    echo "chambers.tsv column order (tab-separated, header row required):"
    echo "  snapshot  q12  q13  q14  q23  q24  q34  m6_max  stratum_k  wall_flag"
    echo ""
    echo "  snapshot  : integer, 1..N"
    echo "  q12..q34  : float, Plücker coordinates at each snapshot"
    echo "  m6_max    : float, max ||m6||_v across all vertices"
    echo "  stratum_k : integer 0..4, current Schubert stratum"
    echo "  wall_flag : 1 if a wall crossing occurred at this snapshot, else 0"
    exit 1
fi

echo ""
echo "=== Step 4: Copy scripts to current directory ==="
SCRIPT_DIR="$(dirname "$0")"
for script in plot_plucker_spectrogram.jl plot9_ihara_pole_dynamics.jl plot10_affinity_reorganisation.jl; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        cp "$SCRIPT_DIR/$script" .
        echo "  Copied: $script"
    elif [ ! -f "$script" ]; then
        echo "  WARNING: $script not found in $SCRIPT_DIR or current directory"
    fi
done

echo ""
echo "=== Step 5: Run scripts ==="

echo ""
echo "--- Running plot_plucker_spectrogram.jl ---"
julia plot_plucker_spectrogram.jl && echo "  SUCCESS: plucker_spectrogram.png" || echo "  FAILED"

echo ""
echo "--- Running plot9_ihara_pole_dynamics.jl ---"
julia plot9_ihara_pole_dynamics.jl && echo "  SUCCESS: plot9_ihara_pole_dynamics.png" || echo "  FAILED"

echo ""
echo "--- Running plot10_affinity_reorganisation.jl ---"
julia plot10_affinity_reorganisation.jl && echo "  SUCCESS: plot10_affinity_reorganisation.png" || echo "  FAILED"

echo ""
echo "=== Done. Check current directory for .png output files. ==="
ls -lh *.png 2>/dev/null || echo "No PNG files found."
