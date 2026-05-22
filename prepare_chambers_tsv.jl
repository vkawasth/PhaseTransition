"""
prepare_chambers_tsv.jl
Converts your existing simulation output into the chambers.tsv
format expected by the three plot scripts.

Edit the INPUT SECTION below to match your actual file names and column layout.
Then run:  julia prepare_chambers_tsv.jl

This script handles three common simulation output formats:
  Format A: SchobarNavigatorV2 output  (chambers_v2.tsv or similar)
  Format B: curved_hh2 output          (ainf_export_*.tsv)
  Format C: IharaAssociahedronBridge    (bridge_b_result.txt + separate files)
"""

using DelimitedFiles, Statistics, LinearAlgebra

# ── EDIT THIS SECTION ────────────────────────────────────────────────────────
INPUT_FORMAT = "A"          # "A", "B", or "C" — see above

# Format A: your existing chambers file
CHAMBERS_IN  = "chambers_v2.tsv"    # change to your actual filename

# Format B: ainf_export file
AINF_EXPORT  = "ainf_export_phase2.tsv"

# Format C: bridge_b_result + separate plücker file
BRIDGE_FILE  = "bridge_b_result.txt"
PLUCKER_FILE = "plucker_coords.tsv"

# Output
OUTPUT_FILE  = "chambers.tsv"
# ─────────────────────────────────────────────────────────────────────────────

function klein_constraint(q12, q13, q14, q23, q24, q34)
    abs(q12*q34 - q13*q24 + q14*q23)
end

function infer_stratum(K_val, K_max)
    # Infer Schubert stratum from Klein constraint
    # X_4 (waking): K ≈ 0 (sphere limit)
    # X_0 (crisis): K ≈ K_max
    ratio = K_val / (K_max + 1e-10)
    if ratio < 0.05;  return 4
    elseif ratio < 0.2;  return 3
    elseif ratio < 0.5;  return 2
    elseif ratio < 0.85; return 1
    else; return 0
    end
end

function detect_wall_crossings(m6_series; threshold_quantile=0.90)
    # Wall crossing = m6 spike above threshold
    thresh = quantile(m6_series, threshold_quantile)
    flags = zeros(Int, length(m6_series))
    for i in 2:length(m6_series)-1
        if m6_series[i] > thresh && m6_series[i] > m6_series[i-1] && m6_series[i] > m6_series[i+1]
            flags[i] = 1
        end
    end
    flags
end

# ── Format A: SchobarNavigatorV2 output ───────────────────────────────────────
if INPUT_FORMAT == "A"
    println("Reading Format A: $CHAMBERS_IN")
    data = readdlm(CHAMBERS_IN, '\t', skipstart=1)
    # Typical SchobarNavigatorV2 columns — adjust indices to match your file:
    # col 1: snapshot, col 2-7: q12..q34, col 8: m6_max
    # (look at header row to confirm)
    header = readline(CHAMBERS_IN)
    println("Header: $header")
    println("Columns found: $(size(data,2))")
    println("First row: $(data[1,:])")
    println()
    println("→ Edit the column indices below to match your file, then re-run.")

    # ADJUST THESE INDICES:
    col_snap  = 1   # snapshot number
    col_q12   = 2   # q12
    col_q13   = 3   # q13
    col_q14   = 4   # q14
    col_q23   = 5   # q23
    col_q24   = 6   # q24
    col_q34   = 7   # q34
    col_m6    = 8   # m6_max

    n = size(data, 1)
    snaps   = Int.(data[:, col_snap])
    q12     = Float64.(data[:, col_q12])
    q13     = Float64.(data[:, col_q13])
    q14     = Float64.(data[:, col_q14])
    q23     = Float64.(data[:, col_q23])
    q24     = Float64.(data[:, col_q24])
    q34     = Float64.(data[:, col_q34])
    m6      = Float64.(data[:, col_m6])

    K        = klein_constraint.(q12, q13, q14, q23, q24, q34)
    K_max    = maximum(K)
    stratum  = infer_stratum.(K, K_max)
    walls    = detect_wall_crossings(m6)

    out = hcat(snaps, q12, q13, q14, q23, q24, q34, m6, stratum, walls)
    header_out = "snapshot\tq12\tq13\tq14\tq23\tq24\tq34\tm6_max\tstratum_k\twall_flag"
    open(OUTPUT_FILE, "w") do io
        println(io, header_out)
        writedlm(io, out, '\t')
    end
    println("Written: $OUTPUT_FILE ($n rows)")
    println("Wall crossings detected: $(sum(walls))")
    println("K range: [$(minimum(K)), $(maximum(K))]")
    println("m6 range: [$(minimum(m6)), $(maximum(m6))]")
end

# ── Format B: ainf_export file ────────────────────────────────────────────────
if INPUT_FORMAT == "B"
    println("Reading Format B: $AINF_EXPORT")
    println("→ Inspect your ainf_export file and adjust column indices.")
    data = readdlm(AINF_EXPORT, '\t', skipstart=1)
    println("Columns: $(size(data,2)), Rows: $(size(data,1))")
    println("Header:", readline(AINF_EXPORT))
end

# ── Format C: bridge_b_result + separate files ───────────────────────────────
if INPUT_FORMAT == "C"
    println("Reading Format C: $BRIDGE_FILE + $PLUCKER_FILE")
    println("→ Adjust file names and column indices for your specific output.")
end
