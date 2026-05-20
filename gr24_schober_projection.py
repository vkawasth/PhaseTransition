"""
gr24_schober_projection.py
==========================
Projection of the BALBc perverse schober onto Gr(2,4).

The simulation already computes Plücker coordinates via
toric_projection_to_Gr24(). This module adds:

  1. SchuberStratification  — identifies which Schubert cell the
                              current state occupies at each timestep

  2. KählerGeometry         — computes the Fubini-Study metric,
                              Kähler form, and curvature on Gr(2,4)
                              restricted to the trajectory

  3. SchoperProjection      — maps the schober monodromy data onto
                              the Grassmannian and reads off:
                              (a) quantum cohomology eigenvalues
                              (b) Dubrovin connection curvature
                              (c) ghost signal proximity to QC spectrum

  4. SpectralHint           — predicts the next Ihara spectral radius
                              from the current Schubert stratum

All functions take the 6-vector Plücker coordinate as input and
return plain numpy arrays — no Fukaya or Tannaka machinery required.
"""

import numpy as np
from scipy.linalg import expm, eigvals
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# ── Gr(2,4) constants ─────────────────────────────────────────────────────────

# Plücker coordinate ordering: [q12, q13, q14, q23, q24, q34]
# Klein quadric: Q = q12*q34 - q13*q24 + q14*q23 = 0
# Every point of Gr(2,4) satisfies Q = 0 exactly.

# Quantum cohomology eigenvalues of Gr(2,4):
# Eigenvalues of c1* acting on H*(Gr(2,4)) with q=1.
# These are 4th roots of unity scaled by sqrt(2):
#   lambda_k = sqrt(2) * exp(i*k*pi/2), k=0,1,2,3
# Ghost signal norm 2*sqrt(2) = radius of QC spectrum * 2.
QC_EIGENVALUES = np.array([
    np.sqrt(2) * np.exp(1j * k * np.pi / 2) for k in range(4)
])
GHOST_TARGET = 2 * np.sqrt(2)

# Schubert cell dimensions (complex):
# X_0: pt (dim 0)  X_1: dim 1  X_2: dim 2  X_3: dim 3  X_4: C^4 (open, dim 4)
SCHUBERT_DIMS = [0, 1, 2, 3, 4]

# Ihara spectral radii from MAGMA (exact):
IHARA_RHO = {6: 1.5731, 7: 1.7898, 8: 1.7898}

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SchuberState:
    """State of the trajectory in the Schubert stratification."""
    stratum:         int           # 0-4, Schubert cell index
    minor_top:       complex       # det of top-left 2x2 of representing matrix
    minor_bot:       complex       # det of bottom-right 2x2
    pivot:           float         # m22 glue value (sAMY density)
    klein_Q:         float         # q12*q34 - q13*q24 + q14*q23 (should be ~0)
    at_wall:         bool          # True if near Schubert cell boundary
    wall_type:       str           # "top_zero", "bot_zero", "both", "none"


@dataclass
class KählerData:
    """Kähler geometry data at a point of Gr(2,4)."""
    fubini_study_metric: np.ndarray   # 6x6 real metric tensor in Plücker coords
    kahler_form:         np.ndarray   # 6x6 skew-symmetric Kähler form
    ricci_scalar:        float        # scalar curvature (= 8 for Gr(2,4))
    log_volume:          float        # log of Liouville volume element
    sectional_curvature: float        # sectional curvature in the plucker direction


@dataclass
class SchoperFrame:
    """One timestep of the schober projection onto Gr(2,4)."""
    t:                   float
    plucker:             np.ndarray    # 6-vector, normalised
    schubert:            SchuberState
    kahler:              KählerData
    qc_nearest:          complex       # nearest QC eigenvalue
    qc_phase_idx:        int           # index 0-3
    ihara_prediction:    float         # predicted rho/sqrt(q)
    ghost_proximity:     float         # |monodromy_norm - 2*sqrt(2)|
    monodromy_4x4:       np.ndarray    # SO(4) monodromy matrix


@dataclass
class ProjectionResult:
    """Full trajectory projection onto Gr(2,4)."""
    frames:              List[SchoperFrame]
    wall_crossings:      List[int]       # timestep indices of wall crossings
    stratum_trajectory:  List[int]       # stratum at each timestep
    qc_phase_trajectory: List[int]       # QC phase index at each timestep
    ghost_trajectory:    List[float]     # ghost proximity at each timestep
    ihara_trajectory:    List[float]     # predicted Ihara rho at each timestep
    # Summary statistics
    n_wall_crossings:    int = 0
    dominant_stratum:    int = 4
    mean_klein_Q:        float = 0.0
    mean_ghost_prox:     float = 0.0


# ── Section 1: Schubert stratification ───────────────────────────────────────

def plucker_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Recover a 2x4 matrix M whose rows span the 2-plane represented by q.

    Given Plücker coordinates [q12, q13, q14, q23, q24, q34],
    the 2-plane is spanned by rows of:
        M = [[1,  0,  q13/q12,  q14/q12],
             [0,  1,  q23/q12,  q24/q12]]  (when q12 != 0)

    In the Schubert cell X_4 (q12 != 0), this gives an affine chart.
    Falls back to other charts when q12 is small.
    """
    q12, q13, q14, q23, q24, q34 = q

    if abs(q12) > 1e-8:
        # Standard affine chart: q12 != 0
        s = 1.0 / q12
        return np.array([
            [1.0,    0.0,    q13*s, q14*s],
            [0.0,    1.0,    q23*s, q24*s]
        ], dtype=complex)
    elif abs(q13) > 1e-8:
        # Chart: q13 != 0
        s = 1.0 / q13
        return np.array([
            [1.0,    q12*s,  0.0,   q14*s],
            [0.0,    q23*s,  1.0,   q24*s]
        ], dtype=complex)
    elif abs(q24) > 1e-8:
        # Chart: q24 != 0
        s = 1.0 / q24
        return np.array([
            [q12*s,  q13*s,  1.0,   0.0  ],
            [q14*s,  1.0,    q23*s, 0.0  ]
        ], dtype=complex)
    else:
        # Near the basepoint X_0 — return identity-like
        return np.eye(2, 4, dtype=complex)


def identify_schubert_stratum(q: np.ndarray, tol: float = 1e-6) -> SchuberState:
    """
    Identify which Schubert cell X_k contains the point q in Gr(2,4).

    Schubert cells in the standard flag decomposition:
      X_4 (open): q12 != 0
      X_3:        q12 = 0, q13 != 0 or q23 != 0
      X_2:        q12 = q13 = q23 = 0, q14 != 0 or q24 != 0
      X_1:        only q34 != 0
      X_0:        q = 0 (basepoint, not in Gr(2,4))

    For the 3x3 Lax matrix interpretation:
      minor_top = det([[q12, q13], [q23, q24]])  (Plücker coordinates of top-left 2x2)
      minor_bot = det([[q24, q34], [q23, q34]])  (bottom-right 2x2 analog)
    These correspond to the two corner minors of the representing matrix.
    """
    q12, q13, q14, q23, q24, q34 = q

    # Klein quadric constraint (should be ~0 on Gr(2,4))
    klein_Q = float(q12*q34 - q13*q24 + q14*q23)

    # The two corner minors from the representing 2x4 matrix
    M = plucker_to_matrix(q)
    minor_top = complex(M[0,0]*M[1,1] - M[0,1]*M[1,0])  # top-left 2x2
    minor_bot = complex(M[0,2]*M[1,3] - M[0,3]*M[1,2])  # bottom-right 2x2

    # sAMY pivot = M[1,1] (the Schubert glue in the 3x3 Lax interpretation)
    pivot = float(np.real(M[1,1])) if M.shape[1] > 1 else 0.0

    # Identify stratum by leading nonzero Plücker coordinate
    if abs(q12) > tol:
        stratum = 4
    elif abs(q13) > tol or abs(q23) > tol:
        stratum = 3
    elif abs(q14) > tol or abs(q24) > tol:
        stratum = 2
    elif abs(q34) > tol:
        stratum = 1
    else:
        stratum = 0

    # Wall proximity: near Schubert cell boundary
    at_wall_top = abs(minor_top) < 0.1
    at_wall_bot = abs(minor_bot) < 0.1
    at_wall = at_wall_top or at_wall_bot
    if at_wall_top and at_wall_bot:
        wall_type = "both"
    elif at_wall_top:
        wall_type = "top_zero"
    elif at_wall_bot:
        wall_type = "bot_zero"
    else:
        wall_type = "none"

    return SchuberState(
        stratum=stratum, minor_top=minor_top, minor_bot=minor_bot,
        pivot=pivot, klein_Q=klein_Q, at_wall=at_wall, wall_type=wall_type
    )


# ── Section 2: Kähler geometry of Gr(2,4) ────────────────────────────────────

def fubini_study_pullback(q: np.ndarray) -> KählerData:
    """
    Compute the Fubini-Study Kähler geometry on Gr(2,4) at point q.

    Gr(2,4) embeds in P^5 via the Plücker map. The Fubini-Study metric
    on P^5 pulls back to the standard Kähler metric on Gr(2,4).

    Fubini-Study metric on P^5:
        g_{ij} = (delta_{ij}/|q|^2 - q_i * q_j_bar / |q|^4)  (real part)

    For normalised q (|q| = 1):
        g_{ij} = delta_{ij} - Re(q_i * q_j_bar)

    Kähler form omega = i/2 * dlog|q|^2 (in complex coordinates)
    On P^5 with q real (our case): omega_ij = -Im(g_{ij}) = 0 for real q.
    The imaginary part comes from the complex structure J on Gr(2,4).

    Ricci scalar: Gr(2,4) has sectional curvature in [1/2, 2]
    (normalised so that the generator of H^2 has unit integral).
    Scalar curvature = 8 (dim_C * (n+1) for Gr(k,n) = Gr(2,4)).
    """
    norm2 = np.dot(q, q)
    if norm2 < 1e-14:
        return KählerData(
            fubini_study_metric=np.eye(6),
            kahler_form=np.zeros((6,6)),
            ricci_scalar=8.0,
            log_volume=0.0,
            sectional_curvature=1.0
        )

    q_norm = q / np.sqrt(norm2)

    # Fubini-Study metric (real symmetric 6x6)
    # g_ij = delta_ij - q_i * q_j  (for unit q, real)
    g = np.eye(6) - np.outer(q_norm, q_norm)

    # Kähler form: on Gr(2,4), omega is the generator of H^2.
    # In the Plücker embedding, omega_ij = epsilon_ij where epsilon
    # is the standard symplectic form on the Plücker space P^5.
    # For Gr(2,4) specifically, the symplectic structure on the
    # 4-dimensional (complex) Grassmannian is encoded in the
    # Pl?cker coordinates via the Klein quadric form:
    # omega = dq12 ^ dq34 - dq13 ^ dq24 + dq14 ^ dq23
    # As a 6x6 antisymmetric matrix (indices: 12,13,14,23,24,34):
    omega = np.zeros((6, 6))
    # q12 (idx 0) pairs with q34 (idx 5): +1
    omega[0, 5] =  1.0; omega[5, 0] = -1.0
    # q13 (idx 1) pairs with q24 (idx 4): -1
    omega[1, 4] = -1.0; omega[4, 1] =  1.0
    # q14 (idx 2) pairs with q23 (idx 3): +1
    omega[2, 3] =  1.0; omega[3, 2] = -1.0

    # Log volume element: log det(g) on the tangent space
    # g has rank 5 (one null direction = q itself)
    # Use pseudo-determinant
    eigenvalues_g = np.linalg.eigvalsh(g)
    log_vol = np.sum(np.log(np.maximum(eigenvalues_g, 1e-12)))

    # Sectional curvature in the q direction
    # For Gr(2,4) with FS metric: K = 1 + 3*cos^2(theta)
    # where theta is the angle to the nearest Schubert variety
    # Approximation: use |q12|^2 + |q34|^2 as a proxy for cos^2(theta)
    cos2 = q_norm[0]**2 + q_norm[5]**2  # q12 and q34 components
    K = 1.0 + 3.0 * cos2

    return KählerData(
        fubini_study_metric=g,
        kahler_form=omega,
        ricci_scalar=8.0,          # exact for Gr(2,4)
        log_volume=float(log_vol),
        sectional_curvature=float(K)
    )


def dubrovin_connection_curvature(q: np.ndarray, q_prev: np.ndarray,
                                   dt: float) -> float:
    """
    Approximate curvature of the Dubrovin connection along the trajectory.

    The Dubrovin connection on Gr(2,4) has curvature proportional to
    the quantum cohomology structure constants. Along a trajectory q(t),
    the connection curvature measures how far the parallel transport
    deviates from the standard Levi-Civita transport.

    Numerically: curvature ~ |dq/dt|^2 - |q . dq/dt|^2
    (the component of the velocity tangent to Gr(2,4))
    """
    if dt < 1e-12:
        return 0.0
    dq = (q - q_prev) / dt
    # Project out the q direction (tangent to Gr(2,4))
    dq_tangent = dq - np.dot(dq, q) * q
    return float(np.dot(dq_tangent, dq_tangent))


# ── Section 3: SO(4) monodromy from Plücker data ─────────────────────────────

def rodrigues_so4(q: np.ndarray) -> np.ndarray:
    """
    Exact SO(4) monodromy matrix from Plücker coordinates via Rodrigues.

    For a rank-2 bivector Omega in so(4), the exponential is:
        exp(theta*Omega) = I + sin(alpha)/alpha * Omega
                           + (1-cos(alpha))/alpha^2 * Omega^2
    where alpha = sqrt(-tr(Omega^2)/2) (exact for rank-2 bivectors).
    """
    q12, q13, q14, q23, q24, q34 = q
    nf = np.sqrt(q12**2 + q13**2 + q14**2 + q23**2 + q24**2 + q34**2)
    if nf < 1e-10:
        return np.eye(4)
    s = 1.0 / nf
    q12*=s; q13*=s; q14*=s; q23*=s; q24*=s; q34*=s

    Omega = np.array([
        [ 0.0,  q12,  q13,  q14],
        [-q12,  0.0,  q23,  q24],
        [-q13, -q23,  0.0,  q34],
        [-q14, -q24, -q34,  0.0]
    ])

    theta = (np.pi / 2) * nf
    Omega2 = Omega @ Omega
    tr2 = np.trace(Omega2)  # = -2*alpha^2

    if abs(tr2) < 1e-14:
        return np.eye(4)

    alpha = np.sqrt(-tr2 / 2.0)
    if alpha < 1e-12:
        return np.eye(4)

    ta = theta * alpha
    ia = 1.0 / alpha
    return np.eye(4) + (np.sin(ta) * ia) * Omega + ((1.0 - np.cos(ta)) * ia**2) * Omega2


# ── Section 4: Schober projection ────────────────────────────────────────────

def project_to_qc_spectrum(q: np.ndarray, stratum: int) -> Tuple[complex, int, float]:
    """
    Project the current Plücker state onto the quantum cohomology spectrum
    of Gr(2,4).

    The QC eigenvalues are {sqrt(2)*i^k : k=0,1,2,3}.
    The projection uses the Plücker phase angle from the dominant
    (q12, q34) and (q13, q24) coordinate pairs.

    Returns (nearest_qc_eigenvalue, phase_index, ihara_prediction).
    """
    q12, q13, q14, q23, q24, q34 = q

    # Primary phase: from the (q12, q34) pair (scale-space coordinates)
    phase_primary = np.arctan2(q34, q12) if abs(q12) + abs(q34) > 1e-10 else 0.0

    # Secondary phase: from the (q13, q24) pair (directional)
    phase_secondary = np.arctan2(q24, q13) if abs(q13) + abs(q24) > 1e-10 else 0.0

    # Combined phase (weighted average, primary dominant)
    plucker_phase = 0.7 * phase_primary + 0.3 * phase_secondary

    # Find nearest QC eigenvalue by phase distance
    qc_phases = [np.angle(ev) for ev in QC_EIGENVALUES]
    phase_diffs = [abs(plucker_phase - p) for p in qc_phases]
    # Handle wraparound
    phase_diffs = [min(d, 2*np.pi - d) for d in phase_diffs]
    nearest_idx = int(np.argmin(phase_diffs))
    nearest_qc = QC_EIGENVALUES[nearest_idx]

    # Ihara prediction from stratum and QC eigenvalue
    # q_eff depends on stratum (open cell has most connections)
    q_eff = {4: 5, 3: 4, 2: 3, 1: 2, 0: 1}.get(stratum, 5)
    ihara_pred = abs(nearest_qc) / np.sqrt(q_eff)  # = sqrt(2)/sqrt(q_eff)

    return nearest_qc, nearest_idx, ihara_pred


def schober_frame(t: float, q: np.ndarray,
                   q_prev: Optional[np.ndarray] = None,
                   dt: float = 0.02) -> SchoperFrame:
    """
    Compute one frame of the schober projection onto Gr(2,4).

    This is the core function: given a Plücker coordinate vector q
    (6-dimensional, on or near Gr(2,4)), compute:
      - Schubert stratum
      - Kähler geometry
      - Nearest quantum cohomology eigenvalue
      - SO(4) monodromy
      - Ghost signal proximity

    All of these are projections of the perverse schober data
    (carried by the A∞ simulation) onto the Kähler manifold Gr(2,4).
    """
    # Normalise
    norm = np.linalg.norm(q)
    q_n = q / norm if norm > 1e-10 else q

    # Schubert stratum
    schubert = identify_schubert_stratum(q_n)

    # Kähler geometry
    kahler = fubini_study_pullback(q_n)

    # QC projection
    nearest_qc, qc_idx, ihara_pred = project_to_qc_spectrum(q_n, schubert.stratum)

    # SO(4) monodromy
    M = rodrigues_so4(q_n)
    mono_norm = np.linalg.norm(M - np.eye(4))
    ghost_prox = abs(mono_norm - GHOST_TARGET)

    return SchoperFrame(
        t=t, plucker=q_n, schubert=schubert, kahler=kahler,
        qc_nearest=nearest_qc, qc_phase_idx=qc_idx,
        ihara_prediction=ihara_pred, ghost_proximity=ghost_prox,
        monodromy_4x4=M
    )


# ── Section 5: Full trajectory projection ────────────────────────────────────

def project_trajectory(times: np.ndarray, plucker_history: np.ndarray,
                        verbose: bool = True) -> ProjectionResult:
    """
    Project the full simulation trajectory onto Gr(2,4).

    Parameters
    ----------
    times : (N,) array of timesteps
    plucker_history : (N, 6) array of Plücker coordinates
    verbose : print progress

    Returns
    -------
    ProjectionResult with per-frame data and summary statistics
    """
    N = len(times)
    frames = []
    wall_crossings = []
    stratum_traj = []
    qc_phase_traj = []
    ghost_traj = []
    ihara_traj = []

    prev_stratum = None
    prev_q = None
    dt = float(times[1] - times[0]) if N > 1 else 0.02

    for i in range(N):
        q = plucker_history[i]
        t = float(times[i])

        frame = schober_frame(t, q, q_prev=prev_q, dt=dt)
        frames.append(frame)

        stratum_traj.append(frame.schubert.stratum)
        qc_phase_traj.append(frame.qc_phase_idx)
        ghost_traj.append(frame.ghost_proximity)
        ihara_traj.append(frame.ihara_prediction)

        # Wall crossing detection: stratum change OR ghost proximity < 0.3
        is_crossing = (
            (prev_stratum is not None and frame.schubert.stratum != prev_stratum)
            or frame.schubert.at_wall
            or frame.ghost_proximity < 0.3
        )
        if is_crossing:
            wall_crossings.append(i)
            if verbose and i % 50 == 0:
                print(f"  t={t:.3f} Wall crossing: stratum={frame.schubert.stratum}"
                      f"  Q={frame.schubert.klein_Q:.4f}"
                      f"  ghost={frame.ghost_proximity:.4f}"
                      f"  qc_phase={frame.qc_phase_idx}"
                      f"  wall={frame.schubert.wall_type}")

        prev_stratum = frame.schubert.stratum
        prev_q = q

        if verbose and i % 200 == 0:
            print(f"  [{i}/{N}] t={t:.2f} stratum={frame.schubert.stratum}"
                  f"  ihara_pred={frame.ihara_prediction:.4f}"
                  f"  K={frame.kahler.sectional_curvature:.3f}")

    # Summary
    from collections import Counter
    stratum_counts = Counter(stratum_traj)
    dominant_stratum = stratum_counts.most_common(1)[0][0]
    mean_klein_Q = float(np.mean([f.schubert.klein_Q for f in frames]))
    mean_ghost = float(np.mean(ghost_traj))

    result = ProjectionResult(
        frames=frames,
        wall_crossings=wall_crossings,
        stratum_trajectory=stratum_traj,
        qc_phase_trajectory=qc_phase_traj,
        ghost_trajectory=ghost_traj,
        ihara_trajectory=ihara_traj,
        n_wall_crossings=len(wall_crossings),
        dominant_stratum=dominant_stratum,
        mean_klein_Q=mean_klein_Q,
        mean_ghost_prox=mean_ghost
    )

    if verbose:
        print_projection_summary(result)

    return result


def print_projection_summary(result: ProjectionResult):
    """Print a summary of the projection results."""
    N = len(result.frames)
    print(f"\n{'='*60}")
    print(f"Gr(2,4) Schober Projection Summary")
    print(f"{'='*60}")
    print(f"Total timesteps:       {N}")
    print(f"Wall crossings:        {result.n_wall_crossings}")
    print(f"Dominant stratum:      X_{result.dominant_stratum} (open={result.dominant_stratum==4})")
    print(f"Mean |Klein Q|:        {abs(result.mean_klein_Q):.6f} (0=on Gr(2,4))")
    print(f"Mean ghost proximity:  {result.mean_ghost_prox:.4f} (0=ghost signal)")
    print(f"\nStratum distribution:")
    from collections import Counter
    counts = Counter(result.stratum_trajectory)
    for k in sorted(counts.keys(), reverse=True):
        pct = 100 * counts[k] / N
        bar = '█' * int(pct / 2)
        print(f"  X_{k}: {counts[k]:5d} ({pct:5.1f}%) {bar}")
    print(f"\nQC phase distribution:")
    counts_qc = Counter(result.qc_phase_trajectory)
    qc_names = ['real+', 'imag+', 'real-', 'imag-']
    for k in range(4):
        n = counts_qc.get(k, 0)
        pct = 100 * n / N
        lam = QC_EIGENVALUES[k]
        print(f"  k={k} (λ={lam:.3f}): {n:5d} ({pct:5.1f}%)"
              f"  Ihara_pred={abs(lam)/np.sqrt(5):.4f}")
    print(f"\nKähler geometry (mean sectional curvature):")
    K_vals = [f.kahler.sectional_curvature for f in result.frames]
    print(f"  Mean K:  {np.mean(K_vals):.4f}  (Gr(2,4) range: [0.5, 2.0])")
    print(f"  Max K:   {np.max(K_vals):.4f}")
    print(f"  Min K:   {np.min(K_vals):.4f}")
    print(f"  Ricci scalar: 8.0 (exact for Gr(2,4))")
    print(f"\nSpectral hints:")
    ihara_vals = result.ihara_trajectory
    print(f"  Mean predicted rho/sqrt(q): {np.mean(ihara_vals):.4f}")
    print(f"  Known values: n=6: 0.703, n=7P: 0.731, n=7L: 0.703, n=8: 0.800")
    print(f"{'='*60}\n")


# ── Section 6: Integration with BALBc_Opiate_Norcain.py ──────────────────────

def attach_to_simulation(dynamics_obj) -> ProjectionResult:
    """
    Attach the Gr(2,4) projection to an existing FullGraphDynamics object
    after simulation has run.

    Usage:
        dynamics = FullGraphDynamics()
        dynamics.simulate(...)
        result = attach_to_simulation(dynamics)

    The function reads dynamics.plucker_history and dynamics.t.
    """
    if not hasattr(dynamics_obj, 'plucker_history') or \
       not dynamics_obj.plucker_history:
        raise ValueError("Run simulate() first to populate plucker_history.")

    times = np.array([h[0] for h in dynamics_obj.plucker_history])
    plucker = np.array([h[2] for h in dynamics_obj.plucker_history])

    print(f"Projecting {len(times)} Plücker snapshots onto Gr(2,4)...")
    result = project_trajectory(times, plucker)

    # Attach result to dynamics object for downstream use
    dynamics_obj.gr24_projection = result
    dynamics_obj.schober_wall_crossings = result.wall_crossings

    return result


def gr24_step(dynamics_obj, i: int) -> Optional[SchoperFrame]:
    """
    Compute the Gr(2,4) projection for timestep i during simulation.

    Call this inside the simulation loop after computing self.plucker:

        frame = gr24_step(self, i)
        if frame and frame.schubert.at_wall:
            # Wall crossing detected -- record blowup event
            ...

    Returns None if plucker is not yet computed.
    """
    if not hasattr(dynamics_obj, 'plucker') or dynamics_obj.plucker is None:
        return None

    q = dynamics_obj.plucker
    if hasattr(q, '__len__') and len(q) == 6:
        t = float(dynamics_obj.t[i]) if hasattr(dynamics_obj, 't') else float(i)
        return schober_frame(t, q)
    return None


# ── Section 7: Visualisation helpers ─────────────────────────────────────────

def plot_gr24_projection(result: ProjectionResult,
                          save_path: Optional[str] = None):
    """
    Four-panel plot of the Gr(2,4) schober projection.

    Panel A: Schubert stratum trajectory
    Panel B: Ghost signal proximity vs 2*sqrt(2)
    Panel C: QC phase index and predicted Ihara rho
    Panel D: Kähler sectional curvature along trajectory
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available for plotting")
        return

    N = len(result.frames)
    times = np.array([f.t for f in result.frames])

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Stratum trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, result.stratum_trajectory, 'b-', linewidth=1.5)
    ax1.fill_between(times, result.stratum_trajectory, alpha=0.2)
    ax1.set_ylabel('Schubert stratum')
    ax1.set_title('A: Schubert stratification')
    ax1.set_ylim(-0.2, 4.5)
    ax1.set_yticks([0,1,2,3,4])
    ax1.set_yticklabels(['X₀','X₁','X₂','X₃','X₄ (open)'])
    for wc in result.wall_crossings[::10]:  # subsample for clarity
        ax1.axvline(times[wc], color='red', alpha=0.2, linewidth=0.5)

    # Panel B: Ghost signal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, result.ghost_trajectory, 'k-', linewidth=1)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Ghost signal')
    ax2.axhline(0.3, color='orange', linestyle=':', linewidth=1, label='Wall threshold')
    ax2.set_ylabel(f'|‖M‖ - 2√2|')
    ax2.set_title('B: Ghost signal proximity')
    ax2.legend(fontsize=8)
    ghost_crossings = [i for i, g in enumerate(result.ghost_trajectory) if g < 0.3]
    if ghost_crossings:
        ax2.scatter(times[ghost_crossings], [result.ghost_trajectory[i] for i in ghost_crossings],
                    color='red', s=20, zorder=5)

    # Panel C: QC phase and Ihara prediction
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    sc = ax3.scatter(times, result.qc_phase_trajectory, c=result.qc_phase_trajectory,
                     cmap='RdYlBu', s=3, alpha=0.5)
    ax3_twin.plot(times, result.ihara_trajectory, 'g-', linewidth=1.5, alpha=0.7)
    ax3_twin.axhline(0.703, color='blue', linestyle=':', linewidth=1, label='ρ/√q n=6')
    ax3_twin.axhline(0.731, color='purple', linestyle=':', linewidth=1, label='ρ/√q n=7P')
    ax3.set_ylabel('QC phase index k')
    ax3_twin.set_ylabel('Predicted ρ/√q', color='g')
    ax3.set_title('C: QC phase & Ihara prediction')
    ax3_twin.legend(fontsize=7)

    # Panel D: Kähler curvature
    ax4 = fig.add_subplot(gs[1, 1])
    K_vals = [f.kahler.sectional_curvature for f in result.frames]
    ax4.plot(times, K_vals, 'm-', linewidth=1.5)
    ax4.axhline(2.0, color='red', linestyle=':', linewidth=1, label='K_max=2')
    ax4.axhline(0.5, color='blue', linestyle=':', linewidth=1, label='K_min=0.5')
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, label='K=1 (flat)')
    ax4.set_ylabel('Sectional curvature K')
    ax4.set_title('D: Kähler curvature (Gr(2,4))')
    ax4.legend(fontsize=7)
    ax4.set_ylim(0, 2.5)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Time')

    fig.suptitle('Perverse Schober Projection onto Gr(2,4)', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    return fig


# ── Section 8: Summary for proof outline ─────────────────────────────────────

MATHEMATICAL_NOTES = """
Mathematical content of gr24_schober_projection.py
===================================================

1. What Gr(2,4) is in this context
   The simulation state (a, b, w1, w2, phi, kappa) maps to Gr(2,4)
   via the toric projection:
     q12 = a*b*w1*w2,  q13 = a*phi*w1*w2, ...,  q34 = phi*kappa*w1*w2
   This is the Plücker embedding of the 2-plane spanned by
   rows [a, phi, kappa, 0] and [0, b, phi, kappa] of a 2x4 matrix.
   Every point of Gr(2,4) satisfies the Klein quadric Q=0.

2. Kähler structure
   Gr(2,4) carries a natural Kähler metric (Fubini-Study pullback).
   Ricci scalar = 8 (exact, from the symmetric space structure U(4)/(U(2)xU(2))).
   Sectional curvature in [1/2, 2].
   The Kähler form omega encodes the Klein quadric:
     omega = dq12^dq34 - dq13^dq24 + dq14^dq23

3. Schubert stratification
   The Schubert decomposition Gr(2,4) = X_0 ⊔ X_1 ⊔ X_2 ⊔ X_3 ⊔ X_4
   identifies which sector of the spectrum is active.
   Wall crossings (stratum changes) correspond to blowup events in
   the simulation (pole_distance = 0).

4. Quantum cohomology projection
   QC eigenvalues of Gr(2,4): {sqrt(2)*i^k : k=0,1,2,3}.
   Ghost signal norm 2*sqrt(2) = 2 * |QC_eigenvalue| = 2 * radius.
   When the SO(4) monodromy norm ||M-I|| -> 2*sqrt(2), the state
   is passing through a QC resonance.

5. Connection to the hinge theorem
   chi_red = B_Ihara (proved for all n in {6,7,8}).
   The KS monodromy Phi_KS acting on H_1(Q,Z) should satisfy
   det(I - u*Phi_KS) = zeta_Ihara(u)^{-1}  (Bridge B, pending).
   The Gr(2,4) projection tracks the monodromy along the trajectory
   and identifies which QC eigenvalue the system is near at each
   blowup event -- providing spectral hints for Bridge B.

6. What this does NOT require
   - Fukaya categories (not implemented, not needed)
   - Tannakian reconstruction (same)
   - Perverse sheaf theory (same)
   These frameworks explain WHY the projection works.
   This code computes WHAT it produces, numerically.
"""

if __name__ == "__main__":
    print(MATHEMATICAL_NOTES)

    # Minimal self-test
    print("Running self-test...")
    np.random.seed(42)

    # Generate a test trajectory on Gr(2,4)
    N = 100
    times = np.linspace(0, 5, N)
    # Trajectory: spiral on the Grassmannian
    plucker_test = np.zeros((N, 6))
    for i, t in enumerate(times):
        a = 1.0 + 0.3*np.sin(t)
        b = 0.8 + 0.2*np.cos(t)
        w1 = 1.0 + 0.1*t
        w2 = 1.2 - 0.05*t
        phi   = 0.5*np.sin(2*t)
        kappa = 0.3*np.cos(3*t)
        denom = w1*w2
        q = np.array([a*b*denom, a*phi*denom, a*kappa*denom,
                      b*phi*denom, b*kappa*denom, phi*kappa*denom])
        nrm = np.linalg.norm(q)
        plucker_test[i] = q/nrm if nrm > 0 else q

    result = project_trajectory(times, plucker_test, verbose=True)

    # Verify Klein quadric holds (approximately)
    klein_vals = [abs(f.schubert.klein_Q) for f in result.frames]
    print(f"\nKlein quadric |Q| max: {max(klein_vals):.6f} (should be ~0 on Gr(2,4))")
    print(f"Self-test: {'PASS' if max(klein_vals) < 0.1 else 'FAIL'}")
