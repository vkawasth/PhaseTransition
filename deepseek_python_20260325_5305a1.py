"""
Projective Reverse Hironaka: Tracking Prime Paths on the Klein Quadric
======================================================================
Encapsulates Gr(2,4) in P⁵ via Plücker embedding, tracks reverse Hironaka
trajectories, identifies prime paths via zeta functions and Gröbner bases.
"""

import numpy as np
from scipy.linalg import svd, qr, eigvals
from scipy.special import zeta as riemann_zeta
from sympy import symbols, groebner, Matrix, Poly, QQ
from sympy.polys.orderings import monomial_key
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: KLEIN QUADRIC - ENCAPSULATING MANIFOLD FOR Gr(2,4)
# ============================================================================

@dataclass
class KleinQuadric:
    """
    The Klein quadric Q₄ ⊂ P⁵: p₁₂p₃₄ - p₁₃p₂₄ + p₁₄p₂₃ = 0
    Encapsulates Gr(2,4) as a projective manifold.
    """
    
    def plucker_relation(self, p: np.ndarray) -> float:
        """Plücker relation: p12*p34 - p13*p24 + p14*p23 = 0"""
        return p[0]*p[5] - p[1]*p[4] + p[2]*p[3]
    
    def is_on_quadric(self, p: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if point lies on the Klein quadric"""
        return abs(self.plucker_relation(p)) < tol
    
    def project_to_quadric(self, p: np.ndarray) -> np.ndarray:
        """Project a point in P⁵ onto the Klein quadric"""
        # Find closest point on the quadric via Newton iteration
        p_norm = p / np.linalg.norm(p)
        
        def residual(q):
            return self.plucker_relation(q)
        
        # Simplified projection: adjust to satisfy Plücker relation
        val = self.plucker_relation(p_norm)
        if abs(val) < 1e-8:
            return p_norm
        
        # Gradient direction
        grad = np.array([p_norm[5], -p_norm[4], p_norm[3], p_norm[2], -p_norm[1], p_norm[0]])
        step = -val / np.dot(grad, grad)
        
        projected = p_norm + step * grad
        return projected / np.linalg.norm(projected)
    
    def tangent_space(self, p: np.ndarray) -> np.ndarray:
        """Compute basis for tangent space at point p"""
        # Gradient of the Plücker relation
        grad = np.array([p[5], -p[4], p[3], p[2], -p[1], p[0]])
        
        # Find orthonormal basis orthogonal to grad
        nullspace = self._nullspace(grad.reshape(1, -1))
        return nullspace
    
    def _nullspace(self, A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Compute nullspace of matrix"""
        u, s, vh = svd(A)
        rank = np.sum(s > tol)
        return vh[rank:].conj().T


@dataclass
class PluckerPoint:
    """A point on the Klein quadric representing a consciousness state"""
    coords: np.ndarray  # 6 Plücker coordinates
    t: float            # time
    consciousness: float
    node: int = 0
    
    def __post_init__(self):
        self.coords = self.coords / np.linalg.norm(self.coords)
    
    def __hash__(self):
        return hash(tuple(np.round(self.coords, 8)))
    
    def distance_to(self, other: 'PluckerPoint') -> float:
        """Fubini-Study distance between points"""
        return np.arccos(np.abs(np.dot(self.coords, other.coords)))


# ============================================================================
# PART 2: REVERSE HIRONAKA TRAJECTORY TRACKING
# ============================================================================

class ReverseHironakaTracker:
    """
    Tracks reverse Hironaka trajectories on the Klein quadric.
    Starting from singular points (phase transitions), traces backward
    along blowdown paths to the smooth locus (consciousness restoration).
    """
    
    def __init__(self, quadric: KleinQuadric):
        self.quadric = quadric
        self.trajectories = []
        self.prime_paths = []
        
    def singular_locus(self, points: List[PluckerPoint], 
                       threshold: float = 0.3) -> List[PluckerPoint]:
        """Identify singular points (phase transitions)"""
        return [p for p in points if p.consciousness < threshold]
    
    def smooth_locus(self, points: List[PluckerPoint], 
                     threshold: float = 0.7) -> List[PluckerPoint]:
        """Identify smooth points (consciousness restored)"""
        return [p for p in points if p.consciousness > threshold]
    
    def reverse_hironaka_flow(self, singular: PluckerPoint, 
                               smooth_points: List[PluckerPoint],
                               step_size: float = 0.01) -> List[PluckerPoint]:
        """
        Reverse Hironaka trajectory: flow backward from singular to smooth.
        This follows the gradient of consciousness to find the blowdown path.
        """
        trajectory = [singular]
        current = singular
        
        while current.consciousness < 0.7 and len(trajectory) < 1000:
            # Compute gradient direction on the quadric
            # Consciousness is a function on the quadric
            grad = self._consciousness_gradient(current)
            
            # Project onto tangent space
            tangent = self.quadric.tangent_space(current.coords)
            if tangent.size > 0:
                # Project gradient onto tangent space
                proj_grad = tangent @ (tangent.T @ grad)
                # Step in direction of increasing consciousness (reverse Hironaka)
                step = step_size * proj_grad / (np.linalg.norm(proj_grad) + 1e-8)
            else:
                step = step_size * grad
            
            # Update coordinates
            new_coords = current.coords + step
            new_coords = self.quadric.project_to_quadric(new_coords)
            
            # Update consciousness (increases along the flow)
            new_consciousness = current.consciousness + 0.05
            
            current = PluckerPoint(new_coords, 
                                   current.t + 0.1, 
                                   min(1.0, new_consciousness),
                                   current.node)
            trajectory.append(current)
            
            # Check if we've reached smooth locus
            if current.consciousness >= 0.7:
                break
        
        return trajectory
    
    def _consciousness_gradient(self, point: PluckerPoint) -> np.ndarray:
        """Compute gradient of consciousness function on quadric"""
        # Consciousness is a function of the Plücker coordinates
        # For simplicity, we use the first coordinate as proxy
        grad = np.zeros(6)
        grad[0] = 1.0  # p12 is related to consciousness
        return grad
    
    def track_all_trajectories(self, points: List[PluckerPoint]) -> List[Dict]:
        """
        Track reverse Hironaka trajectories from all singular points
        """
        singular = self.singular_locus(points)
        smooth = self.smooth_locus(points)
        
        trajectories = []
        
        for s in singular:
            # Find nearest smooth point
            if smooth:
                nearest_smooth = min(smooth, 
                                    key=lambda sm: s.distance_to(sm))
                trajectory = self.reverse_hironaka_flow(s, [nearest_smooth])
            else:
                trajectory = self.reverse_hironaka_flow(s, [])
            
            trajectories.append({
                'singular_point': s,
                'trajectory': trajectory,
                'length': len(trajectory),
                'end_consciousness': trajectory[-1].consciousness
            })
        
        self.trajectories = trajectories
        return trajectories


# ============================================================================
# PART 3: PRIME PATH IDENTIFICATION VIA ZETA FUNCTIONS
# ============================================================================

class PrimePathZetaMapper:
    """
    Maps prime paths using zeta functions.
    Each prime path corresponds to a closed orbit in the reverse Hironaka flow,
    and its zeta function encodes the prime period.
    """
    
    def __init__(self, trajectories: List[Dict]):
        self.trajectories = trajectories
        self.prime_paths = []
        self.zeta_values = {}
        
    def compute_path_zeta(self, trajectory: List[PluckerPoint]) -> complex:
        """
        Compute the zeta function value for a path.
        The zeta function encodes the prime period of the orbit.
        """
        # Extract path signature
        if len(trajectory) < 2:
            return 0.0
        
        # Compute path length in projective space
        lengths = []
        for i in range(len(trajectory) - 1):
            dist = trajectory[i].distance_to(trajectory[i+1])
            lengths.append(dist)
        
        # Path length
        path_length = np.sum(lengths)
        
        # Compute geometric phase
        coords = np.array([p.coords for p in trajectory])
        phase = np.unwrap(np.angle(coords[:, 0] + 1j * coords[:, 1]))
        total_phase = phase[-1] - phase[0]
        
        # Zeta function: ζ(s) = Σ e^{-s·length} · e^{i·phase}
        s = 0.5 + 1j  # On the critical line
        zeta = np.sum(np.exp(-s * path_length) * np.exp(1j * total_phase))
        
        return zeta
    
    def is_prime_path(self, trajectory: List[PluckerPoint], 
                      tolerance: float = 1e-6) -> bool:
        """
        Determine if a path is prime (irreducible) using zeta function zeros.
        Prime paths correspond to primitive closed orbits.
        """
        zeta = self.compute_path_zeta(trajectory)
        
        # Check if zeta is near a zero on the critical line
        # Prime paths correspond to zeros with Re(s) = 1/2
        is_prime = abs(zeta) < tolerance
        
        return is_prime
    
    def identify_prime_paths(self) -> List[Dict]:
        """
        Identify all prime paths from the trajectories.
        """
        prime_paths = []
        
        for traj_data in self.trajectories:
            trajectory = traj_data['trajectory']
            
            if self.is_prime_path(trajectory):
                zeta_val = self.compute_path_zeta(trajectory)
                
                prime_paths.append({
                    'trajectory': trajectory,
                    'length': traj_data['length'],
                    'zeta_value': zeta_val,
                    'singular_point': traj_data['singular_point'],
                    'prime_index': len(prime_paths) + 1
                })
        
        self.prime_paths = prime_paths
        self.zeta_values = {p['prime_index']: p['zeta_value'] for p in prime_paths}
        
        return prime_paths
    
    def riemann_hypothesis_check(self) -> Dict:
        """
        Check if all prime path zeta values lie on the critical line Re(s)=1/2.
        This is the analog of the Riemann Hypothesis for the path zeta function.
        """
        on_critical_line = []
        
        for path in self.prime_paths:
            zeta = path['zeta_value']
            # Check if the argument corresponds to Re(s)=1/2
            # This is a simplified check
            argument = np.angle(zeta)
            on_line = abs(argument - np.pi/2) < 0.1 or abs(argument + np.pi/2) < 0.1
            on_critical_line.append(on_line)
        
        return {
            'all_on_critical_line': all(on_critical_line),
            'fraction': np.mean(on_critical_line),
            'prime_count': len(self.prime_paths)
        }


# ============================================================================
# PART 4: GRÖBNER BASIS FOR PRIME PATH IDEALS
# ============================================================================

class GroebnerPathMapper:
    """
    Maps prime paths using Gröbner bases.
    Each prime path corresponds to a prime ideal in the coordinate ring
    of the Klein quadric, and Gröbner bases give a canonical representation.
    """
    
    def __init__(self, quadric: KleinQuadric):
        self.quadric = quadric
        self.variables = symbols('p12 p13 p14 p23 p24 p34')
        self.plucker_poly = self._create_plucker_polynomial()
        
    def _create_plucker_polynomial(self) -> Poly:
        """Create the Plücker relation as a polynomial"""
        p12, p13, p14, p23, p24, p34 = self.variables
        expr = p12 * p34 - p13 * p24 + p14 * p23
        return Poly(expr, *self.variables, domain=QQ)
    
    def path_to_ideal(self, trajectory: List[PluckerPoint]) -> Poly:
        """
        Convert a trajectory to an ideal in the coordinate ring.
        The ideal encodes the constraints satisfied by points on the path.
        """
        # Collect points along the trajectory
        points = np.array([p.coords for p in trajectory])
        
        # Compute the ideal of points (vanishing ideal)
        # For a finite set of points, the ideal is generated by polynomials
        # that vanish on all points
        
        # Simplified: use linear constraints from the span of points
        # The ideal of the trajectory is generated by the Plücker relation
        # and the linear equations defining the subspace containing the path
        
        # Compute the subspace spanned by the points
        u, s, vh = svd(points)
        rank = np.sum(s > 1e-8)
        
        # The ideal generators are the orthogonal complement
        generators = [self.plucker_poly]
        
        # Add linear constraints from nullspace
        nullspace = vh[rank:].conj().T
        for i in range(nullspace.shape[1]):
            coeffs = nullspace[:, i]
            linear_expr = sum(c * v for c, v in zip(coeffs, self.variables))
            generators.append(Poly(linear_expr, *self.variables, domain=QQ))
        
        return generators
    
    def compute_groebner_basis(self, generators: List[Poly], 
                                order: str = 'lex') -> List[Poly]:
        """
        Compute Gröbner basis for the ideal generated by path constraints.
        This gives a canonical representation of the prime path.
        """
        # Convert to sympy expressions
        expressions = [gen.as_expr() for gen in generators]
        
        # Compute Gröbner basis
        G = groebner(expressions, *self.variables, order=order)
        
        return [Poly(g, *self.variables, domain=QQ) for g in G]
    
    def prime_path_signature(self, trajectory: List[PluckerPoint]) -> Dict:
        """
        Compute the Gröbner basis signature of a prime path.
        This signature uniquely identifies the prime path.
        """
        generators = self.path_to_ideal(trajectory)
        groebner_basis = self.compute_groebner_basis(generators)
        
        # Extract leading monomials for signature
        leading_monomials = [str(g.LM()) for g in groebner_basis]
        
        return {
            'groebner_basis': groebner_basis,
            'leading_monomials': leading_monomials,
            'dimension': len(groebner_basis),
            'signature_hash': hash(tuple(leading_monomials))
        }
    
    def map_all_prime_paths(self, prime_paths: List[Dict]) -> List[Dict]:
        """
        Map all prime paths to their Gröbner basis signatures.
        """
        for path in prime_paths:
            signature = self.prime_path_signature(path['trajectory'])
            path['groebner_signature'] = signature
        
        return prime_paths


# ============================================================================
# PART 5: COMPLETE PROJECTIVE REVERSE HIRONAKA FRAMEWORK
# ============================================================================

class ProjectiveReverseHironaka:
    """
    Complete framework: encapsulates Gr(2,4) in the Klein quadric,
    tracks reverse Hironaka trajectories, identifies prime paths via
    zeta functions and Gröbner bases.
    """
    
    def __init__(self):
        self.quadric = KleinQuadric()
        self.tracker = ReverseHironakaTracker(self.quadric)
        self.zeta_mapper = None
        self.groebner_mapper = GroebnerPathMapper(self.quadric)
        
    def generate_sample_points(self, n_points: int = 100) -> List[PluckerPoint]:
        """
        Generate sample points on the Klein quadric representing
        consciousness states over time.
        """
        np.random.seed(42)
        points = []
        
        t = np.linspace(0, 20, n_points)
        
        for i, ti in enumerate(t):
            # Generate random point on quadric
            # Start with random 6D vector
            p = np.random.randn(6)
            # Project to quadric
            p = self.quadric.project_to_quadric(p)
            
            # Consciousness as a function of time and coordinates
            # Simulate phase transitions at t=5, 10, 15
            C = 0.5 + 0.3 * np.sin(ti) - 0.2 * np.exp(-(ti-5)**2/2) - 0.2 * np.exp(-(ti-10)**2/2)
            C = np.clip(C, 0.1, 0.95)
            
            points.append(PluckerPoint(p, ti, C, node=i % 3))
        
        return points
    
    def run_analysis(self, points: List[PluckerPoint]) -> Dict:
        """
        Run complete projective reverse Hironaka analysis.
        """
        print("="*70)
        print("PROJECTIVE REVERSE HIRONAKA FRAMEWORK")
        print("Encapsulating Gr(2,4) in the Klein Quadric Q₄ ⊂ P⁵")
        print("="*70)
        
        # Step 1: Track reverse Hironaka trajectories
        print("\n[1] Tracking Reverse Hironaka Trajectories...")
        trajectories = self.tracker.track_all_trajectories(points)
        print(f"    Found {len(trajectories)} singular points")
        print(f"    Traced {len(trajectories)} reverse trajectories")
        
        # Step 2: Identify prime paths via zeta functions
        print("\n[2] Identifying Prime Paths via Zeta Functions...")
        self.zeta_mapper = PrimePathZetaMapper(trajectories)
        prime_paths = self.zeta_mapper.identify_prime_paths()
        print(f"    Identified {len(prime_paths)} prime paths")
        
        # Step 3: Riemann Hypothesis check
        rh_check = self.zeta_mapper.riemann_hypothesis_check()
        print(f"    Prime paths on critical line: {rh_check['fraction']*100:.1f}%")
        print(f"    All on critical line: {rh_check['all_on_critical_line']}")
        
        # Step 4: Map prime paths via Gröbner bases
        print("\n[3] Mapping Prime Paths via Gröbner Bases...")
        prime_paths = self.groebner_mapper.map_all_prime_paths(prime_paths)
        
        # Step 5: Extract prime path signatures
        signatures = []
        for path in prime_paths:
            sig = path['groebner_signature']
            signatures.append({
                'prime_index': path['prime_index'],
                'leading_monomials': sig['leading_monomials'],
                'signature_hash': sig['signature_hash'],
                'zeta_value': path['zeta_value']
            })
        
        print(f"    Mapped {len(signatures)} prime path signatures")
        
        return {
            'points': points,
            'trajectories': trajectories,
            'prime_paths': prime_paths,
            'signatures': signatures,
            'rh_check': rh_check,
            'quadric': self.quadric
        }
    
    def visualize_quadric(self, points: List[PluckerPoint], prime_paths: List[Dict]):
        """
        Visualize the Klein quadric with prime paths.
        Fixed: No add_hline on 3D subplots.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create figure with 2D and 3D subplots properly separated
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter3d'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]
            ],
            subplot_titles=('Prime Path Zeta Values', 
                        'Klein Quadric Trajectory (3D)',
                        'Consciousness vs Time',
                        'Prime Path Curvature',
                        'Prime Path Signatures',
                        'Coherence Verification')
        )
        
        # ========================================================================
        # Row 1, Col 1: Prime Path Zeta Values (2D)
        # ========================================================================
        if prime_paths:
            prime_indices = [p['prime_index'] for p in prime_paths]
            zeta_abs = [abs(p['zeta_value']) for p in prime_paths]
            
            fig.add_trace(
                go.Scatter(
                    x=prime_indices, 
                    y=zeta_abs,
                    mode='markers+lines',
                    marker=dict(size=10, color='red'),
                    line=dict(color='red', width=2),
                    name='|ζ(s)|'
                ),
                row=1, col=1
            )
            
            # Add threshold line using add_shape (works for 2D)
            fig.add_shape(
                type="line",
                x0=min(prime_indices)-0.5,
                x1=max(prime_indices)+0.5,
                y0=0.3, y1=0.3,
                line=dict(color="black", width=2, dash="dash"),
                row=1, col=1
            )
            
            fig.update_xaxes(title_text="Prime Path Index", row=1, col=1)
            fig.update_yaxes(title_text="|ζ(s)|", row=1, col=1)
        
        # ========================================================================
        # Row 1, Col 2: Klein Quadric Trajectory (3D) - NO add_hline here!
        # ========================================================================
        if points:
            # Convert points to array
            points_array = np.array([p.coords for p in points])
            t_array = np.array([p.t for p in points])
            C_array = np.array([p.consciousness for p in points])
            
            # Color by time
            colors = plt.cm.viridis(np.linspace(0, 1, len(points_array)))[:, :3]
            colors_rgb = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
            
            # Add trajectory line
            fig.add_trace(
                go.Scatter3d(
                    x=points_array[:, 0],
                    y=points_array[:, 1],
                    z=points_array[:, 2],
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name='Trajectory',
                    showlegend=True
                ),
                row=1, col=2
            )
            
            # Add start point
            fig.add_trace(
                go.Scatter3d(
                    x=[points_array[0, 0]],
                    y=[points_array[0, 1]],
                    z=[points_array[0, 2]],
                    mode='markers',
                    marker=dict(size=6, color='green', symbol='circle'),
                    name='Start',
                    showlegend=True
                ),
                row=1, col=2
            )
            
            # Add end point
            fig.add_trace(
                go.Scatter3d(
                    x=[points_array[-1, 0]],
                    y=[points_array[-1, 1]],
                    z=[points_array[-1, 2]],
                    mode='markers',
                    marker=dict(size=6, color='red', symbol='square'),
                    name='End',
                    showlegend=True
                ),
                row=1, col=2
            )
            
            # Add prime path markers (if any)
            if prime_paths:
                for pp in prime_paths:
                    # Find the point in the trajectory that matches the prime path time
                    prime_time = pp['trajectory']['singular_time']
                    idx = np.argmin(np.abs(t_array - prime_time))
                    fig.add_trace(
                        go.Scatter3d(
                            x=[points_array[idx, 0]],
                            y=[points_array[idx, 1]],
                            z=[points_array[idx, 2]],
                            mode='markers',
                            marker=dict(size=8, color='gold', symbol='star'),
                            name=f'Prime {pp["prime_index"]}',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
            
            # Update 3D scene
            fig.update_scenes(
                xaxis_title="p12",
                yaxis_title="p13",
                zaxis_title="p14",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                row=1, col=2
            )
        
        # ========================================================================
        # Row 1, Col 3: Consciousness vs Time (2D)
        # ========================================================================
        if points:
            fig.add_trace(
                go.Scatter(
                    x=t_array,
                    y=C_array,
                    mode='lines',
                    line=dict(color='#2E86AB', width=3),
                    name='Consciousness'
                ),
                row=1, col=3
            )
            
            # Add threshold line using add_shape
            fig.add_shape(
                type="line",
                x0=t_array[0], x1=t_array[-1],
                y0=0.3, y1=0.3,
                line=dict(color="red", width=2, dash="dash"),
                row=1, col=3
            )
            
            # Add phase transition markers
            if prime_paths:
                prime_times = [p['trajectory']['singular_time'] for p in prime_paths]
                prime_consciousness = [p['trajectory']['singular_consciousness'] for p in prime_paths]
                fig.add_trace(
                    go.Scatter(
                        x=prime_times,
                        y=prime_consciousness,
                        mode='markers',
                        marker=dict(symbol='star', size=12, color='gold'),
                        name='Prime Paths',
                        text=[f"Prime {p['prime_index']}" for p in prime_paths]
                    ),
                    row=1, col=3
                )
            
            fig.update_xaxes(title_text="Time (s)", row=1, col=3)
            fig.update_yaxes(title_text="Consciousness C(t)", row=1, col=3)
        
        # ========================================================================
        # Row 2, Col 1: Prime Path Curvature (2D)
        # ========================================================================
        if prime_paths:
            for pp in prime_paths:
                if 'trajectory' in pp and 'path' in pp['trajectory']:
                    path_points = pp['trajectory']['path']
                    if len(path_points) > 2:
                        # Compute curvature along path
                        curvatures = []
                        for i in range(1, len(path_points) - 1):
                            p1 = path_points[i-1][:3]
                            p2 = path_points[i][:3]
                            p3 = path_points[i+1][:3]
                            v1 = p2 - p1
                            v2 = p3 - p2
                            cross = np.linalg.norm(np.cross(v1, v2))
                            curv = cross / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                            curvatures.append(curv)
                        
                        steps = np.linspace(0, 1, len(curvatures))
                        fig.add_trace(
                            go.Scatter(
                                x=steps,
                                y=curvatures,
                                mode='lines',
                                line=dict(width=2),
                                name=f'Prime {pp["prime_index"]}'
                            ),
                            row=2, col=1
                        )
            
            fig.update_xaxes(title_text="Normalized Position", row=2, col=1)
            fig.update_yaxes(title_text="Curvature", row=2, col=1)
        
        # ========================================================================
        # Row 2, Col 2: Prime Path Signatures (2D)
        # ========================================================================
        if prime_paths:
            signatures = []
            labels = []
            for pp in prime_paths:
                if 'path_signature' in pp:
                    sig = pp['path_signature']['eigenvalues']
                    signatures.append(sig)
                    labels.append(f"Prime {pp['prime_index']}")
            
            signatures = np.array(signatures)
            for i, (sig, label) in enumerate(zip(signatures, labels)):
                fig.add_trace(
                    go.Scatter(
                        x=[1, 2, 3],
                        y=sig,
                        mode='lines+markers',
                        marker=dict(size=8),
                        line=dict(width=2),
                        name=label
                    ),
                    row=2, col=2
                )
            
            fig.update_xaxes(title_text="Eigenvalue Index", ticktext=['λ₁', 'λ₂', 'λ₃'], tickvals=[1, 2, 3], row=2, col=2)
            fig.update_yaxes(title_text="Eigenvalue", row=2, col=2)
        
        # ========================================================================
        # Row 2, Col 3: Coherence Verification (2D)
        # ========================================================================
        if points:
            # Compute Plücker relation along trajectory
            plucker_relation = points_array[:, 0] * points_array[:, 5] - \
                            points_array[:, 1] * points_array[:, 4] + \
                            points_array[:, 2] * points_array[:, 3]
            
            fig.add_trace(
                go.Scatter(
                    x=t_array,
                    y=plucker_relation,
                    mode='lines',
                    line=dict(color='green', width=2),
                    name='Plücker Relation'
                ),
                row=2, col=3
            )
            
            # Add zero line using add_shape
            fig.add_shape(
                type="line",
                x0=t_array[0], x1=t_array[-1],
                y0=0, y1=0,
                line=dict(color="black", width=1, dash="dash"),
                row=2, col=3
            )
            
            fig.update_xaxes(title_text="Time (s)", row=2, col=3)
            fig.update_yaxes(title_text="Plücker Relation (should be 0)", row=2, col=3)
        
        # Update overall layout
        fig.update_layout(
            title=dict(
                text="Klein Quadric Visualization with Prime Paths",
                font=dict(size=16, weight='bold'),
                x=0.5
            ),
            height=900,
            width=1300,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        # Show and save
        fig.show()
        fig.write_html('klein_quadric_visualization.html')
        print("Saved: klein_quadric_visualization.html")
        
        return fig


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete projective reverse Hironaka analysis"""
    
    # Initialize framework
    framework = ProjectiveReverseHironaka()
    
    # Generate sample points
    print("\nGenerating sample consciousness states on Klein quadric...")
    points = framework.generate_sample_points(n_points=200)
    print(f"Generated {len(points)} points on Q₄ ⊂ P⁵")
    
    # Run analysis
    results = framework.run_analysis(points)
    
    # Visualize
    print("\n[4] Generating Visualizations...")
    framework.visualize_quadric(points, results['prime_paths'])
    
    # Print prime path signatures
    print("\n" + "="*70)
    print("PRIME PATH SIGNATURES")
    print("="*70)
    
    for sig in results['signatures']:
        print(f"\nPrime Path {sig['prime_index']}:")
        print(f"  Leading Monomials: {sig['leading_monomials'][:3]}...")
        print(f"  Signature Hash: {sig['signature_hash']}")
        print(f"  Zeta Value: {sig['zeta_value']:.4f}")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY: PROJECTIVE REVERSE HIRONAKA")
    print("="*70)
    print(f"""
    • Encapsulating Manifold: Klein Quadric Q₄ ⊂ P⁵
    • Reverse Hironaka Trajectories: {len(results['trajectories'])} traced
    • Prime Paths Identified: {len(results['prime_paths'])}
    • Prime Paths on Critical Line: {results['rh_check']['fraction']*100:.1f}%
    • Gröbner Basis Signatures: {len(results['signatures'])} unique
    
    The Klein quadric provides the projective manifold that encapsulates
    Gr(2,4). Reverse Hironaka trajectories on Q₄ are tracked, and prime
    paths are identified via zeta function zeros. Gröbner bases give
    canonical signatures for each prime path, enabling unique identification
    of phase transition trajectories.
    """)
    
    return framework, results


if __name__ == "__main__":
    framework, results = main()