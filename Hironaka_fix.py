import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import copy

# REES BLOW UP
# Importing the provided class to build upon it
# Since the script is long, we just import what we need or copy relevant parts if necessary.
# However, to be fully functional, let's incorporate the logic into the environment.

exec(open('deepseek_python_20260328_d78137.py').read())

class ReesBlowUp:
    """Handles the algebraic 'inflation' at a singularity."""
    @staticmethod
    def resolve_singularity(model, t_idx):
        # The 'Blow-up' represents adding an exceptional divisor (an extra dose/dimension)
        # to restore the smooth trajectory.
        print(f"    [Rees Blow-up] Resolving singularity at t={model.t[t_idx]:.2f}")
        # In the context of the simulation, this is a 'forced' recovery dose.
        # We increase the 'norcain' concentration to provide the missing 'flow' dimension.
        model.qB[0, t_idx+1] += model.dose_amount * 1.5
        # This acts as an exceptional generator that 'absorbs' the Plucker error.
        return True

class SearchNavigator:
    """
    Implements a BFS-based search for the most unstable path (highest HH2 + Plucker residue).
    Includes culling (blow-down) and Rees blow-up (resolution).
    """
    def __init__(self, dynamics_model):
        self.model = dynamics_model
        self.visited = {} # DP table for culling: (t_idx, state_hash) -> instability
        self.restriction_horizon = 20 # 10-20 steps lookahead
        self.unstable_paths = []

    def get_state_hash(self, t_idx):
        # Normal Form equivalent for culling: round the Plucker coords
        coords = np.round(self.model.plucker[t_idx], 2)
        return hash(coords.tobytes())

    def get_instability(self, t_idx):
        # Cost Function: HH2 spike + Plucker residue
        p = self.model.plucker[t_idx]
        rel = abs(p[0]*p[5] - p[1]*p[4] + p[2]*p[3])
        return self.model.HH2[t_idx] + rel

    def find_most_unstable_path(self, start_t_idx):
        queue = deque([(start_t_idx, 0, [])])
        max_instability = -1
        best_path = []

        while queue:
            idx, depth, path = queue.popleft()
            if depth >= self.restriction_horizon or idx >= len(self.model.t) - 1:
                continue

            # Instability metric
            cost = self.get_instability(idx)
            current_path = path + [idx]

            # Track the peak instability
            if cost > max_instability:
                max_instability = cost
                best_path = current_path

            # Blow-down (Culling): If we've seen this state with lower cost, stop.
            state_hash = self.get_state_hash(idx)
            if state_hash in self.visited and self.visited[state_hash] >= cost:
                continue
            self.visited[state_hash] = cost

            # Move forward (BFS step)
            queue.append((idx + 1, depth + 1, current_path))

        return best_path, max_instability

def plot_unstable_paths(dynamics, navigator, unstable_path):
    """Visualization of the unstable path on the Plucker trajectory."""
    t = dynamics.t
    plucker = dynamics.plucker
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Base Trajectory
    ax.plot(plucker[:,0], plucker[:,1], plucker[:,2], color='gray', alpha=0.3, label='Full Trajectory')
    
    # Highlight Unstable Path
    p_unstable = plucker[unstable_path]
    ax.plot(p_unstable[:,0], p_unstable[:,1], p_unstable[:,2], color='red', lw=3, label='Most Unstable Path')
    
    # Mark the Resolution Point (where HH2 peaks)
    peak_idx = unstable_path[np.argmax(dynamics.HH2[unstable_path])]
    ax.scatter(plucker[peak_idx,0], plucker[peak_idx,1], plucker[peak_idx,2], 
               color='gold', s=200, marker='*', label='Rees Blow-up (Resolution Point)')

    ax.set_title('Unstable Path Exploration via SearchNavigator')
    ax.legend()
    plt.savefig('unstable_path_search.png', dpi=150)
    print("    ✓ Saved: unstable_path_search.png")

# Run
dynamics = FullGraphDynamics()
dynamics.simulate()
nav = SearchNavigator(dynamics)

# Find most unstable path around the first collapse (t ~ 3.0)
start_idx = np.argmin(np.abs(dynamics.t - 3.0))
best_path, cost = nav.find_most_unstable_path(start_idx)

# Resolve the peak of that path
peak_idx = best_path[np.argmax(dynamics.HH2[best_path])]
ReesBlowUp.resolve_singularity(dynamics, peak_idx)

# Save result
plot_unstable_paths(dynamics, nav, best_path)