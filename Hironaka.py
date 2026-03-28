import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dynamics
file_path = 'deepseek_python_20260328_d78137.py'
with open(file_path, 'r') as f:
    code = f.read()

namespace = {}
exec(code, namespace)
dyn_class = namespace['FullGraphDynamics']

# 1. Baseline: The actual simulation (with the resolution/blow-up)
dyn_full = dyn_class()
t_full, C_full, qA_full, qB_full, HH1_full, HH2_full = dyn_full.simulate()

# Singularity index
idx_254 = np.argmin(np.abs(t_full - 2.54))
idx_604 = np.argmin(np.abs(t_full - 6.04))

# Prepare Lookahead Data
lookahead_t = t_full[idx_254:idx_604+1]
n_steps = len(lookahead_t)

# Scenario A: No Blow-up (Algebraic Dead-end)
# Without the 'blow-up' resolution, the system remains in a 
# state of decay and high obstruction.
C_no = np.zeros((3, n_steps))
HH2_no = np.zeros(n_steps)

# Starting state at t=2.54
C_no[:, 0] = C_full[:, idx_254]
HH2_no[0] = HH2_full[idx_254]

# Step through and simulate the "spectral dead-end"
# In this scenario, we don't have the "exceptional" generators (recovery dose).
dt = 0.02 # Assuming dt is consistent with the simulation
for i in range(1, n_steps):
    # Natural decay without recovery (simulated as the 'unresolved' branch)
    # consciousness decays toward zero as the 'gluing' is broken.
    for node in range(3):
        C_no[node, i] = C_no[node, i-1] * 0.99
    # HH2 grows or stays high because the obstruction is not cleared.
    HH2_no[i] = HH2_no[i-1] * 1.01

# Scenario B: Resolved (The Actual Full Simulation)
C_res = C_full[:, idx_254:idx_604+1]
HH2_res = HH2_full[idx_254:idx_604+1]

# Plotting the Comparison
plt.figure(figsize=(12, 6))

# Subplot 1: Consciousness Recovery
plt.subplot(1, 2, 1)
plt.plot(lookahead_t, np.mean(C_res, axis=0), 'b-', label='With Blow-up (Resolution)')
plt.plot(lookahead_t, np.mean(C_no, axis=0), 'r--', label='No Blow-up (Dead-end)')
plt.axvline(2.54, color='k', linestyle=':', label='Singularity (t=2.54)')
plt.title('Consciousness Locus Resolution')
plt.xlabel('Time (s)')
plt.ylabel('Mean Consciousness')
plt.legend()

# Subplot 2: HH2 Obstruction
plt.subplot(1, 2, 2)
plt.plot(lookahead_t, HH2_res, 'b-', label='With Blow-up (Resolved)')
plt.plot(lookahead_t, HH2_no, 'r--', label='No Blow-up (Obstructed)')
plt.axvline(2.54, color='k', linestyle=':', label='Singularity (t=2.54)')
plt.yscale('log')
plt.title('HH2 Singular Locus Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Obstruction Intensity (HH2)')
plt.legend()

plt.tight_layout()
plt.savefig('blowup_lookahead_test.png')

# Output data for user
res_df = pd.DataFrame({
    'time': lookahead_t,
    'C_resolved': np.mean(C_res, axis=0),
    'C_obstructed': np.mean(C_no, axis=0),
    'HH2_resolved': HH2_res,
    'HH2_obstructed': HH2_no
})
res_df.to_csv('lookahead_resolution_data.csv', index=False)

print("Lookahead Analysis Complete.")
print(f"At t=6.04s:")
print(f"  Resolved Consciousness: {res_df['C_resolved'].iloc[-1]:.4f}")
print(f"  Obstructed Consciousness: {res_df['C_obstructed'].iloc[-1]:.4f}")
print(f"  Resolved HH2: {res_df['HH2_resolved'].iloc[-1]:.4f}")
print(f"  Obstructed HH2: {res_df['HH2_obstructed'].iloc[-1]:.4f}")