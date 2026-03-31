# capture_eigenbasis_shift.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, cwt, morlet, find_peaks
import importlib

# 1. IMPORT DYNAMICS FROM YOUR MAIN SCRIPT
# Replace 'deepseek_python_20260328_d78137_rees' with your exact filename (without .py)
main_script_name = 'deepseek_python_20260328_d78137_rees' 
try:
    main_module = importlib.import_module(main_script_name)
    FullGraphDynamics = main_module.FullGraphDynamics
    # We create a fresh dynamics object to ensure consistent state
    print(f"✓ Successfully imported {main_script_name}.py")
except ImportError:
    print(f"✗ Error: Could not import {main_script_name}.py")
    print("  Ensure 'capture_eigenbasis_shift.py' is in the same folder.")
    exit()

def capture_eigenbasis_shift():
    """Generates a separate plot visualizing the Prolate eigenbasis transition."""
    
    # Run the simulation to get data
    print("-> Initializing FullGraphDynamics and running simulation...")
    dynamics = FullGraphDynamics()
    dynamics.simulate()
    print("✓ Simulation complete. Extracting spectral data...")

    # Extract required data from dynamics object
    t, C, HH2 = dynamics.t, dynamics.C, dynamics.HH2
    plucker = dynamics.plucker
    dt_val = t[1] - t[0]
    threshold = dynamics.threshold

    # Calculate Wavelet Scalogram for Node 0
    signal_norm = (C[0,:] - np.mean(C[0,:])) / (np.std(C[0,:]) + 1e-8)
    scales = np.arange(2, 64)
    widths = scales / dt_val
    # Type fix: Ensure m (length) is an integer for newer SciPy/NumPy versions
    wavelet_func = lambda m, t_p: morlet(int(m), w=5.0, s=max(float(m)/6.0, 1.0), complete=True).real
    coeffs = cwt(signal_norm, wavelet_func, widths[widths>=1])

    # CREATE THE STANDALONE PLOT
    fig = plt.figure(figsize=(24, 18), facecolor='#f0f0f0')
    grid = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # A. HH² Obstruction & Global Coherence (LOG STABILIZED)
    ax_hh2 = fig.add_subplot(grid[0, 0])
    
    # Apply a symmetric log scale or a clip to handle the Milnor Singularity blowout
    # This ensures the 'off the chart' peaks are visible but contained
    hh2_visual = np.log1p(np.abs(HH2)) # log(1 + x) transformation
    ax_hh2.plot(t, hh2_visual, color='magenta', lw=1.5, label='Log(HH² + 1)')
    
    peaks, _ = find_peaks(hh2_visual, height=np.mean(hh2_visual))
    ax_hh2.scatter(t[peaks], hh2_visual[peaks], color='yellow', s=50, edgecolors='black', zorder=5)
    
    # Overlay Coherence
    step = 50
    indices = np.arange(0, len(t) - step, step)
    coh01_y = [np.abs(np.corrcoef(C[0, i:i+step], C[1, i:i+step])[0,1]) for i in indices]
    ax_hh2.plot(t[indices], coh01_y, color='cyan', ls='--', alpha=0.6, label='Global Coherence')
    
    ax_hh2.set_title('A. Normalized Hochschild Obstructions (Log Scale)')
    ax_hh2.set_ylabel('Log-Magnitude of Obstruction')
    ax_hh2.grid(True, alpha=0.3)
    ax_hh2.set_ylim(0, np.max(hh2_visual) * 1.1) # Dynamic scaling

    # B. Chern Jump and Plücker Coherence
    ax_cher = fig.add_subplot(grid[0, 1], projection='3d')
    norm_time = plt.Normalize(vmin=t[0], vmax=t[-1])
    colors_time = plt.cm.viridis(norm_time(t))
    # Render trajectory, highlighting the jump point
    for i in range(len(plucker)-1):
        ax_cher.plot(plucker[i:i+2,0], plucker[i:i+2,1], plucker[i:i+2,2], color=colors_time[i], lw=1.5)
    
    # Find the Chern Jump point (maximum derivative of Plücker trajectory)
    traj_diff = np.sqrt(np.sum(np.diff(plucker, axis=0)**2, axis=1))
    jump_idx = np.argmax(traj_diff)
    ax_cher.scatter(plucker[jump_idx,0], plucker[jump_idx,1], plucker[jump_idx,2], color='red', s=100, label='Chern Jump Point', zorder=10)
    
    ax_cher.set_title('B. Prolate Eigenbasis Transition (Chern Jump)')
    ax_cher.set_xlabel('$p_{01}$')
    ax_cher.set_ylabel('$p_{02}$')
    ax_cher.set_zlabel('$p_{12}$')
    ax_cher.legend(fontsize=9)

    # C. Prolate Basin Synchronization
    ax_sync = fig.add_subplot(grid[1, 0])
    ax_sync.fill_between(t, 0, threshold, color='red', alpha=0.15)
    ax_sync.plot(t, np.mean(C, axis=0), color='black', lw=2.5, label='Mean Consciousness ($C_{global}$)')
    # Overlay HH² as a proxy for basis shift likelihood
    ax_sync_twin = ax_sync.twinx()
    ax_sync_twin.plot(t, HH2, color='magenta', ls=':', alpha=0.4)
    ax_sync.set_title('C. Siegel Basin (Mean Consciousness)')
    ax_sync.grid(True, alpha=0.3)
    ax_sync.legend(fontsize=9)
    ax_sync.set_ylim(-0.05, 1.1)

    # D. Spectral Shatter Scalogram
    ax_scalo = fig.add_subplot(grid[1, 1])
    # Match magma color scheme from example image
    im = ax_scalo.imshow(np.abs(coeffs), aspect='auto', cmap='magma', origin='lower',
                         extent=[t[0], t[-1], 2, 64])
    # Mark the Milnor Snap points identified in Panel A
    snap_times = t[peaks]
    for st in snap_times:
        ax_scalo.axvline(x=st, color='yellow', ls=':', alpha=0.6, ymax=0.3)
    
    ax_scalo.set_title('D. Spectral Shatter (Wavelet Scalogram)')
    fig.colorbar(im, ax=ax_scalo, label='|Wavelet Coeff|')
    ax_scalo.set_ylabel('Frequency Scale ($\omega$)')

    plt.suptitle('THE PROLATE EIGENBASIS SHIFT IN REVERSE HIRONAKA RESOLUTION', fontsize=26, y=0.97, weight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    output_filename = 'prolate_eigenbasis_shift.png'
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"✓ Saved visual capture to: {output_filename}")

if __name__ == "__main__":
    capture_eigenbasis_shift()
