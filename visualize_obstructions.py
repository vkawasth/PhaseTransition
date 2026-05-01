import json
import glob
import os
import re
import numpy as np
import pyvista as pv
import time
from collections import defaultdict

def symbol_to_region(sym):
    if sym.startswith('e_'): return sym[2:]
    if sym.startswith('f_'): return sym.split('_')[1]
    return "unknown"


def safe_sort_key(fname):
    match = re.search(r'(\d+\.\d+)', fname)
    return float(match.group(1)) if match else 999999999.0


def compute_differentiated_scalars(data, mesh, region_to_nodes, region_name_to_id):
    n_pts = mesh.n_points
    s_supp = np.zeros(n_pts)
    s_ideal = np.zeros(n_pts)
    s_m6   = np.zeros(n_pts)
    s_ann  = np.ones(n_pts)

    # Support + Annihilator
    for sym in data.get('support_infty', []):
        reg = symbol_to_region(sym)
        if reg in region_name_to_id:
            nodes = region_to_nodes[region_name_to_id[reg]]
            s_supp[nodes] = 1.0
            s_ann[nodes]  = 0.0

    # Prime Ideals (sectors)
    ideals = data.get('prime_higher_ideals', []) or data.get('export_ideals', [])
    for i, sector in enumerate(ideals):
        for sym in sector:
            reg = symbol_to_region(sym)
            if reg in region_name_to_id:
                nodes = region_to_nodes[region_name_to_id[reg]]
                s_ideal[nodes] = i + 1

    # m6 Obstructions
    for path, weights in data.get('m6', {}).items():
        val = sum(abs(v) for v in weights.values())
        match = re.search(r'f_([A-Za-z0-9]+)_', str(path))
        if match:
            reg = match.group(1)
            if reg in region_name_to_id:
                nodes = region_to_nodes[region_name_to_id[reg]]
                s_m6[nodes] += val

    if s_m6.max() > 0:
        s_m6 = np.log10(s_m6 + 1e-12)

    return s_supp, s_ideal, s_m6, s_ann


def run_high_contrast_dynamics(mesh_file='nodes_edges_filtered_six.vtp', delay=0.12):
    json_files = sorted(glob.glob("ainf_export_*.json"), key=safe_sort_key)
    if not json_files:
        print("No JSON files found.")
        return

    region_name_to_id = {"CA1sp": 0, "HPF": 1, "BLA": 2, "sAMY": 3, "HY": 4, "LA": 5}

    mesh = pv.read(mesh_file)
    region_to_nodes = defaultdict(list)
    for i, rid in enumerate(mesh['region_id']):
        region_to_nodes[rid].append(i)

    # ====================== Plotter ======================
    pl = pv.Plotter(shape=(2, 2), window_size=(1650, 1250))

    titles = ["Support Variety", "Prime Ideals", "m6 Obstructions (log10)", "Annihilator"]
    cmaps  = ["hot", "tab20", "magma", "bone"]
    clims  = [(0, 1), (0, 10), (0, 8), (0, 1)]

    mesh_copies = []
    actors = []

    for i in range(4):
        pl.subplot(i // 2, i % 2)
        pl.add_text(titles[i], font_size=14, color="white", position="upper_edge")

        mcopy = mesh.copy()
        mesh_copies.append(mcopy)

        # Use unique scalar name per subplot (very important!)
        scalar_name = f"scalars_{i}"
        mcopy.point_data[scalar_name] = np.zeros(mcopy.n_points, dtype=np.float32)

        actor = pl.add_mesh(
            mcopy,
            scalars=scalar_name,
            cmap=cmaps[i],
            clim=clims[i],
            show_scalar_bar=True,
            scalar_bar_args={"title": titles[i], "width": 0.4}
        )
        actors.append(actor)

    pl.link_views()
    pl.show(auto_close=False, interactive_update=True)

    print(f"Animating {len(json_files)} frames... (delay = {delay}s)")

    for jf in json_files:
        if not pl.renderer:
            print("Window closed by user.")
            break

        try:
            with open(jf, 'r') as f:
                data = json.load(f)

            s_supp, s_ideal, s_m6, s_ann = compute_differentiated_scalars(
                data, mesh, region_to_nodes, region_name_to_id
            )

            scalars_list = [s_supp, s_ideal, s_m6, s_ann]
            step_name = os.path.basename(jf).replace('ainf_export_', '').replace('.json', '')

            for i in range(4):
                pl.subplot(i // 2, i % 2)
                
                scalar_name = f"scalars_{i}"
                mesh_copies[i].point_data[scalar_name] = scalars_list[i].astype(np.float32)
                
                # Force mapper to recognize the change
                actors[i].mapper.SetScalarRange(*clims[i])
                actors[i].mapper.Modified()

            # Update time label
            pl.add_text(f"Step: {step_name}", name="time_label",
                       position="lower_right", font_size=12, color="yellow")

            # Stronger redraw for stubborn subplots
            pl.render()
            pl.update(force_redraw=True)

            time.sleep(delay)

        except Exception as e:
            print(f"Error processing {jf}: {e}")

    print("Animation finished.")
    pl.close()


if __name__ == "__main__":
    run_high_contrast_dynamics(delay=0.10)   # Try 0.15 if too fast, 0.05 if too slow