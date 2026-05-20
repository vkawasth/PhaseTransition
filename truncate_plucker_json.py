import json

with open("plucker_zeta_dense.json") as f:
    data = json.load(f)

# Truncate snapshot_indices from 801 to 800
if len(data["snapshot_indices"]) == 801:
    data["snapshot_indices"] = data["snapshot_indices"][:800]
    print(f"Truncated: 801 → 800")

with open("plucker_zeta_dense.json", "w") as f:
    json.dump(data, f)
print("Saved.")
