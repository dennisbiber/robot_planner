# makeLanes.py
from intersectionBuild import build_intersection_lanes
from roadwayBuild import build_lanes
from connectionsBuild import find_lane_connections
from pprint import pprint
import yaml
import sys


def intersection(boundaries, all_lanes):
    res = build_intersection_lanes(
        boundaries["id"],
        boundaries["boundaries"],
        all_lanes,
        id_prefix=boundaries["id"],
        tol=1e-2
    )
    return res


def load_yaml_map() -> dict:
    file = sys.argv[-1]
    with open(file, 'r') as f:
        raw = yaml.safe_load(f)
    return raw


config = load_yaml_map()

roadway_submaps = []
intersection_submaps = []

# --- Build roadway submaps ---
for item in config["submaps"]:
    if item["type"] == "roadway":
        lanes = build_lanes(item["boundaries"], id_prefix=item["id"])
        # merge into a single dictionary, not a list
        compiled = {
            "id": item["id"],
            "type": item["type"],
            "boundaries": item["boundaries"],
            "lanes": lanes["lanes"]
        }
        roadway_submaps.append(compiled)

# --- Build intersection submaps ---
# flatten all roadway lanes first
all_lanes = [lane for submap in roadway_submaps for lane in submap["lanes"]]

for item in config["submaps"]:
    if item["type"] == "intersection":
        lanes = intersection(item, all_lanes)
        compiled = {
            "id": item["id"],
            "type": item["type"],
            "boundaries": item["boundaries"],
            "lanes": lanes["lanes"]
        }
        intersection_submaps.append(compiled)

# --- Combine everything ---
all_submaps = roadway_submaps + intersection_submaps

# --- Find connections ---
connections = find_lane_connections(all_submaps)

# --- Debug preview ---
print("\nGenerated submaps summary:")
for sub in all_submaps:
    print(f"  {sub['id']}: {len(sub['lanes'])} lanes")

print("\nGenerated connections summary:")
print(f"  {len(connections['connections'])} total connections")

# --- Write output file ---
with open("mapTest.yml", "w") as file:
    yaml.dump(
        {"submaps": all_submaps, "connections": connections["connections"]},
        file,
        sort_keys=False
    )
