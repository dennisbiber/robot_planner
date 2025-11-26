import math

def find_lane_connections(submaps, tol=0.1):
    """
    Given a list of submaps (each with 'id' and 'lanes'),
    returns a dictionary like:
      {'connections': [{'from': 'A2', 'to': 'B5'}, ...]}
    """

    def dist(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    # Build a lookup of all lane entrances
    enters = []
    for submap in submaps:
        print(submap)
        if "lanes" in submap.keys():
            for lane in submap["lanes"]:
                enters.append({
                    "id": lane["id"],
                    "enter": tuple(map(float, lane["enter"]))
                })

    # Compare each exit to all enters
    connections = []
    for submap in submaps:
        for lane in submap["lanes"]:
            exit_pt = tuple(map(float, lane["exit"]))
            matches = []
            for target in enters:
                d = dist(exit_pt, target["enter"])
                if d <= tol and target["id"] != lane["id"]:
                    matches.append(target["id"])
            # If multiple within tolerance, add all
            for m in matches:
                connections.append({"from": lane["id"], "to": m})

    return {"connections": connections}