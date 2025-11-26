# intersection_builder_v3.py
from typing import List, Dict, Any, Tuple, Optional
import math
import yaml

Point = Tuple[float, float]
TolDefault = 1e-3  # tolerant distance (meters/units) to snap to boundary

def road_name_from_lane_id(lane_id: str) -> str:
    i = 0
    while i < len(lane_id) and not lane_id[i].isdigit():
        i += 1
    return lane_id[:i] if i > 0 else lane_id

def segs_from_boundaries(boundaries: List[List[List[float]]]) -> List[Tuple[Point, Point]]:
    segs = []
    for b in boundaries:
        x0, x1 = b[0]; y0, y1 = b[1]
        segs.append(((float(x0), float(y0)), (float(x1), float(y1))))
    return segs

def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))

def closest_point_on_segment(pt: Point, seg: Tuple[Point, Point]) -> Point:
    (x1,y1), (x2,y2) = seg
    vx, vy = x2-x1, y2-y1
    wx, wy = pt[0]-x1, pt[1]-y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return (x1,y1)
    t = (vx*wx + vy*wy) / seg_len2
    t = clamp(t, 0.0, 1.0)
    return (x1 + t*vx, y1 + t*vy)

def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _roundish(v: float):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return round(v,6)

def centroid_of_points(pts: List[Point]) -> Point:
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))

def build_intersection_lanes(intersection_id: str,
                                intersection_boundaries: List[List[List[float]]],
                                all_road_lane_dicts: List[Dict[str,Any]],
                                id_prefix: Optional[str] = None,
                                tol: float = TolDefault) -> Dict[str,Any]:
    """
    Robust intersection lane builder:
      - finds attachments by nearest-point-within-tol to boundary segments
      - groups by road name (prefix of lane id)
      - requires at least one incoming (exit-on-boundary) and one outgoing (enter-on-boundary) per road
      - sorts roads circularly and creates 2*N lanes: incoming -> next_outgoing and incoming -> prev_outgoing
    """
    if id_prefix is None:
        id_prefix = intersection_id

    segs = segs_from_boundaries(intersection_boundaries)

    # collect attachments per road (allow multiple attachments)
    roads: Dict[str, Dict[str, Any]] = {}
    for lane in all_road_lane_dicts:
        lid = lane['id']
        road = road_name_from_lane_id(lid)
        rec = roads.setdefault(road, {'road': road, 'incoming_pts': [], 'outgoing_pts': [], 'from_lanes': [], 'to_lanes': []})

        # check exit -> it's an incoming attachment if it is near boundary
        exit_pt = (float(lane['exit'][0]), float(lane['exit'][1]))
        closest = None
        min_d = float('inf')
        for s in segs:
            cp = closest_point_on_segment(exit_pt, s)
            d = dist(exit_pt, cp)
            if d < min_d:
                min_d = d; closest = cp
        if min_d <= tol:
            # snap to closest point (to avoid tiny mismatch)
            rec['incoming_pts'].append((closest[0], closest[1]))
            rec['from_lanes'].append(lid)

        # check enter -> outgoing attachment
        enter_pt = (float(lane['enter'][0]), float(lane['enter'][1]))
        closest = None
        min_d = float('inf')
        for s in segs:
            cp = closest_point_on_segment(enter_pt, s)
            d = dist(enter_pt, cp)
            if d < min_d:
                min_d = d; closest = cp
        if min_d <= tol:
            rec['outgoing_pts'].append((closest[0], closest[1]))
            rec['to_lanes'].append(lid)

    # build list of connected roads: need at least one incoming and outgoing attachment
    connected_roads = []
    for r in roads.values():
        if r['incoming_pts'] and r['outgoing_pts']:
            # choose representative incoming/outgoing (if multiple, pick the average)
            in_x = sum(p[0] for p in r['incoming_pts']) / len(r['incoming_pts'])
            in_y = sum(p[1] for p in r['incoming_pts']) / len(r['incoming_pts'])
            out_x = sum(p[0] for p in r['outgoing_pts']) / len(r['outgoing_pts'])
            out_y = sum(p[1] for p in r['outgoing_pts']) / len(r['outgoing_pts'])
            r['incoming'] = (in_x, in_y)
            r['outgoing'] = (out_x, out_y)
            # rep for angle sorting
            r['rep'] = ((in_x + out_x)/2.0, (in_y + out_y)/2.0)
            connected_roads.append(r)

    if not connected_roads:
        # nothing connected
        return {'lanes': []}

    # sort roads around centroid
    rep_pts = [r['rep'] for r in connected_roads]
    cx, cy = centroid_of_points(rep_pts)
    def ang(p): return math.atan2(p[1]-cy, p[0]-cx)
    connected_roads.sort(key=lambda r: (ang(r['rep']) + 2*math.pi) % (2*math.pi))

    N = len(connected_roads)
    lanes_out = []

    # for each road, create 2 lanes from its incoming to neighbor outgoings
    for i, r in enumerate(connected_roads):
        incoming = r['incoming']
        next_i = (i + 1) % N
        prev_i = (i - 1) % N
        next_out = connected_roads[next_i]['outgoing']
        prev_out = connected_roads[prev_i]['outgoing']

        # skip same-road movements (prevent immediate U-turn)
        if connected_roads[next_i]['road'] != r['road']:
            lanes_out.append({
                'centerline': [[_roundish(incoming[0]), _roundish(next_out[0])],
                               [_roundish(incoming[1]), _roundish(next_out[1])]],
                'enter': [_roundish(incoming[0]), _roundish(incoming[1])],
                'exit':  [_roundish(next_out[0]), _roundish(next_out[1])]
            })

        if connected_roads[prev_i]['road'] != r['road']:
            lanes_out.append({
                'centerline': [[_roundish(incoming[0]), _roundish(prev_out[0])],
                               [_roundish(incoming[1]), _roundish(prev_out[1])]],
                'enter': [_roundish(incoming[0]), _roundish(incoming[1])],
                'exit':  [_roundish(prev_out[0]), _roundish(prev_out[1])]
            })

    # dedupe identical centerlines (very unlikely)
    seen = set()
    final = []
    for l in lanes_out:
        key = (tuple(sorted(tuple(l['centerline'][0]))),
               tuple(sorted(tuple(l['centerline'][1]))), 
               tuple(l['enter']),
               tuple(l['exit']))
        if key in seen:
            continue
        seen.add(key)
        final.append(l)

    # assign ids
    out_lanes = []
    for idx, l in enumerate(final, start=1):
        out_lanes.append({
            'id': f"{id_prefix}{idx}",
            'centerline': l['centerline'],
            'enter': l['enter'],
            'exit': l['exit']
        })

    return {'lanes': out_lanes}