from typing import List, Tuple, Dict, Any
import math

Point = Tuple[float, float]
Seg = Tuple[Point, Point]


def segs_from_boundaries(boundaries: List[List[List[float]]]) -> List[Seg]:
    segs = []
    for b in boundaries:
        x0, x1 = b[0]
        y0, y1 = b[1]
        segs.append(((float(x0), float(y0)), (float(x1), float(y1))))
    return segs


def _roundish(v: float) -> float:
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return float(round(v, 6))


def build_polygon_ordered(segs: List[Seg]) -> List[Point]:
    """Link segment endpoints into an ordered polygon (handles non-rectangular)."""
    adj = {}
    for (x1, y1), (x2, y2) in segs:
        a = (x1, y1)
        b = (x2, y2)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = next(iter(adj))
    poly = [start]
    prev = None
    cur = start
    while True:
        neighs = adj[cur]
        nxt = None
        if prev is None:
            nxt = neighs[0]
        else:
            if len(neighs) == 1:
                nxt = neighs[0]
            else:
                nxt = neighs[0] if neighs[0] != prev else neighs[1]
        if nxt == start:
            break
        poly.append(nxt)
        prev, cur = cur, nxt
        if len(poly) > 100:
            break
    return poly


def bbox(poly: List[Point]):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def find_nearest_horizontal_edge(poly: List[Point], y_target: float) -> Tuple[Point, Point]:
    """Find the edge with average y closest to y_target (tolerant of slanted edges)."""
    n = len(poly)
    best = None
    best_d = float('inf')
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        y_avg = (a[1] + b[1]) / 2.0
        d = abs(y_avg - y_target)
        if d < best_d:
            best_d = d
            best = (a, b)
    # ensure left→right order
    if best[0][0] <= best[1][0]:
        return best
    return best[1], best[0]


def find_nearest_vertical_edge(poly: List[Point], x_target: float) -> Tuple[Point, Point]:
    """Find the edge with average x closest to x_target (tolerant of slanted edges)."""
    n = len(poly)
    best = None
    best_d = float('inf')
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        x_avg = (a[0] + b[0]) / 2.0
        d = abs(x_avg - x_target)
        if d < best_d:
            best_d = d
            best = (a, b)
    # ensure bottom→top order
    if best[0][1] <= best[1][1]:
        return best
    return best[1], best[0]


def build_lanes(boundaries: List[List[List[float]]], id_prefix: str = "L") -> Dict[str, Any]:
    segs = segs_from_boundaries(boundaries)
    poly = build_polygon_ordered(segs)
    xmin, xmax, ymin, ymax = bbox(poly)
    width = xmax - xmin
    height = ymax - ymin

    lanes = []

    if height > width:
        # vertical roadway
        bottom_left, bottom_right = find_nearest_horizontal_edge(poly, ymin)
        top_left, top_right = find_nearest_horizontal_edge(poly, ymax)

        bottom_width = bottom_right[0] - bottom_left[0]
        top_width = top_right[0] - top_left[0]

        try:
            bw_int = int(round(abs(bottom_width)))
            tw_int = int(round(abs(top_width)))
            nlanes = math.gcd(bw_int, tw_int)
            if nlanes <= 0:
                raise Exception()
        except Exception:
            nlanes = max(1, int(round(min(abs(bottom_width), abs(top_width)))))

        max_possible = max(1, int(max(bw_int if bw_int > 0 else 1, tw_int if tw_int > 0 else 1)))
        nlanes = max(1, min(nlanes, max_possible))

        for i in range(nlanes):
            bottom_seg_width = bottom_width / nlanes
            top_seg_width = top_width / nlanes
            bottom_mid = bottom_left[0] + (i + 0.5) * bottom_seg_width
            top_mid = top_left[0] + (i + 0.5) * top_seg_width
            lanes.append({
                "center_x_bottom": bottom_mid,
                "center_x_top": top_mid,
                "y_bottom": ymin,
                "y_top": ymax
            })

        lanes = list(reversed(lanes))
        final = []
        for idx, l in enumerate(lanes, start=1):
            bottom_mid = _roundish(l["center_x_bottom"])
            top_mid = _roundish(l["center_x_top"])
            ymin_r = _roundish(l["y_bottom"])
            ymax_r = _roundish(l["y_top"])
            if (idx % 2) == 1:
                enter = [bottom_mid, ymin_r]
                exit = [top_mid, ymax_r]
            else:
                enter = [top_mid, ymax_r]
                exit = [bottom_mid, ymin_r]
            final.append({
                "id": f"{id_prefix}{idx}",
                "centerline": [[bottom_mid, top_mid], [ymin_r, ymax_r]],
                "enter": enter,
                "exit": exit
            })
        return {"lanes": final}

    else:
        # horizontal roadway
        left_bottom, left_top = find_nearest_vertical_edge(poly, xmin)
        right_bottom, right_top = find_nearest_vertical_edge(poly, xmax)

        nlanes = max(1, int(round(abs(height))))
        lanes_tmp = []
        for i in range(nlanes):
            seg_h = height / nlanes
            top_y = ymin
            bottom_y = ymax
            y_mid = top_y + (i + 0.5) * seg_h
            lanes_tmp.append({
                "y_mid": y_mid,
                "x_left": xmin,
                "x_right": xmax
            })

        final = []
        for idx, l in enumerate(reversed(lanes_tmp), start=1):
            y_mid_r = _roundish(l["y_mid"])
            x_left_r = _roundish(l["x_left"])
            x_right_r = _roundish(l["x_right"])
            if (idx % 2) == 1:
                enter = [x_right_r, y_mid_r]
                exit = [x_left_r, y_mid_r]
            else:
                enter = [x_left_r, y_mid_r]
                exit = [x_right_r, y_mid_r]
            final.append({
                "id": f"{id_prefix}{idx}",
                "centerline": [[x_left_r, x_right_r], [y_mid_r, y_mid_r]],
                "enter": enter,
                "exit": exit
            })
        return {"lanes": final}
