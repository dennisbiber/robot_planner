#!/usr/bin/env python3

import sys
import math
import yaml
import numpy as np
from map_data import Lane, SubMap, CompiledMap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

# ----------------------------
# YAML -> compiled map
# ----------------------------
def load_yaml_map(path: str) -> dict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return raw

def compile_map(yaml_dict: dict) -> CompiledMap:
    submaps: Dict[str, SubMap] = {}
    lane_graph: Dict[str, List[str]] = {}

    for sub in yaml_dict.get('submaps', []):
        lanes: Dict[str, Lane] = {}
        for l in sub.get('lanes', []):
            lane_obj = Lane(
                id=l['id'],
                centerline_raw=l['centerline'],
                enter=tuple(l['enter']),
                exit=tuple(l['exit'])
            )
            lanes[l['id']] = lane_obj
            lane_graph[l['id']] = []  # create adjacency slot

        boundaries = []
        for a,b in sub.get('boundaries', []):
            # a = [x1,x2], b=[y1,y2] -> convert to two points (x1,y1),(x2,y2)
            boundaries.append(((a[0], b[0]), (a[1], b[1])))
        submap = SubMap(id=sub['id'], boundaries=boundaries, lanes=lanes)
        submaps[sub['id']] = submap

    for conn in yaml_dict.get('connections', []):
        src = conn['from']; dst = conn['to']
        if src not in lane_graph:
            lane_graph[src] = []
        lane_graph[src].append(dst)

    return CompiledMap(submaps=submaps, lane_graph=lane_graph)

# ----------------------------
# Pathfinding (BFS) over lane graph
# ----------------------------
def find_lane_path(compiled: CompiledMap, start_lane: str, end_lane: str) -> Optional[List[str]]:
    if start_lane == end_lane:
        return [start_lane]
    if start_lane not in compiled.lane_graph or end_lane not in compiled.lane_graph:
        return None
    q = deque([(start_lane, [start_lane])])
    visited = {start_lane}
    while q:
        cur, path = q.popleft()
        for nxt in compiled.lane_graph.get(cur, []):
            if nxt in visited:
                continue
            if nxt == end_lane:
                return path + [nxt]
            visited.add(nxt)
            q.append((nxt, path + [nxt]))
    return None

# ----------------------------
# Waypoint generation along lane centerlines
# ----------------------------
def sample_line(a: Tuple[float,float], b: Tuple[float,float], spacing: float = 0.5) -> List[Tuple[float,float]]:
    ax, ay = a; bx, by = b
    length = math.hypot(bx-ax, by-ay)
    if length == 0:
        return [(ax, ay)]
    n = max(1, int(math.ceil(length / spacing)))
    pts = [(ax + (bx-ax)*t, ay + (by-ay)*t) for t in np.linspace(0.0, 1.0, n+1)]
    return pts

# ----------------------------
# Cubic Bezier sampling
# ----------------------------
def sample_cubic_bezier(p0, p1, p2, p3, spacing=0.5):
    """
    Sample a cubic Bézier curve defined by points p0,p1,p2,p3.
    Returns list of points from p0->p3 spaced approximately by 'spacing'.
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    # quick length estimate for sample count
    approx_len = np.linalg.norm(p0-p1) + np.linalg.norm(p1-p2) + np.linalg.norm(p2-p3)
    n = max(2, int(math.ceil(max(1e-6, approx_len) / spacing)))
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        u = 1 - t
        p = (u**3)*p0 + 3*(u**2)*t*p1 + 3*u*(t**2)*p2 + (t**3)*p3
        pts.append((float(p[0]), float(p[1])))
    return pts

def build_smooth_transition_bezier(A, B, last_dir_norm, cur_dir_norm,
                                   back_dist, forward_dist, spacing=0.5,
                                   tangent_scale=0.8):
    """
    Build a smooth cubic Bézier between A and B.
    """
    handle_A = tangent_scale * max(back_dist, 0.5 * spacing)
    handle_B = tangent_scale * max(forward_dist, 0.5 * spacing)

    cp1 = (A[0] + last_dir_norm[0]*handle_A, A[1] + last_dir_norm[1]*handle_A)
    cp2 = (B[0] - cur_dir_norm[0]*handle_B, B[1] - cur_dir_norm[1]*handle_B)

    bezier_pts = sample_cubic_bezier(A, cp1, cp2, B, spacing=spacing)
    return bezier_pts


# ----------------------------
# Main waypoint generator with Bézier transitions
# ----------------------------
def build_waypoints_for_lane_sequence(compiled, lane_seq: List[str], spacing: float = 0.5,
                                     backtrack_ratio: float = 0.2, forward_ratio: float = 0.35,
                                     angle_threshold_deg: float = 25.0) -> List[Tuple[float,float]]:
    """
    Build waypoints for a sequence of lanes, inserting smooth Bézier transitions at turns.
    """
    waypoints: List[Tuple[float,float]] = []
    last_exit = None
    last_dir = None
    last_segment_len = 0.0
    last_sample_counts = 0

    def find_lane_obj(lane_id: str):
        for sm in compiled.submaps.values():
            if lane_id in sm.lanes:
                return sm.lanes[lane_id]
        raise ValueError(f"Lane {lane_id} not found in compiled map")

    for lane_id in lane_seq:
        lane_obj = find_lane_obj(lane_id)
        p_raw1, p_raw2 = lane_obj.endpoints()
        enter = lane_obj.enter

        # determine orientation of lane
        d1 = math.hypot(p_raw1[0]-enter[0], p_raw1[1]-enter[1])
        d2 = math.hypot(p_raw2[0]-enter[0], p_raw2[1]-enter[1])
        if d1 <= d2:
            p1, p2 = p_raw1, p_raw2
        else:
            p1, p2 = p_raw2, p_raw1

        if last_exit is not None:
            if math.hypot(last_exit[0]-p2[0], last_exit[1]-p2[1]) < math.hypot(last_exit[0]-p1[0], last_exit[1]-p1[1]):
                p1, p2 = p2, p1

        cur_vec = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        cur_len = np.linalg.norm(cur_vec)
        cur_dir = (cur_vec / cur_len) if cur_len > 1e-8 else np.array([0.0, 0.0])

        if last_dir is not None and cur_len > 1e-8 and np.linalg.norm(last_dir) > 1e-8:
            cosang = np.clip(np.dot(last_dir, cur_dir) / (np.linalg.norm(last_dir)*np.linalg.norm(cur_dir)), -1.0, 1.0)
            turn_angle_deg = math.degrees(math.acos(cosang))

            if turn_angle_deg >= angle_threshold_deg:
                # compute transition chord endpoints
                back_dist = min(backtrack_ratio * last_segment_len, max(spacing, spacing*3))
                forward_dist = min(forward_ratio * cur_len, max(spacing, spacing*3))

                last_dir_norm = last_dir / (np.linalg.norm(last_dir)+1e-12)
                A = (last_exit[0] - last_dir_norm[0]*back_dist, last_exit[1] - last_dir_norm[1]*back_dist)
                B = (p1[0] + cur_dir[0]*forward_dist, p1[1] + cur_dir[1]*forward_dist)

                # pop overlapping last waypoints
                pop_count = int(math.ceil(back_dist / max(1e-6, spacing)))
                pop_count = min(pop_count, last_sample_counts)
                for _ in range(pop_count):
                    if waypoints:
                        waypoints.pop()

                # build Bézier transition
                bezier_pts = build_smooth_transition_bezier(A, B, last_dir_norm, cur_dir,
                                                            back_dist, forward_dist,
                                                            spacing=spacing, tangent_scale=0.4)

                # sanity fallback if nan/empty
                if not bezier_pts or any(math.isnan(x) or math.isnan(y) for x,y in bezier_pts):
                    bezier_pts = sample_line(A, B, spacing)

                # merge to path
                if waypoints and bezier_pts:
                    if math.hypot(waypoints[-1][0]-bezier_pts[0][0], waypoints[-1][1]-bezier_pts[0][1]) < 1e-6:
                        bezier_pts = bezier_pts[1:]
                waypoints.extend(bezier_pts)

                # tail from B -> p2
                if math.hypot(B[0]-p2[0], B[1]-p2[1]) > 1e-6:
                    tail = sample_line(B, p2, spacing)
                else:
                    tail = []
                if waypoints and tail:
                    if math.hypot(waypoints[-1][0]-tail[0][0], waypoints[-1][1]-tail[0][1]) < 1e-6:
                        tail = tail[1:]
                waypoints.extend(tail)

                last_exit = p2
                last_dir = cur_dir * cur_len
                last_segment_len = cur_len
                last_sample_counts = len(bezier_pts) + len(tail)
                continue

        # no curve, straight line
        sampled = sample_line(p1, p2, spacing)
        if waypoints and sampled:
            if math.hypot(waypoints[-1][0]-sampled[0][0], waypoints[-1][1]-sampled[0][1]) < 1e-6:
                sampled = sampled[1:]
        waypoints.extend(sampled)

        last_exit = p2
        last_dir = cur_dir * cur_len
        last_segment_len = cur_len
        last_sample_counts = len(sampled)

    return waypoints

# ----------------------------
# Geometry helpers for collision / distances
# ----------------------------
def closest_point_on_segment_point(p: Tuple[float,float], a: Tuple[float,float], b: Tuple[float,float]) -> Tuple[Tuple[float,float], float]:
    p = np.array(p); a = np.array(a); b = np.array(b)
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0:
        return (tuple(a), 0.0)
    t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
    closest = a + t * ab
    return (tuple(closest), float(t))

def point_distance_to_boundaries(point: Tuple[float,float], compiled: CompiledMap) -> float:
    # returns min distance to any boundary segment in the whole map
    min_d = float('inf')
    for sm in compiled.submaps.values():
        for a,b in sm.boundaries:
            cp, _ = closest_point_on_segment_point(point, a, b)
            d = math.hypot(point[0]-cp[0], point[1]-cp[1])
            if d < min_d: min_d = d
    return min_d

# ----------------------------
# Vehicle model (kinematic bicycle like earlier)
# ----------------------------
class RobotState:
    def __init__(self, x=2.0, y=5.0, yaw=0.0):
        self.x = float(x); self.y = float(y); self.yaw = float(yaw)
        self.v = 0.0
        self.delta = 0.0
        self.wheelbase = 0.6
        self.linear_damping = 0.5

    def update(self, accel: float, steer_rate: float, dt: float):
        max_steer = math.radians(45)
        max_speed = 20.0
        self.delta += steer_rate * dt
        self.delta = max(-max_steer, min(max_steer, self.delta))
        self.v += accel * dt
        # damping
        self.v -= self.v * self.linear_damping * dt
        self.v = max(-max_speed, min(max_speed, self.v))
        if abs(self.delta) < 1e-8:
            yaw_rate = 0.0
        else:
            yaw_rate = self.v / self.wheelbase * math.tan(self.delta)
        self.yaw += yaw_rate * dt
        self.yaw = (self.yaw + math.pi) % (2*math.pi) - math.pi
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt

# ----------------------------
# Controller: follow waypoint list using heading + stanley lateral correction
# ----------------------------
class WaypointFollower:
    def __init__(self, robot: RobotState, compiled: CompiledMap, waypoints: List[Tuple[float,float]]):
        self.robot = robot
        self.compiled = compiled
        self.waypoints = waypoints
        self.idx = 0
        self.waypoint_tol = 0.6
        # gains
        self.kp_speed = 1.4
        self.kp_heading = 1.0
        self.k_stanley = 1.0
        self.max_accel = 2.0
        self.max_steer_rate = math.radians(35)

    def current_goal(self) -> Tuple[float,float]:
        if self.idx >= len(self.waypoints):
            return self.waypoints[-1]
        return self.waypoints[self.idx]
    
    def lookahead_goal(self) -> Tuple[float, float]:
        """
        Returns a target point by blending the next three waypoints
        (current idx, idx+1, idx+2) with weights 0.2, 0.5, and 0.3 respectively.
        This provides a smoother lookahead target that anticipates turns.
        """
        n = len(self.waypoints)
        if n == 0:
            return (0.0, 0.0)
        if self.idx >= n:
            return self.waypoints[-1]

        # get the 3 waypoints ahead (clamp to last point if needed)
        p0 = self.waypoints[self.idx]
        p1 = self.waypoints[self.idx + 1] if self.idx + 1 < n else p0
        p2 = self.waypoints[self.idx + 2] if self.idx + 2 < n else p1

        # weighted combination: 0.2 * p0 + 0.5 * p1 + 0.3 * p2
        w0, w1, w2 = 0.3, 0.3, 0.4
        goal_x = w0 * p0[0] + w1 * p1[0] + w2 * p2[0]
        goal_y = w0 * p0[1] + w1 * p1[1] + w2 * p2[1]

        return (goal_x, goal_y)



    def step(self, dt: float):
        if self.idx >= len(self.waypoints):
            # already at final goal: brake
            self.robot.update(-self.max_accel, 0.0, dt)
            return

        gx, gy = self.lookahead_goal()
        # if close enough, advance
        dist_goal = math.hypot(gx - self.robot.x, gy - self.robot.y)
        if dist_goal < self.waypoint_tol:
            self.idx += 1
            if self.idx >= len(self.waypoints):
                # stop at final
                self.robot.update(-self.max_accel, 0.0, dt)
                return
            gx, gy = self.lookahead_goal()


        # desired heading
        desired_heading = math.atan2(gy - self.robot.y, gx - self.robot.x)
        heading_error = (desired_heading - self.robot.yaw + math.pi) % (2*math.pi) - math.pi
        if self.idx == 0:
            prev = (self.robot.x, self.robot.y)
        else:
            prev = self.waypoints[self.idx - 1]
        # compute cross-track error
        cp, _ = closest_point_on_segment_point((self.robot.x, self.robot.y), prev, (gx, gy))
        cross_err = math.hypot(self.robot.x - cp[0], self.robot.y - cp[1])
        # sign of cross track: use cross product sign between path direction and robot vector
        path_vec = np.array([gx - prev[0], gy - prev[1]])
        if np.linalg.norm(path_vec) > 1e-8:
            # previous:
            # cross_sign = np.sign(np.cross(path_vec, np.array([self.robot.x - prev[0], self.robot.y - prev[1]])))

            # fixed 2D->3D cross
            robot_vec3 = np.array([self.robot.x - prev[0], self.robot.y - prev[1], 0.0])
            path_vec3  = np.array([gx - prev[0], gy - prev[1], 0.0])
            cross_sign = np.sign(np.cross(path_vec3, robot_vec3)[2])

        else:
            cross_sign = 0.0
        cross_err_signed = cross_err * (-cross_sign)  # choose convention so positive => right

        # Stanley term
        stanley_term = math.atan2(self.k_stanley * cross_err_signed, (self.robot.v + 0.5))

        # desired steering: heading + stanley
        desired_steer = self.kp_heading * heading_error + stanley_term

        # saturate and create steer rate
        max_steer = math.radians(35)
        desired_steer = max(-max_steer, min(max_steer, desired_steer))
        steer_rate_cmd = (desired_steer - self.robot.delta) / max(dt, 1e-6)
        steer_rate_cmd = max(-self.max_steer_rate, min(self.max_steer_rate, steer_rate_cmd))

        # speed: modest target, slow on large heading error
        # Estimate curvature ahead
        def path_curvature(p1, p2, p3):
            x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
            a = math.hypot(x2 - x1, y2 - y1)
            b = math.hypot(x3 - x2, y3 - y2)
            c = math.hypot(x3 - x1, y3 - y1)
            if a < 1e-6 or b < 1e-6 or c < 1e-6:
                return 0.0
            num = abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
            denom = a * b * c
            return (2 * num / denom)

        # pick lookahead points
        p1 = self.waypoints[max(self.idx - 1, 0)]
        p2 = self.waypoints[self.idx]
        p3 = self.waypoints[min(self.idx + 3, len(self.waypoints)-1)]
        curv = path_curvature(p1, p2, p3)

        # map curvature to speed limit
        # tweak 'curve_slowdown_gain' to control aggressiveness
        curve_slowdown_gain = 2.5
        v_curve_limit = 2.0 / (1.0 + curve_slowdown_gain * curv)

        # normal heading-based speed target
        desired_speed = 2.0 * (1.0 - 0.6 * min(1.0, abs(heading_error)))

        # apply curvature-based limit
        desired_speed = min(desired_speed, v_curve_limit)

        accel_cmd = self.kp_speed * (desired_speed - self.robot.v)
        accel_cmd = max(-self.max_accel, min(self.max_accel, accel_cmd))

        # update robot
        self.robot.update(accel_cmd, steer_rate_cmd, dt)


# ----------------------------
# Visualization helpers
# ----------------------------
def draw_robot_polygon(ax, robot: RobotState, size=0.35):
    x, y, yaw = robot.x, robot.y, robot.yaw
    L = size; W = size*0.6
    pts = np.array([[L,0], [-L*0.6, -W/2], [-L*0.6, W/2]])
    c = math.cos(yaw); s = math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts_world = pts.dot(R.T) + np.array([x,y])
    poly = Polygon(pts_world, closed=True, color='dodgerblue', zorder=5)
    ax.add_patch(poly)
    return poly

# ----------------------------
# Main simulation / animation
# ----------------------------
def main():
    # parse args
    argv = sys.argv[1:]
    if len(argv) >= 3:
        map_file, start_lane, end_lane = argv[0], argv[1], argv[2]
    elif len(argv) == 2:
        map_file = argv[0]; start_lane = argv[1]; end_lane = "H5"
    else:
        map_file = "map.yaml"; start_lane = "A2"; end_lane = "H5"

    raw = load_yaml_map(map_file)
    compiled = compile_map(raw)

    path = find_lane_path(compiled, start_lane, end_lane)
    if path is None:
        print(f"No lane path from {start_lane} to {end_lane}")
        sys.exit(1)

    print("Lane path:", path)

    # generate waypoints sampled along each lane centerline
    waypoints = build_waypoints_for_lane_sequence(compiled, path, spacing=0.75)
    print(f"Generated {len(waypoints)} waypoints")

    # starting robot pose: use the enter coordinate of the start lane
    start_lane_obj = None
    for sm in compiled.submaps.values():
        if start_lane in sm.lanes:
            start_lane_obj = sm.lanes[start_lane]
            break

    if start_lane_obj is not None:
        sx, sy = start_lane_obj.enter
        ex, ey = start_lane_obj.exit
        dx, dy = ex - sx, ey - sy
        yaw = math.atan2(dy, dx)
    else:
        sx, sy, yaw = 2.0, 5.0, 0.0

    robot = RobotState(x=sx, y=sy, yaw=yaw)
    follower = WaypointFollower(robot, compiled, waypoints)

    # Setup plot
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', 'box')

    # draw boundaries and lane centerlines
    # boundaries: gray thick
    all_x = []; all_y = []
    for sm in compiled.submaps.values():
        for a,b in sm.boundaries:
            ax.plot([a[0], b[0]], [a[1], b[1]], color='gray', linewidth=3, zorder=1)
            all_x += [a[0], b[0]]; all_y += [a[1], b[1]]
    margin = 1.0
    ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax.set_ylim(min(all_y)-margin, max(all_y)+margin)

    # draw centerlines for all lanes
    for sm in compiled.submaps.values():
        for lane in sm.lanes.values():
            p1, p2 = lane.endpoints()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='orange', linestyle='--', linewidth=1, zorder=2)
            ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, lane.id, color='darkorange', zorder=4, fontsize=8)

    # draw waypoints and path
    wx = [p[0] for p in waypoints]; wy = [p[1] for p in waypoints]
    path_line, = ax.plot(wx, wy, '-', color='green', linewidth=1.2, alpha=0.8, zorder=3)
    # waypoint scatter
    ax.scatter(wx, wy, s=8, c='green', zorder=3)

    # goal marker
    gx, gy = waypoints[-1]
    ax.plot([gx], [gy], 's', color='red', markersize=8, zorder=6)

    # initial robot polygon
    robot_patch = draw_robot_polygon(ax, robot)

    trail_x = [robot.x]; trail_y = [robot.y]
    trail_line, = ax.plot(trail_x, trail_y, '-', linewidth=1.0, color='blue', zorder=4)

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

    dt = 0.1
    sim_time = {'t':0.0}

    def update(frame):
        follower.step(dt)
        sim_time['t'] += dt

        nonlocal robot_patch
        robot_patch.remove()
        robot_patch = draw_robot_polygon(ax, robot)

        trail_x.append(robot.x); trail_y.append(robot.y)
        if len(trail_x) > 1000:
            trail_x.pop(0); trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        info_text.set_text(
            f"t={sim_time['t']:.1f}s  pos=({robot.x:.2f},{robot.y:.2f}) v={robot.v:.2f} steer={math.degrees(robot.delta):.1f}°\n"
            f"waypoint_idx={follower.idx}/{len(follower.waypoints)-1}"
        )

        return robot_patch, trail_line, info_text

    ani = FuncAnimation(fig, update, interval=dt*1000, frames=2000, blit=False)
    plt.title(f"Following lane path: {' -> '.join(path)}")
    plt.show()

if __name__ == "__main__":
    main()
