import yaml
from map_data import Lane, SubMap, CompiledMap
from collections import deque
from typing import List, Optional, Tuple
import sys

def load_yaml_map(path):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return raw

def compile_map(yaml_dict):
    submaps = {}
    lane_graph = {}

    for sub in yaml_dict['submaps']:
        lanes = {}
        for l in sub['lanes']:
            lane = Lane(
                id=l['id'],
                centerline_raw=[tuple(p) for p in l['centerline']],
                enter=tuple(l['enter']),
                exit=tuple(l['exit'])
            )
            lanes[l['id']] = lane
            lane_graph[l['id']] = []

        submap = SubMap(
            id=sub['id'],
            boundaries=[(tuple(a), tuple(b)) for a,b in sub['boundaries']],
            lanes=lanes
        )
        submaps[sub['id']] = submap

    # process connections
    for conn in yaml_dict.get('connections', []):
        src = conn['from']
        dst = conn['to']
        if src in lane_graph:
            lane_graph[src].append(dst)
        else:
            lane_graph[src] = [dst]

    return CompiledMap(submaps=submaps, lane_graph=lane_graph)

def compile_map_from_yaml(path):
    raw = load_yaml_map(path)
    return compile_map(raw)


def validate_connection(compiled_map, from_lane: str, to_lane: str, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Validate that a connection between two lanes has matching coordinates.
    
    Args:
        compiled_map: CompiledMap object containing the submaps and lanes
        from_lane: Source lane ID
        to_lane: Destination lane ID
        tolerance: Maximum allowed distance between coordinates (default: 1e-6)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Find the lanes in the submaps
    from_lane_obj = None
    to_lane_obj = None
    
    for submap in compiled_map.submaps.values():
        if from_lane in submap.lanes:
            from_lane_obj = submap.lanes[from_lane]
        if to_lane in submap.lanes:
            to_lane_obj = submap.lanes[to_lane]
    
    if from_lane_obj is None:
        return False, f"Lane '{from_lane}' not found in any submap"
    if to_lane_obj is None:
        return False, f"Lane '{to_lane}' not found in any submap"
    
    # Check if exit of from_lane matches enter of to_lane
    exit_coord = from_lane_obj.exit
    enter_coord = to_lane_obj.enter
    
    distance = ((exit_coord[0] - enter_coord[0])**2 + (exit_coord[1] - enter_coord[1])**2)**0.5
    
    if distance > tolerance:
        return False, (f"Connection {from_lane} -> {to_lane}: "
                      f"exit coordinate {exit_coord} doesn't match "
                      f"enter coordinate {enter_coord} (distance: {distance:.6f})")
    
    return True, ""


def validate_all_connections(compiled_map, tolerance: float = 1e-6) -> List[str]:
    """
    Validate all connections in the lane graph.
    
    Args:
        compiled_map: CompiledMap object
        tolerance: Maximum allowed distance between coordinates
    
    Returns:
        List of error messages for invalid connections (empty if all valid)
    """
    errors = []
    
    for from_lane, to_lanes in compiled_map.lane_graph.items():
        for to_lane in to_lanes:
            is_valid, error_msg = validate_connection(compiled_map, from_lane, to_lane, tolerance)
            if not is_valid:
                errors.append(error_msg)
    
    return errors


def find_lane_path(compiled_map, current_lane: str, ending_lane: str, 
                   validate: bool = True, tolerance: float = 1e-6) -> Optional[List[str]]:
    """
    Find a path from current_lane to ending_lane using BFS.
    
    Args:
        compiled_map: CompiledMap object containing the lane graph
        current_lane: Starting lane ID (e.g., "A2")
        ending_lane: Destination lane ID (e.g., "C1")
        validate: If True, validate coordinate matching for connections (default: True)
        tolerance: Maximum allowed distance between coordinates (default: 1e-6)
    
    Returns:
        List of lane IDs representing the path from current to ending lane,
        or None if no path exists.
    
    Raises:
        ValueError: If validation fails for any connection in the path
    
    Example:
        >>> path = find_lane_path(compiled_map, "A2", "C1")
        >>> print(path)  # ["A2", "B5", "C1"]
    """
    # Validate inputs
    if current_lane not in compiled_map.lane_graph:
        raise ValueError(f"Current lane '{current_lane}' not found in map")
    if ending_lane not in compiled_map.lane_graph:
        raise ValueError(f"Ending lane '{ending_lane}' not found in map")
    
    # Special case: already at destination
    if current_lane == ending_lane:
        return [current_lane]
    
    # BFS to find shortest path
    queue = deque([(current_lane, [current_lane])])
    visited = {current_lane}
    
    while queue:
        current, path = queue.popleft()
        
        # Check all connected lanes
        for next_lane in compiled_map.lane_graph.get(current, []):
            # Validate connection if requested
            if validate:
                is_valid, error_msg = validate_connection(compiled_map, current, next_lane, tolerance)
                if not is_valid:
                    raise ValueError(f"Invalid connection in path: {error_msg}")
            
            if next_lane == ending_lane:
                # Found the destination
                return path + [next_lane]
            
            if next_lane not in visited:
                visited.add(next_lane)
                queue.append((next_lane, path + [next_lane]))
    
    # No path found
    return None


def find_all_paths(compiled_map, current_lane: str, ending_lane: str, 
                   max_depth: int = 10, validate: bool = True, 
                   tolerance: float = 1e-6) -> List[List[str]]:
    """
    Find all possible paths from current_lane to ending_lane (up to max_depth).
    
    Args:
        compiled_map: CompiledMap object containing the lane graph
        current_lane: Starting lane ID
        ending_lane: Destination lane ID
        max_depth: Maximum path length to consider (prevents infinite loops)
        validate: If True, validate coordinate matching for connections (default: True)
        tolerance: Maximum allowed distance between coordinates (default: 1e-6)
    
    Returns:
        List of all paths found, each path is a list of lane IDs.
    
    Raises:
        ValueError: If validation fails for any connection
    """
    all_paths = []
    
    def dfs(current, target, path, visited):
        if len(path) > max_depth:
            return
        
        if current == target:
            all_paths.append(path[:])
            return
        
        for next_lane in compiled_map.lane_graph.get(current, []):
            # Validate connection if requested
            if validate:
                is_valid, error_msg = validate_connection(compiled_map, current, next_lane, tolerance)
                if not is_valid:
                    raise ValueError(f"Invalid connection: {error_msg}")
            
            if next_lane not in visited:
                visited.add(next_lane)
                path.append(next_lane)
                dfs(next_lane, target, path, visited)
                path.pop()
                visited.remove(next_lane)
    
    if current_lane not in compiled_map.lane_graph:
        raise ValueError(f"Current lane '{current_lane}' not found in map")
    if ending_lane not in compiled_map.lane_graph:
        raise ValueError(f"Ending lane '{ending_lane}' not found in map")
    
    dfs(current_lane, ending_lane, [current_lane], {current_lane})
    return all_paths



file = sys.argv[-1]

compiled = compile_map_from_yaml(file)

# Option 1: Validate entire map once at startup
errors = validate_all_connections(compiled)
if errors:
    print("Map has invalid connections - fix YAML file!")
else:
    # Then use fast pathfinding without validation
    path = find_lane_path(compiled, "A2", "K1", validate=False)
    print(path)