from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Lane:
    id: str
    centerline_raw: List[List[float]]  # [[x1,x2],[y1,y2]] like YAML
    enter: Tuple[float, float]
    exit: Tuple[float, float]

    def endpoints(self) -> Tuple[Tuple[float,float], Tuple[float,float]]:
        """Convert centerline_raw to two endpoints (x1,y1),(x2,y2)."""
        xs = [self.exit[0], self.enter[0]]
        ys = [self.exit[-1], self.enter[-1]]
        return (xs[0], ys[0]), (xs[1], ys[1])

@dataclass
class SubMap:
    id: str
    boundaries: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    lanes: Dict[str, Lane]

@dataclass
class CompiledMap:
    submaps: Dict[str, SubMap]
    lane_graph: Dict[str, List[str]]  # adjacency list for lane connectivity
