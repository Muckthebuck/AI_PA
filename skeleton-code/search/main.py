"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part A: Searching

This script contains the entry point to the program (the code in
`__main__.py` calls `main()`). Your solution starts here!
"""
import json
import sys
import heapq
# If you want to separate your code into separate files, put them
# inside the `search` directory (like this one and `util.py`) and
# then import from them like this:
from search.util import print_board
from typing import Dict, List, Tuple, TypeVar, Optional

# class Graph:
#     """
#     adapted from https://www.redblobgames.com/pathfinding/a-star/implementation.html
#     """
#     def __init__(self):
# def heuristic(a: GridLocation, b: GridLocation) -> float:
#     (x1, y1) = a
#     (x2, y2) = b
#     return abs(x1 - x2) + abs(y1 - y2)

T = TypeVar('T')
Location = TypeVar('Location')


class Graph:
    """
        adapted from https://www.redblobgames.com/pathfinding/a-star/implementation.html
    """

    def __init__(self, n, board_dict):
        self.cell: Dict[Location, List[Location]] = {}
        moves = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]]
        for i in range(0, n):
            for j in range(0, n):
                curr = [i, j]
                neighbours = []
                if tuple(curr) in board_dict.keys():
                    continue
                for k in moves:
                    nCell = [0] * 2
                    nCell[0] = curr[0] + k[0]
                    nCell[1] = curr[1] + k[1]
                    if not (nCell[0] < 0 or nCell[0] >= n or nCell[1] < 0 or nCell[1] >= n or
                            tuple(nCell) in board_dict.keys()):
                        neighbours.append(tuple(nCell))

                self.cell.update({tuple(curr): neighbours})
                # print(curr, neighbours)

    def print(self):
        for cell, neighbours in self.cell.items():
            print(cell, neighbours)

    def neighbors(self, id: Location) -> List[Location]:
        return self.cell[id]

    def cost(self, src: Location, dst: Location) -> float:
        pass


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heapq.heappop(self.elements)[1]


def reconstruct_path(came_from: Dict[Location, Location],
                     start: Location, goal: Location) -> List[Location]:
    current: Location = goal
    path: List[Location] = []
    while current != start:  # note: this will fail if no path found
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional
    return path


def find_dist(H1, H2):
    """
    # adapted from redblobgames.com/grids/hexagon
    Args:
        H1: src hexagon cell coordinate tuple
        H2: dst hexagon cell coordinate tuple

    #Relative direction, i.e. which wedge side i need to go towards
    1  .-'-. 0
    2 |     | 5
    3 '-._.-'4
    Returns:(manhattan distance, relative direction H2 to H1)

    """

    r = H1[0] - H2[0]
    q = H1[1] - H2[1]
    s = -r - q
    d = (abs(r) + abs(q) + abs(s))/2

    return d


def a_star_search(graph: Graph, start: Location, goal: Location):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: Dict[Location, Optional[Location]] = {}
    cost_so_far: Dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current: Location = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + 1 #+1 for cost to next node
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + find_dist(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def main():
    try:
        with open(sys.argv[1]) as file:
            data = json.load(file)
    except IndexError:
        print("usage: python3 -m search path/to/input.json", file=sys.stderr)
        sys.exit(1)

    # TODO:
    # Find and print a solution to the board configuration described
    # by `data`.
    # Why not start by trying to print this configuration out using the
    # `print_board` helper function? (See the `util.py` source code for
    # usage information).
    def list_to_dict(lst):
        dic = {}
        for i in lst:
            dic.update({(i[1], i[2]): i[0]})
        return dic

    n = data['n']
    board_dict = list_to_dict(data['board'])
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    print_board(n, board_dict, "", False)
    board_graph = Graph(n, board_dict)
    board_graph.print()
    (came_from, cost_so_far) = a_star_search(board_graph, start, goal)
    print(reconstruct_path(came_from, start, goal))
