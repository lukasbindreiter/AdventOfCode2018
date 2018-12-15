import warnings
from collections import deque

import numpy as np
from scipy.signal import convolve

warnings.simplefilter("ignore")


class CombatOver(Exception):
    pass


class ElfDied(Exception):
    pass


class Cavern:
    def __init__(self, data, initial_hp, elf_ad, goblin_ad,
                 can_elfs_die=False):
        self.board = np.asarray([list(l) for l in data.strip().split("\n")])
        self.walls = (self.board == "#")
        self.elves = (self.board == "E") * initial_hp
        self.goblins = (self.board == "G") * initial_hp
        self.elf_ad = elf_ad
        self.goblin_ad = goblin_ad
        self.can_elfs_die = can_elfs_die

        self._adj_kernel = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

    def combat_ongoing(self):
        """
        True if there are it least 1 goblin and 1 elf still alive
        """
        return (self.elves > 0).sum() >= 1 and (self.goblins > 0).sum() >= 1

    def walkable_map(self):
        """
        Return an 2D array of the board which is true on every tile which can be walked on, and false
        everywhere else (walls, elves, goblins)
        """
        return ~self.walls & (self.elves <= 0) & (self.goblins <= 0)

    def sorted_units_list(self):
        # np.where already returns coordinates in the correct sorting order
        coords_y, coords_x = np.where((self.elves > 0) | (self.goblins > 0))
        types = ["Elf" if self.elves[y, x] > 0 else "Goblin" for y, x in
                 zip(coords_y, coords_x)]
        return types, coords_y, coords_x

    def get_attack_target(self, y, x, enemy_type):
        """
        Return the position (y, x) of an enemy this unit can attack, or None if no enemy is adjacent
        If there are multiple adjacent enemies, the one with the lowest hp will be returned
        """
        enemy_arr = self.elves if enemy_type == "Elf" else self.goblins

        adj = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        enemies = []
        for dy, dx in adj:
            hp = enemy_arr[y + dy, x + dx]
            if hp > 0:
                enemies.append((hp, y + dy, x + dx))
        if len(enemies) == 0:
            return None

        hps = [hp for hp, y, x in enemies]
        lowest_hp = min(hps)
        # enemies are sorted by reading order
        for hp, y, x in enemies:
            if hp == lowest_hp:
                return y, x

    def attack(self, target):
        y, x = target
        if self.elves[y, x] > 0:
            self.elves[y, x] -= self.goblin_ad
            new_hp = self.elves[y, x]
            if not self.can_elfs_die and new_hp <= 0:
                raise ElfDied()
        else:
            assert self.goblins[y, x] > 0
            self.goblins[y, x] -= self.elf_ad
            new_hp = self.goblins[y, x]

        if not self.combat_ongoing():
            raise CombatOver()

        if new_hp <= 0:
            return True  # unit died after this attack

    def tiles_adjacent_to(self, unit):
        """
        Return a 2D mask which is true for the tiles adjacent to the given unit type ("Elf" or "Goblin")
        """
        unit_arr = self.elves if unit == "Elf" else self.goblins
        mask = unit_arr > 0
        # get all neighbouring tiles of the given unit type, but not the positions of the units themselves
        adj = (convolve(mask, self._adj_kernel, "same") >= 1) & (~mask) & (
            ~self.walls)
        return adj

    def get_walk_step_to_target(self, y, x, target_unit):
        walk_goals = self.tiles_adjacent_to(target_unit)
        walkable = self.walkable_map()

        # first find the closest target
        t_ys, t_xs = np.where(walk_goals)
        min_dist = None
        curr_target = None
        for t_y, t_x in zip(t_ys, t_xs):
            dist_to_target = walk_distance_bfs(y, x, t_y, t_x, walkable)
            if dist_to_target is None:
                continue
            if curr_target is None:
                min_dist = dist_to_target
                curr_target = t_y, t_x
            elif dist_to_target < min_dist:
                min_dist = dist_to_target
                curr_target = t_y, t_x

        if curr_target is None:  # no reachable target
            return None

        # now we know which tile we want to move to
        # check which of the four possible moves brings us closest to target
        t_y, t_x = curr_target
        adj = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        min_dist = None
        curr_step = None
        for dy, dx in adj:
            ny, nx = y + dy, x + dx
            if not walkable[ny, nx]:
                continue
            dist_to_target = walk_distance_bfs(ny, nx, t_y, t_x, walkable)
            if dist_to_target is None:
                continue
            if curr_step is None:
                min_dist = dist_to_target
                curr_step = ny, nx
            elif dist_to_target < min_dist:
                min_dist = dist_to_target
                curr_step = ny, nx

        return curr_step

    def get_remaining_hp(self):
        return int(
            np.clip(self.goblins, 0, np.inf).sum() + np.clip(self.elves, 0,
                                                             np.inf).sum())

    def move(self, unit_type, old_y, old_x, new_y, new_x):
        unit_arr = self.elves if unit_type == "Elf" else self.goblins
        unit_arr[new_y, new_x] = unit_arr[old_y, old_x]
        unit_arr[old_y, old_x] = 0

    def __str__(self):
        board = np.empty(self.walls.shape, "<U1")
        board[:, :] = "."
        board[self.walls] = "#"
        board[self.elves > 0] = "E"
        board[self.goblins > 0] = "G"
        return "\n".join(["".join(b) for b in board])

    def __repr__(self):
        return str(self)


def walk_distance_bfs(y, x, goal_y, goal_x, walkable):
    if not walkable[goal_y, goal_x]:
        return None

    visited = np.zeros(walkable.shape, np.bool)
    adj = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    q = deque([(0, y, x)])
    while len(q) > 0:
        dist, y, x = q.popleft()
        visited[y, x] = True
        if goal_y == y and goal_x == x:
            return dist

        for dy, dx in adj:
            ny, nx = y + dy, x + dx
            if 0 <= ny < walkable.shape[0] and 0 <= nx < walkable.shape[1] and \
                    walkable[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((dist + 1, ny, nx))
