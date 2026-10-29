# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


"""KeyCorridor's door placement mirrors MiniGrid's RoomGrid.connect_all
- every room stays reachable from the corridor, the locked room never gets
a second (openable) connector, no two doors ever share a cell, every wall
connect_all does not open is a real wall, and the number of doors follows
MiniGrid's, checked directly across many seeds."""

from typing import Optional, Set, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import navix as nx
from navix.components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID

_ENV_IDS = (
    "Navix-KeyCorridorS3R1-v0",
    "Navix-KeyCorridorS3R2-v0",
    "Navix-KeyCorridorS3R3-v0",
    "Navix-KeyCorridorS4R3-v0",
    "Navix-KeyCorridorS5R3-v0",
    "Navix-KeyCorridorS6R3-v0",
)
_N_SEEDS = 200
_N_ROWS_CONFIG = {3: 1, 5: 2}

# Frequencies of the number of unlocked doors in real MiniGrid 3.1.0,
# `KeyCorridorEnv(room_size, num_rows)` reset on seeds 0-2999. The count
# depends only on the room graph, so the four R3 sizes are pooled (12000
# resets). Keyed by `num_rows`.
_MINIGRID_UNLOCKED_DOORS = {
    1: {1: 1.0},
    2: {3: 0.754, 4: 0.246},
    3: {5: 0.414, 6: 0.366, 7: 0.220},
}
_N_COUNT_SEEDS = 2000
_COUNT_TOLERANCE = 0.05  # over four standard errors at these sample sizes


def _room_of(pos: Tuple[int, int], room_size: int) -> Optional[Tuple[int, int]]:
    """(room_row, room_col) iff `pos` is strictly inside a room, else
    None - naive floor division can't tell a wall cell from an interior
    one once room_size==1 (walls and interiors sit back-to-back)."""
    r_off = (pos[0] - 1) % (room_size + 1)
    c_off = (pos[1] - 1) % (room_size + 1)
    if not (0 <= r_off < room_size and 0 <= c_off < room_size):
        return None
    return (pos[0] - 1) // (room_size + 1), (pos[1] - 1) // (room_size + 1)


def _rooms_touching_wall(
    pos: Tuple[int, int], room_size: int, n_rows: int
) -> Set[Tuple[int, int]]:
    """A wall/door cell borders exactly two rooms - inspect its 4 neighbours."""
    rooms = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        room = _room_of((pos[0] + dr, pos[1] + dc), room_size)
        if room is not None and 0 <= room[0] < n_rows and 0 <= room[1] < 3:
            rooms.add(room)
    return rooms


def _reachable_rooms(
    door_positions, room_size: int, n_rows: int, start: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """Rooms reachable from `start` via door entities (any `requires`
    counts as a structural edge, matching MiniGrid's own reachability
    check) plus the always-open middle corridor."""
    adjacency = {(r, c): set() for r in range(n_rows) for c in range(3)}
    for row in range(n_rows - 1):
        adjacency[(row, 1)].add((row + 1, 1))
        adjacency[(row + 1, 1)].add((row, 1))
    for pos in door_positions:
        touching = list(_rooms_touching_wall(tuple(pos), room_size, n_rows))
        if len(touching) == 2:
            a, b = touching
            adjacency[a].add(b)
            adjacency[b].add(a)

    seen = {start}
    stack = [start]
    while stack:
        room = stack.pop()
        for neighbour in adjacency[room]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def _cells(mask: np.ndarray) -> Set[Tuple[int, int]]:
    return set(map(tuple, np.argwhere(mask).tolist()))


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_connect_all_reachability_and_door_uniqueness(env_id):
    env = nx.make(env_id)
    n_rows = _N_ROWS_CONFIG.get(env.height, 3)
    room_size = (env.width - 3) // 3

    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_N_SEEDS))
    timestep = jax.jit(jax.vmap(env.reset))(keys)  # also checks jit/vmap-compatibility

    positions = np.asarray(timestep.state.entities["door"].position)
    requires = np.asarray(timestep.state.entities["door"].requires)
    key_ids = np.asarray(timestep.state.entities["key"].id)[:, 0]

    for seed in range(_N_SEEDS):
        on_grid = np.all(positions[seed] >= 0, axis=-1)
        assert np.all(positions[seed][~on_grid] == np.asarray(DISCARD_PILE_COORDS)), (
            f"{env_id} seed={seed}: an off-grid door is not on the discard pile"
        )
        seed_positions = positions[seed][on_grid]
        seed_requires = requires[seed][on_grid]
        key_id = int(key_ids[seed])

        unique_positions = set(map(tuple, seed_positions.tolist()))
        assert len(unique_positions) == len(seed_positions), (
            f"{env_id} seed={seed}: expected every door on its own cell, got "
            f"{len(seed_positions)} doors on {len(unique_positions)} distinct cells"
        )

        goal_mask = seed_requires == key_id
        assert goal_mask.sum() == 1, (
            f"{env_id} seed={seed}: expected exactly one locked goal door, "
            f"got {goal_mask.sum()}"
        )
        assert np.all(seed_requires[~goal_mask] == EMPTY_POCKET_ID)
        goal_touching = _rooms_touching_wall(
            tuple(seed_positions[goal_mask][0]), room_size, n_rows
        )
        assert len(goal_touching) == 2
        locked_room = next(r for r in goal_touching if r[1] == 2)

        reach = _reachable_rooms(seed_positions, room_size, n_rows, (0, 1))
        all_rooms = {(r, c) for r in range(n_rows) for c in range(3)}
        assert reach == all_rooms, (
            f"{env_id} seed={seed}: unreachable rooms {all_rooms - reach}"
        )

        for pos in seed_positions[~goal_mask]:
            touching = _rooms_touching_wall(tuple(pos), room_size, n_rows)
            assert locked_room not in touching, (
                f"{env_id} seed={seed}: unlocked door at {tuple(pos)} "
                "bypasses the locked room"
            )


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_unopened_candidate_walls_are_walls(env_id):
    """The only floor cells on the room walls are doors and the corridor."""
    env = nx.make(env_id)
    pitch = (env.width - 3) // 3 + 1

    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_N_SEEDS))
    timestep = jax.jit(jax.vmap(env.reset))(keys)
    grids = np.asarray(timestep.state.grid)
    positions = np.asarray(timestep.state.entities["door"].position)

    rows, cols = np.indices(grids.shape[1:])
    wall_lines = (rows % pitch == 0) | (cols % pitch == 0)
    corridor = (
        (rows % pitch == 0)
        & (0 < rows)
        & (rows < env.height - 1)
        & (pitch < cols)
        & (cols < 2 * pitch)
    )
    for seed in range(_N_SEEDS):
        doors = {tuple(p) for p in positions[seed].tolist() if min(p) >= 0}
        openings = _cells(wall_lines & (grids[seed] == 0))
        assert openings == doors | _cells(corridor), (
            f"{env_id} seed={seed}: wall openings {sorted(openings)} are not "
            f"exactly the doors {sorted(doors)} plus the corridor"
        )


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_agent_starts_as_in_minigrid(env_id):
    """MiniGrid's `place_agent(1, num_rows // 2)`: any free cell of the middle
    corridor room, its corridor openings included, never facing the locked
    door."""
    env = nx.make(env_id)
    n_rows = _N_ROWS_CONFIG.get(env.height, 3)
    pitch = (env.width - 3) // 3 + 1

    def start(key):
        entities = env.reset(key).state.entities
        return entities["player"], entities["door"], entities["key"].id

    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_N_COUNT_SEEDS))
    player, doors, key_ids = jax.jit(jax.vmap(start))(keys)
    positions = np.asarray(player.position)[:, 0]
    directions = np.asarray(player.direction)[:, 0]

    steps = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]])  # east, south, west, north
    locked = np.asarray(doors.requires) == np.asarray(key_ids)
    locked_doors = np.asarray(doors.position)[locked]
    assert not np.any(np.all(positions + steps[directions] == locked_doors, axis=-1)), (
        f"{env_id}: an agent starts facing the locked door"
    )

    top = (n_rows // 2) * pitch
    rows, cols = np.indices((env.height, env.width))
    interior = (rows % pitch != 0) & (cols % pitch != 0)
    corridor = (rows % pitch == 0) & (0 < rows) & (rows < env.height - 1)
    box = (top <= rows) & (rows <= top + pitch) & (pitch < cols) & (cols < 2 * pitch)
    assert set(map(tuple, positions.tolist())) == _cells(box & (interior | corridor))
    # S3R1's middle room is one cell, and east of it is always the locked door
    assert set(directions.tolist()) == ({1, 2, 3} if n_rows == 1 else {0, 1, 2, 3})


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_door_count_matches_minigrid(env_id):
    env = nx.make(env_id)
    expected = _MINIGRID_UNLOCKED_DOORS[_N_ROWS_CONFIG.get(env.height, 3)]

    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(_N_COUNT_SEEDS))
    doors = jax.jit(jax.vmap(lambda k: env.reset(k).state.entities["door"]))(keys)
    on_grid = np.all(np.asarray(doors.position) >= 0, axis=-1)
    unlocked = on_grid & (np.asarray(doors.requires) == EMPTY_POCKET_ID)
    counts = np.bincount(unlocked.sum(axis=1), minlength=16) / _N_COUNT_SEEDS

    assert counts[list(expected)].sum() == pytest.approx(1.0), (
        f"{env_id}: unlocked-door counts {np.nonzero(counts)[0].tolist()} "
        f"outside MiniGrid's {sorted(expected)}"
    )
    for n, frequency in expected.items():
        assert abs(counts[n] - frequency) < _COUNT_TOLERANCE, (
            f"{env_id}: {n} unlocked doors in {counts[n]:.3f} of resets, "
            f"MiniGrid {frequency:.3f}"
        )


_LEGACY_IDS = tuple(env_id.replace("-v0", "-Legacy-v0") for env_id in _ENV_IDS)


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_standard_has_no_drop(env_id):
    assert nx.actions.drop not in nx.make(env_id).action_set


@pytest.mark.parametrize("env_id", _LEGACY_IDS)
def test_legacy_keeps_minigrid_actions(env_id):
    assert nx.make(env_id).action_set == nx.actions.MINIGRID_ACTION_SET


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_legacy_lays_out_like_standard_with_a_ball_target(env_id):
    standard = nx.make(env_id)
    legacy = nx.make(env_id.replace("-v0", "-Legacy-v0"))
    for seed in range(20):
        key = jax.random.PRNGKey(seed)
        a = standard.reset(key).state
        b = legacy.reset(key).state
        np.testing.assert_array_equal(a.grid, b.grid)
        for name in ("player", "key", "door"):
            for x, y in zip(
                jax.tree_util.tree_leaves(a.entities[name]),
                jax.tree_util.tree_leaves(b.entities[name]),
                strict=True,
            ):
                np.testing.assert_array_equal(x, y)
        assert "ball" not in a.entities and "goal" not in b.entities
        ball = b.get_balls()
        np.testing.assert_array_equal(ball.position[0], a.get_goals().position[0])
        np.testing.assert_array_equal(b.mission[0].position, ball.position[0])
        np.testing.assert_array_equal(b.mission[0].colour, ball.colour[0])


def _facing_ball(env, pocket_key: bool):
    """The player west of the ball, facing it, with a free floor cell to its
    north to drop into. Doors are cleared so nothing else occupies either."""
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state
    row, col = (int(x) for x in state.get_balls().position[0])
    keys = state.get_keys()
    state = state.set_player(
        state.get_player().replace(
            position=jnp.asarray([row, col - 1]),
            direction=jnp.asarray(0),
            pocket=keys.id[0] if pocket_key else jnp.asarray(EMPTY_POCKET_ID),
        )
    )
    if pocket_key:
        state = state.set_keys(
            keys.replace(position=jnp.broadcast_to(DISCARD_PILE_COORDS, keys.position.shape))
        )
    doors = state.get_doors()
    state = state.set_doors(
        doors.replace(position=jnp.broadcast_to(DISCARD_PILE_COORDS, doors.position.shape))
    )
    grid = state.grid.at[row, col - 1].set(0).at[row - 1, col - 1].set(0)
    return timestep.replace(state=state.replace(grid=grid))


def _act(env, timestep, action):
    return jax.jit(env.step)(timestep, jnp.asarray(env.action_set.index(action)))


@pytest.mark.parametrize("env_id", _LEGACY_IDS)
def test_legacy_ends_on_picking_up_the_ball(env_id):
    env = nx.make(env_id)
    result = _act(env, _facing_ball(env, pocket_key=False), nx.actions.pickup)
    assert float(result.reward) == 1
    assert bool(result.is_done())


@pytest.mark.parametrize("env_id", _LEGACY_IDS)
def test_legacy_needs_the_key_dropped_first(env_id):
    # As in MiniGrid: unlocking keeps the key, and a full pocket blocks pickup.
    env = nx.make(env_id)
    timestep = _facing_ball(env, pocket_key=True)
    timestep = _act(env, timestep, nx.actions.pickup)
    assert float(timestep.reward) == 0 and not bool(timestep.is_done())
    for action in (nx.actions.rotate_ccw, nx.actions.drop, nx.actions.rotate_cw):
        timestep = _act(env, timestep, action)
        assert float(timestep.reward) == 0 and not bool(timestep.is_done())
    assert int(timestep.state.get_player().pocket) == EMPTY_POCKET_ID
    timestep = _act(env, timestep, nx.actions.pickup)
    assert float(timestep.reward) == 1
    assert bool(timestep.is_done())


@pytest.mark.parametrize("env_id", _LEGACY_IDS)
def test_legacy_ball_stays_put(env_id):
    env = nx.make(env_id)
    timestep = env.reset(jax.random.PRNGKey(0))
    start = timestep.state.get_balls().position
    for _ in range(10):
        timestep = _act(env, timestep, nx.actions.rotate_cw)
        np.testing.assert_array_equal(timestep.state.get_balls().position, start)
