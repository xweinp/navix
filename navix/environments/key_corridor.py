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

"""MiniGrid's KeyCorridor environment - find a key off a corridor to reach a locked room.

Two variants share the layout. The standard `Navix-KeyCorridorS*R*-v0` puts a
walkable `Goal` in the locked room and ends when the player steps onto it; it
has no `drop`, since nothing in the task needs the pocket emptied. The
`-Legacy-v0` ids are MiniGrid's original task: the target is a `Ball`, the
episode ends when it is picked up, and since the key is still carried after
unlocking (MiniGrid's `Door.toggle` keeps it) and the pocket holds one object,
the key has to be dropped first - so these keep MiniGrid's full action set.

See the environment class in this module for the task, layout and
reward/termination details.
"""


from __future__ import annotations

from typing import List, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flax import struct

from navix import actions, observations, rewards, terminations, transitions

from ..components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from ..environments import Environment
from ..entities import Ball, Goal, Player, Key, Door
from ..states import Event, State
from ..environments import Timestep
from ..grid import random_colour, RoomsGrid
from .registry import register_env


def _room_id(row: int, col: int) -> int:
    return row * 3 + col


class KeyCorridor(Environment):
    """Find a key hidden behind unlocked doors, then use it to open the one
    locked door guarding the goal.

    Rooms form an `n_rows x 3` grid: the middle column is one corridor, the
    key lives in a left-column room and the goal in a right-column room
    behind the one locked door.

    Every other door follows MiniGrid's `RoomGrid.connect_all`: each wall not
    touching the locked room gets a door, in uniformly random order, until
    every room is reachable. MiniGrid draws walls with replacement, but the
    order distinct walls are first drawn in is a uniform permutation, so one
    pass over a permutation adds the same doors.

    `jax.jit` needs a fixed entity count, so every candidate wall is a `Door`;
    the ones left shut sit at `DISCARD_PILE_COORDS` and their cell stays a wall.

    `legacy` swaps the `Goal` for MiniGrid's randomly-coloured `Ball`, tracked
    as `state.mission` for `on_target_fetched`. Its colour is drawn off a key
    folded from the goal-position key, so both variants lay out identically
    for the same reset key.
    """

    legacy: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        n_rows_config = {3: 1, 5: 2}
        n_rows = n_rows_config.get(self.height, 3)
        room_size = (self.width - 3) // 3
        pitch = room_size + 1
        k_key_row, k_key_pos, k_colour, k_goal_row, k_goal_pos, k_doors, k_agent = (
            jax.random.split(key, num=7)
        )

        # grid of rooms
        grid = RoomsGrid.create(n_rows, 3, (room_size, room_size))

        # key
        key_room_row = jax.random.randint(k_key_row, (), minval=0, maxval=n_rows)
        key_pos = grid.position_in_room(
            key_room_row, jnp.asarray(0, dtype=jnp.int32), key=k_key_pos
        )
        key_colour = random_colour(k_colour)
        key_id = jnp.asarray(1)
        key_obj = Key.create(key_pos, key_colour, key_id)

        # goal
        goal_room_row = jax.random.randint(k_goal_row, (), minval=0, maxval=n_rows)
        goal_pos = grid.position_in_room(goal_room_row, jnp.asarray(2), key=k_goal_pos)
        if self.legacy:
            target_colour = random_colour(jax.random.fold_in(k_goal_pos, 1))
            target = {
                "ball": Ball.create(
                    goal_pos,
                    target_colour,
                    probability=jnp.asarray(1.0),
                    id=jnp.asarray(2),
                )[None]
            }
            mission = (
                Event(
                    position=goal_pos,
                    colour=target_colour,
                    happened=jnp.asarray(False),
                ),
            )
        else:
            target = {"goal": Goal.create(goal_pos, probability=jnp.asarray(1.0))[None]}
            mission = ()

        # Doors: connect_all - see the class docstring. `parent` (union-find)
        # is kept fully flat as an invariant: a union from a flat state only
        # ever strands nodes one hop further off, so a single
        # `parent[parent]` pass after each union always restores it, and a
        # lookup is one gather however many rooms have merged.
        num_rooms = 3 * n_rows
        # (row, u, v, wall cell before the offset, offset axis) - `u`/`v` are
        # the two rooms the wall connects; the door sits `1..room_size` cells
        # along the wall, uniformly, as MiniGrid's `door_pos` does.
        candidates: List[Tuple[int, int, int, Tuple[int, int], int]] = []
        for row in range(n_rows):
            r = row * pitch
            candidates.append((row, _room_id(row, 0), _room_id(row, 1), (r, pitch), 0))
            candidates.append(
                (row, _room_id(row, 1), _room_id(row, 2), (r, 2 * pitch), 0)
            )
        for row in range(n_rows - 1):
            r = (row + 1) * pitch
            candidates.append((row, _room_id(row, 0), _room_id(row + 1, 0), (r, 0), 1))
            candidates.append(
                (row, _room_id(row, 2), _room_id(row + 1, 2), (r, 2 * pitch), 1)
            )
        num_candidates = len(candidates)

        k_offsets, k_colours, k_perm = jax.random.split(k_doors, num=3)
        offsets = jax.random.randint(
            k_offsets, (num_candidates,), minval=1, maxval=room_size + 1
        )
        wall_cells = jnp.asarray([cell for *_, cell, _ in candidates], dtype=jnp.int32)
        along = jnp.asarray(np.eye(2, dtype=np.int32)[[a for *_, a in candidates]])
        positions = wall_cells + offsets[:, None] * along
        colours = random_colour(k_colours, num_candidates)
        perm = jax.random.permutation(k_perm, num_candidates)

        row_ids = jnp.asarray([row for row, *_ in candidates])
        u_ids = jnp.asarray([u for _, u, _, _, _ in candidates])
        v_ids = jnp.asarray([v for _, _, v, _, _ in candidates])
        # the (row, col=1)<->(row, col=2) candidate, for whichever row
        # turns out to be the goal row - the one mandatory locked door.
        is_goal_slot = (u_ids % 3 == 1) & (v_ids % 3 == 2) & (row_ids == goal_room_row)

        # The middle column's inter-row walls are unconditionally carved
        # below regardless of doors, so every (row, col=1) room is
        # already one component - point them all directly at row 0's
        # (already flat, no unions/lookups needed to build this).
        parent = jnp.arange(num_rooms)
        corridor_root = _room_id(0, 1)
        col1_ids = jnp.asarray([_room_id(row, 1) for row in range(n_rows)])
        parent = parent.at[col1_ids].set(corridor_root)
        # the mandatory locked door: the goal row's col=2 room joins the
        # (already-flat) corridor component in one hop.
        locked_room = goal_room_row * 3 + 2
        parent = parent.at[locked_room].set(corridor_root)

        eligible = (u_ids != locked_room) & (v_ids != locked_room)
        # the corridor merges `n_rows` rooms into one, the locked door one more
        components = jnp.asarray(num_rooms - n_rows)

        u_perm, v_perm, eligible_perm = u_ids[perm], v_ids[perm], eligible[perm]
        adds = []
        for i in range(num_candidates):
            ru, rv = parent[u_perm[i]], parent[v_perm[i]]
            add = eligible_perm[i] & (components > 1)
            merge = add & (ru != rv)
            parent = parent.at[ru].set(jnp.where(merge, rv, ru))
            parent = parent[parent]  # one pointer-doubling pass restores flatness
            components = components - merge
            adds.append(add)
        added = jnp.zeros((num_candidates,), dtype=jnp.bool_).at[perm].set(jnp.stack(adds))

        on_grid = is_goal_slot | added
        door_colours = jnp.where(is_goal_slot, key_colour, colours)
        doors = Door.create(
            position=jnp.where(on_grid[:, None], positions, DISCARD_PILE_COORDS),
            requires=jnp.where(is_goal_slot, key_id, EMPTY_POCKET_ID),
            colour=door_colours,
            open=jnp.zeros((num_candidates,), dtype=jnp.int32),
        )
        # room walls with the corridor's inter-row walls removed, fixed per
        # env, so built once at trace time as a constant
        walls = np.zeros((self.height, self.width), dtype=np.float32)
        walls[::pitch, :] = -1
        walls[:, ::pitch] = -1
        walls[pitch : self.height - 1 : pitch, pitch + 1 : 2 * pitch] = 0
        # a wall `connect_all` never reached stays a wall: its carve goes to
        # row `self.height`, out of bounds, which `mode="drop"` discards
        carved = jnp.where(on_grid[:, None], positions, jnp.asarray([self.height, 0]))
        grid = jnp.asarray(walls).at[carved[:, 0], carved[:, 1]].set(0, mode="drop")

        # agent: MiniGrid's `place_agent(1, n_rows // 2)`, a uniform (cell,
        # direction) in the middle corridor room, never facing the locked door.
        # The free cells form a rectangle and only the east-facing pose beside
        # the locked door can be invalid, so draw one index over the valid
        # count and map that pose's slot to the last (north-facing) pose.
        middle = n_rows // 2
        first_row = middle * pitch + (0 if middle > 0 else 1)
        last_row = middle * pitch + (pitch if middle < n_rows - 1 else room_size)
        num_poses = (last_row - first_row + 1) * room_size * 4
        locked_door_row = positions[jnp.argmax(is_goal_slot), 0]
        blocked = (goal_room_row == middle).astype(jnp.int32)
        blocked_pose = ((locked_door_row - first_row) * room_size + room_size - 1) * 4
        pose = jax.random.randint(k_agent, (), minval=0, maxval=num_poses - blocked)
        pose = jnp.where((blocked == 1) & (pose == blocked_pose), num_poses - 1, pose)
        cell = pose // 4
        agent_pos = jnp.stack([first_row + cell // room_size, pitch + 1 + cell % room_size])
        # direction indices follow `translate`: east, south, west, north
        player = Player.create(agent_pos, pose % 4, pocket=EMPTY_POCKET_ID)

        entities = {
            "player": player[None],
            "key": key_obj[None],
            "door": doors,
            **target,
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
            mission=mission,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


# MiniGrid's actions without `drop`: the standard task never needs the pocket
# emptied, so the action only ever wasted a step. Indices after `pickup` shift
# down by one relative to `MINIGRID_ACTION_SET`.
ACTION_SET_WITHOUT_DROP = tuple(
    a for a in actions.MINIGRID_ACTION_SET if a is not actions.drop
)


def _register(name: str, height: int, width: int) -> None:
    register_env(
        f"Navix-KeyCorridor{name}-v0",
        lambda *args, **kwargs: KeyCorridor.create(
            height=height,
            width=width,
            observation_fn=kwargs.pop("observation_fn", observations.symbolic),
            reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
            termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
            action_set=kwargs.pop("action_set", ACTION_SET_WITHOUT_DROP),
            *args,
            **kwargs,
        ),
    )
    register_env(
        f"Navix-KeyCorridor{name}-Legacy-v0",
        lambda *args, **kwargs: KeyCorridor.create(
            height=height,
            width=width,
            legacy=True,
            # the default transition walks every `Ball` around, as in
            # DynamicObstacles; MiniGrid's target stays where it was placed
            transitions_fn=kwargs.pop(
                "transitions_fn", transitions.deterministic_transition
            ),
            observation_fn=kwargs.pop("observation_fn", observations.symbolic),
            reward_fn=kwargs.pop("reward_fn", rewards.on_target_fetched),
            termination_fn=kwargs.pop("termination_fn", terminations.on_target_fetched),
            *args,
            **kwargs,
        ),
    )


for _name, _height, _width in (
    ("S3R1", 3, 7),
    ("S3R2", 5, 7),
    ("S3R3", 7, 7),
    ("S4R3", 10, 10),
    ("S5R3", 13, 13),
    ("S6R3", 16, 16),
):
    _register(_name, _height, _width)
