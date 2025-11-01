# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Santorini, implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel
import bisect

_NUM_PLAYERS = 2
_NUM_ROWS = 5
_NUM_COLS = _NUM_ROWS
_NUM_FLOORS = 3
_NUM_FLOOR_BITS = int.bit_length(_NUM_FLOORS + 1)
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_ALL_LEGAL_PLACEMENT_ACTIONS = [(i, j) for i in range(_NUM_CELLS) for j in range(_NUM_CELLS) if i < j]
_ALL_DIRECTIONS = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if i!=0 or j!=0]
_NUM_DISTINCT_ACTIONS = 2*(len(_ALL_DIRECTIONS)**2) + len(_ALL_LEGAL_PLACEMENT_ACTIONS)

_GAME_TYPE = pyspiel.GameType(
    short_name="python_santorini",
    long_name="Python Santorini",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_CELLS*(_NUM_FLOORS+1))


class SantoriniGame(pyspiel.Game):
  """A Python version of the Santorini game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return SantoriniState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class SantoriniState(pyspiel.State):
  """A python version of the Santorini state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self.board = np.full((_NUM_ROWS, _NUM_COLS), 0, dtype=np.uint8)
    self.num_workers_placed = 0
    self._is_terminal = False
    self._legal_actions_cache = None
    self._player0_score = 0.

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self.is_terminal() else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    return sorted([self._encode_action(action) for action in self._legal_actions_cache_lookup()])

  def _legal_actions_cache_lookup(self):
    """Returns a list of legal actions, sorted in ascending order."""
    if self._legal_actions_cache is None:
      if self.num_workers_placed < 2:
        self._legal_actions_cache = [(i, j) for i in range(_NUM_CELLS) for j in range(_NUM_CELLS) if i < j]
        self._is_terminal = False
      elif self.num_workers_placed < 4:
        self._legal_actions_cache = [(i, j) for i in range(_NUM_CELLS) for j in range(_NUM_CELLS) if i < j and self.board[_coord(i)] == 0 and self.board[_coord(j)] == 0]
        self._is_terminal = False
      else:
        worker_positions = [i for i in range(_NUM_CELLS) if self.board[_coord(i)] & (1<<(_NUM_FLOOR_BITS + self._cur_player))]
        assert len(worker_positions) == 2, str(self.board) + str(worker_positions) + str(self._cur_player) + str(self.num_workers_placed)

        self._legal_actions_cache = []
        move_actions = []

        for w, position in enumerate(worker_positions):
          for i in range(_NUM_CELLS):
            if _is_neighbour(position, i) and not _is_occuppied(self.board[_coord(i)]) and _height(self.board[_coord(i)]) <= min(_NUM_FLOORS, _height(self.board[_coord(position)]) + 1):
              move_actions.append((w, i))

        for move in move_actions:
          for i in range(_NUM_CELLS):
            if _is_neighbour(move[1], i) and (worker_positions[move[0]] == i or not _is_occuppied(self.board[_coord(i)])) and _height(self.board[_coord(i)]) + 1 <= _NUM_FLOORS:
              self._legal_actions_cache.append((move[0], move[1], i))

        self._is_terminal = (len(self._legal_actions_cache) == 0) or any(_height(self.board[_coord(worker_position)]) == _NUM_FLOORS for worker_position in worker_positions)

        if self._is_terminal:
          self._returns_cache = [0., 0.]
          self._returns_cache[self._cur_player] = -1.0 if len(self._legal_actions_cache) == 0 else 1.0
          self._returns_cache[1 - self._cur_player] = -self._returns_cache[self._cur_player]
          self._player0_score = self._returns_cache[0]

    return self._legal_actions_cache

  def _apply_action(self, encoded_action):
    """Applies the specified action to the state."""
    action = self._decode_action(encoded_action)
    if self.num_workers_placed < 4:
      self.board[_coord(action[0])] = (1<<(_NUM_FLOOR_BITS + self._cur_player))
      self.board[_coord(action[1])] = (1<<(_NUM_FLOOR_BITS + self._cur_player))
      self.num_workers_placed += 2
    else:
      worker_positions = [i for i in range(_NUM_CELLS) if self.board[_coord(i)] & (1<<(_NUM_FLOOR_BITS + self._cur_player))]
      self.board[_coord(worker_positions[action[0]])] &= (1<<_NUM_FLOOR_BITS - 1)
      self.board[_coord(action[1])] |= (1<<(_NUM_FLOOR_BITS+self._cur_player))
      self.board[_coord(action[2])] += 1

    self._cur_player = 1 - self._cur_player
    self._legal_actions_cache = None

  def _action_to_string(self, player, encoded_action):
    """Action -> string."""
    action = self._decode_action(encoded_action)
    if self.num_workers_placed < 4:
      return f"{self._cur_player}{_coord(action[0])} {self._cur_player}{_coord(action[1])}"
    else:
      worker_positions = [i for i in range(_NUM_CELLS) if self.board[_coord(i)] & (1<<(_NUM_FLOOR_BITS + self._cur_player))]
      return f"{self._cur_player}[{_coord(worker_positions[action[0]])}->{_coord(action[1])}]@{_coord(action[2])}"

  def is_terminal(self):
    """Returns True if the game is over."""
    if self._legal_actions_cache is None:
      self._legal_actions_cache_lookup()
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if self.is_terminal():
      return self._returns_cache
    return [0., 0.]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)
  
  def _information_state_string(self, player):
    """Returns a string that uniquely encodes the information state of the player."""
    self._legal_actions_cache_lookup()
    return str(self._legal_actions_cache)

  def _encode_action(self, action):
    """Encodes an action as an integer."""
    if len(action) == 2:
      return bisect.bisect_left(_ALL_LEGAL_PLACEMENT_ACTIONS, action)
    else:
      worker_positions = [i for i in range(_NUM_CELLS) if self.board[_coord(i)] & (1<<(_NUM_FLOOR_BITS + self._cur_player))]
      coords = (_coord(worker_positions[action[0]]), _coord(action[1]), _coord(action[2]))
      direction_move = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
      direction_build = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
      return len(_ALL_LEGAL_PLACEMENT_ACTIONS) + action[0] * 64 + bisect.bisect_left(_ALL_DIRECTIONS, direction_move) * 8 + bisect.bisect_left(_ALL_DIRECTIONS, direction_build)

  def _decode_action(self, encoded_action):
    """Decodes an integer into an action."""
    if encoded_action < len(_ALL_LEGAL_PLACEMENT_ACTIONS):
      return _ALL_LEGAL_PLACEMENT_ACTIONS[encoded_action]
    else:
      encoded_action -= len(_ALL_LEGAL_PLACEMENT_ACTIONS)
      worker_positions = [i for i in range(_NUM_CELLS) if self.board[_coord(i)] & (1<<(_NUM_FLOOR_BITS + self._cur_player))]
      from_square = worker_positions[encoded_action // 64]
      direction_move = _ALL_DIRECTIONS[(encoded_action % 64) // 8]
      direction_build = _ALL_DIRECTIONS[encoded_action % 8]
      coords = _coord(from_square)
      to_square =  _move(*_add_tuples(coords, direction_move))
      build_square = _move(*_add_tuples(_add_tuples(coords, direction_move), direction_build))
      return (encoded_action // 64, to_square, build_square)



class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (1 + _NUM_FLOORS + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs.fill(0)
    player_mask = 1 << player
    for row in range(_NUM_ROWS):
      for col in range(_NUM_COLS):
        cell_height = _height(state.board[row, col])
        if cell_height > 0:
          obs[cell_height-1, row, col] = 1
        if _is_occuppied(state.board[row, col]):
          obs[1 + _NUM_FLOORS + (0 if ((state.board[row, col]>>_NUM_FLOOR_BITS) & player_mask) else 1), row, col] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)


# Helper functions for game details.


def _coord(move):
  """Returns (row, col) from an action id."""
  return (move // _NUM_COLS, move % _NUM_COLS)

def _move(row, col):
  """Returns an square id from (row, col)."""
  return row * _NUM_COLS + col

def _height(cell):
  """Returns the height of a cell."""
  return cell & ((1 << _NUM_FLOOR_BITS) - 1)

def _is_occuppied(cell):
  """Returns True if the cell is occupied."""
  return (cell >> _NUM_FLOOR_BITS) > 0

def _board_to_string(board):
  """Returns a string representation of the board."""
  height_strings = _visualize_heights(board)
  player_strings = _visualize_players(height_strings, board)
  return "\n".join(" ".join(row) for row in player_strings)

def _is_neighbour(cell1, cell2):
  """Returns True if the two cells are neighbours."""
  row1, col1 = _coord(cell1)
  row2, col2 = _coord(cell2)
  return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1 and cell1 != cell2

def _add_tuples(t1, t2):
  """Adds two tuples elementwise."""
  return tuple(x + y for x, y in zip(t1, t2))

_visualize_heights = np.frompyfunc(lambda i: ['⚬', '−', '=', '≡', '●'][_height(i)], 1, 1)
# _visualize_players = np.frompyfunc(lambda h, i: colored(h, 'red') if (i>>_NUM_FLOOR_BITS) & 1 else (colored(h, 'blue') if (i>>_NUM_FLOOR_BITS) & 2 else h), 2, 1)
_visualize_players = np.frompyfunc(lambda h, i: h+'A' if (i>>_NUM_FLOOR_BITS) & 1 else (h+'B' if (i>>_NUM_FLOOR_BITS) & 2 else h+' '), 2, 1)

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, SantoriniGame)
