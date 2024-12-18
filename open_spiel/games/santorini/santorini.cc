// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/santorini/santorini.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"
#include "santorini.h"

namespace open_spiel {
namespace santorini {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"santorini",
    /*long_name=*/"Santorini",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SantoriniGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

int Height(CellState cell) {
  return cell & ((1 << kNumFloorBits) - 1);
}

char CellStateToChar(CellState state) {
  // std::string str, height_char = " ";
  // switch (Height(state))
  // {
  // case 0: height_char = "⚬"; break;
  // case 1: height_char = "−"; break;
  // case 2: height_char = "="; break;
  // case 3: height_char = "≡"; break;
  // case 4: height_char = "●"; break;
  // default: break;
  // }
  char base_char = '0';
  int h = Height(state);
  if(state >> kNumFloorBits == 1) {
    return 'a' + h;
  } else if(state >> kNumFloorBits == 2) {
    return 'A' + h;
  } else {
    return '0' + h;
  }
}

bool IsOccupied(CellState cell) {
  return (cell >> kNumFloorBits) > 0;
}

bool IsNeighbour(CellState cell1, CellState cell2) {
  auto [row1, col1] = Coord(cell1);
  auto [row2, col2] = Coord(cell2);
  return abs(row1 - row2) <= 1 && abs(col1 - col2) <= 1 && cell1 != cell2;
}

std::pair<int, int> Coord(CellState cell) {
  return {cell / kNumCols, cell % kNumCols};
}

std::array<std::pair<CellState, CellState>, kNumPlacementActions> GeneratePlacementActionWorkerPositions() {
  std::array<std::pair<CellState, CellState>, kNumPlacementActions> positions;
  int index = 0;
  for (CellState i = 0; i < kNumCells; ++i) {
    for (CellState j = i + 1; j < kNumCells; ++j) {
      positions[index++] = {i, j};
    }
  }
  return positions;
}


void SantoriniState::DoApplyAction(Action action_id) {
  auto action = SantoriniAction(action_id);
  if (num_workers_placed_ < 4) {
    auto [worker1, worker2] = kPlacementActionWorkerPositions[action.action()];
    board_[worker1] = (1 << (kNumFloorBits + current_player_));
    board_[worker2] = (1 << (kNumFloorBits + current_player_));
    worker_positions_[current_player_] = {worker1, worker2};
    num_workers_placed_ += 2;
  } else {
    int worker_id = action.worker_id();
    CellState from_position = (worker_id == 0)? worker_positions_[current_player_].first : worker_positions_[current_player_].second;
    CellState to_position = from_position + action.move_direction().first * kNumCols + action.move_direction().second;
    CellState build_position = to_position + action.build_direction().first * kNumCols + action.build_direction().second;
    board_[from_position] = board_[from_position] & ((1 << kNumFloorBits) - 1);
    board_[to_position] = board_[to_position] | (1 << (kNumFloorBits + current_player_));
    board_[build_position] += 1;
    if(worker_id == 0) {
      worker_positions_[current_player_].first = to_position;
    } else {
      worker_positions_[current_player_].second = to_position;
    }
    // (worker_id == 0)? worker_positions_[current_player_].first : worker_positions_[current_player_].second = to_position;
    if (worker_positions_[current_player_].first > worker_positions_[current_player_].second) {
      std::swap(worker_positions_[current_player_].first, worker_positions_[current_player_].second);
    }
    if(Height(board_[to_position]) == kNumFloors) {
      outcome_ = current_player_;
    }
  }

  current_player_ = 1 - current_player_;
  SetLegalActions();
}

void SantoriniState::SetLegalActions() {
  legal_actions_.clear();
  if(outcome_ != kInvalidPlayer){
    return;
  }
  if(num_workers_placed_ < 4){
    for (int i = 0; i < kNumPlacementActions; ++i) {
      if (board_[kPlacementActionWorkerPositions[i].first] == 0 &&
          board_[kPlacementActionWorkerPositions[i].second] == 0) {
        legal_actions_.push_back(i);
      }
    }
  } else if(
      (Height(board_[worker_positions_[current_player_].first]) < kNumFloors) &&
      (Height(board_[worker_positions_[current_player_].second]) < kNumFloors)
    )
  {
    // Iterate over all workers and all possible moves and builds, and check if they are legal.
    for (int worker_id = 0; worker_id < 2; ++worker_id) {
      CellState from_position = (worker_id == 0) ? worker_positions_[current_player_].first : worker_positions_[current_player_].second;
      for (int move_direction_id = 0; move_direction_id < kDirections.size(); ++move_direction_id) {
        const auto& move_direction = kDirections[move_direction_id];
        auto [from_x, from_y] = Coord(from_position);
        auto to_x = from_x + move_direction.first, to_y = from_y + move_direction.second;
        if (to_x < 0 || to_x >= kNumRows || to_y < 0 || to_y >= kNumCols) {
          continue;
        }
        CellState to_position = to_x * kNumCols + to_y;
        if (IsOccupied(board_[to_position]) || Height(board_[to_position]) > Height(board_[from_position]) + 1) {
          continue;
        }
        for (int build_direction_id = 0; build_direction_id < kDirections.size(); ++build_direction_id) {
          const auto& build_direction = kDirections[build_direction_id];
          auto build_x = to_x + build_direction.first, build_y = to_y + build_direction.second;
          if(build_x < 0 || build_x >= kNumRows || build_y < 0 || build_y >= kNumCols) {
            continue;
          }
          CellState build_position = build_x * kNumCols + build_y;
          if ((IsOccupied(board_[build_position]) && (move_direction_id + build_direction_id != 7)) || Height(board_[build_position]) == kNumFloors + 1) {
            continue;
          }
          legal_actions_.push_back(SantoriniAction(worker_id, move_direction_id, build_direction_id).action());
        }
      }
    }
  }
  if (legal_actions_.size() == 0) {
    outcome_ = 1 - current_player_;
  }
}


std::vector<Action> SantoriniState::LegalActions() const {
  return legal_actions_;
}

std::string SantoriniState::ActionToString(Player player, Action action_id) const {
  SantoriniAction action(action_id);
  return action.ToString();
}

Action SantoriniState::StringToAction(Player player, const std::string& action_str) const{
  return SantoriniAction(action_str).action();
}

SantoriniState::SantoriniState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), 0);
  SetLegalActions();
}

std::string SantoriniState::ToString() const {
  char str[kNumCells + kNumRows - 1];
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      str[r * (kNumCols + 1) + c] = CellStateToChar(board_[r * kNumCols + c]);
    }
    if (r < (kNumRows - 1)) {
      str[(r + 1) * (kNumCols + 1) - 1] = '\n';
    }
  }
  return std::string(str);
}

bool SantoriniState::IsTerminal() const {
  return outcome_ != kInvalidPlayer;
}

std::vector<double> SantoriniState::Returns() const {
  auto returns = std::vector<double>(num_players_, 0.);
  if(outcome_ != kInvalidPlayer) {
    returns[outcome_] = 1.0;
    returns[1 - outcome_] = -1.0;
  } else if(LegalActions().size() == 0) {
    returns[current_player_] = -1.0;
    returns[1 - current_player_] = 1.0;
  }
  return returns;
}

std::string SantoriniState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string SantoriniState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void SantoriniState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<3> view(values, {kCellStates, kNumRows, kNumCols}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    auto [i, j] = Coord(cell);
    auto h = Height(board_[cell]);
    if(h > 0) {
      view[{h - 1, i, j}] = 1.0;
      }
    if (IsOccupied(board_[cell])){
      auto k = ((board_[cell] >> kNumFloorBits) == 1) == (current_player_ == 0) ? 0 : 1;
      view[{kNumFloors + 1 + k, i, j}] = (float) h;
    }
  }
}

std::unique_ptr<State> SantoriniState::Clone() const {
  return std::unique_ptr<State>(new SantoriniState(*this));
}

std::string SantoriniGame::ActionToString(Player player,
                                          Action action_id) const
{
  SantoriniAction action(action_id);
  return action.ToString();
}

SantoriniGame::SantoriniGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace santorini
}  // namespace open_spiel