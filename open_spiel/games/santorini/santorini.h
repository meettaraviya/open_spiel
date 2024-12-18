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

#ifndef OPEN_SPIEL_GAMES_SANTORINI_H_
#define OPEN_SPIEL_GAMES_SANTORINI_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace santorini {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 5;
inline constexpr int kNumCols = 5;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kNumFloors = 3;
inline constexpr int kNumFloorBits = 3;
inline constexpr int kCellStates = 1 + kNumFloors + kNumPlayers;
inline constexpr int kNumPlacementActions = (kNumCells * (kNumCells - 1)) / 2;
inline constexpr int kNumDistinctActions = kNumPlacementActions + 2 * 8 * 8;  // Adjust as needed
inline constexpr std::array<std::pair<short, short>, 8> kDirections = {
  std::make_pair(static_cast<short>(-1), static_cast<short>(-1)), std::make_pair(static_cast<short>(-1), static_cast<short>(0)), std::make_pair(static_cast<short>(-1), static_cast<short>(1)),
  std::make_pair(static_cast<short>(0), static_cast<short>(-1)), std::make_pair(static_cast<short>(0), static_cast<short>(1)),
  std::make_pair(static_cast<short>(1), static_cast<short>(-1)), std::make_pair(static_cast<short>(1), static_cast<short>(0)), std::make_pair(static_cast<short>(1), static_cast<short>(1))
};

inline constexpr std::array<const char*, 8> kDirectionSymbols = {
  "7", "8", "9", "4", "6", "1", "2", "3"
};
// State of a cell.
using CellState = int;

std::array<std::pair<CellState, CellState>, kNumPlacementActions> GeneratePlacementActionWorkerPositions();
inline const std::array<std::pair<CellState, CellState>, kNumPlacementActions> kPlacementActionWorkerPositions = GeneratePlacementActionWorkerPositions();

// Helper functions
int Height(CellState cell);
bool IsOccupied(CellState cell);
bool IsNeighbour(CellState cell1, CellState cell2);
std::pair<int, int> Coord(CellState cell);
char CellStateToChar(CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << CellStateToChar(state);
}

enum class SantoriniActionType {
  kPlacement,
  kMoveAndBuild,
};

class SantoriniAction {
 public:
  SantoriniAction() : action_(0) {}
  explicit SantoriniAction(Action action) : action_(action) {}
  explicit SantoriniAction(std::string action_str) {
      if (action_str[0] == 'P') {
        // Parse placement action
        int cell1_x = action_str[1] - '0';
        int cell1_y = action_str[2] - '0';
        int cell2_x = action_str[3] - '0';
        int cell2_y = action_str[4] - '0';
        CellState cell1 = cell1_x * kNumCols + cell1_y;
        CellState cell2 = cell2_x * kNumCols + cell2_y;
        action_ = (cell2 - 1) + (kNumCells - 2) * cell1 - cell1 * (cell1 - 1) / 2;
      } else if (action_str[1] == 'M' && action_str[3] == 'B') {
        // Parse move and build action
        int worker_id = action_str[0] - '0';
        std::string move_symbol(1, action_str[2]);
        std::string build_symbol(1, action_str[4]);
        int move_direction_id = std::distance(kDirectionSymbols.begin(), std::find(kDirectionSymbols.begin(), kDirectionSymbols.end(), move_symbol));
        int build_direction_id = std::distance(kDirectionSymbols.begin(), std::find(kDirectionSymbols.begin(), kDirectionSymbols.end(), build_symbol));
        action_ = worker_id * 64 + move_direction_id * 8 + build_direction_id + kNumPlacementActions;
      } else {
        throw std::invalid_argument("Invalid action string");
      }
  }
  explicit SantoriniAction(CellState cell1, CellState cell2) : action_((cell2 - 1) + (kNumCells - 2) * cell1 - cell1 * (cell1 - 1) / 2) {}
  explicit SantoriniAction(int worker_id, int move_direction_id, int build_direction_id)
      : action_(worker_id * 64 + move_direction_id * 8 + build_direction_id + kNumPlacementActions) {}

  SantoriniActionType action_type() const { return (action_ < kNumPlacementActions) ? SantoriniActionType::kPlacement : SantoriniActionType::kMoveAndBuild; }
  int worker_id() const { return (action_ - kNumPlacementActions) / 64; }
  std::pair<short, short> move_direction() const { return kDirections[((action_ - kNumPlacementActions) % 64) / 8]; }
  std::pair<short, short> build_direction() const { return kDirections[(action_ - kNumPlacementActions) % 8]; }
  int action() const { return action_; }
  std::string ToString() const {
    if (action_type() == SantoriniActionType::kPlacement) {
      auto [cell1, cell2] = kPlacementActionWorkerPositions[action_];
      auto [cell1_x, cell1_y] = Coord(cell1);
      auto [cell2_x, cell2_y] = Coord(cell2);
      // format "P{cell1_row}{cell1_col}{cell2_row}{cell2_col}"
      return "P" + std::to_string(cell1_x) + std::to_string(cell1_y) +
         std::to_string(cell2_x) + std::to_string(cell2_y);
    } else {
      // format "MB{cell1_row}{cell1_col}{cell2_row}{cell2_col}"
      auto move_direction_id = ((action_ - kNumPlacementActions) % 64) / 8;
      auto build_direction_id = (action_ - kNumPlacementActions) % 8;
      return std::to_string(worker_id()) + "M" + kDirectionSymbols[move_direction_id] + "B" + kDirectionSymbols[build_direction_id];
    }
  }
 private:
  Action action_;
};

// State of an in-play game.
class SantoriniState : public State {
 public:
  SantoriniState(std::shared_ptr<const Game> game);

  SantoriniState(const SantoriniState&) = default;
  SantoriniState& operator=(const SantoriniState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  Action StringToAction(Player player, const std::string& action_str) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  void SetLegalActions();
  Player outcome() const { return outcome_; }

 protected:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;

 private:
  Player current_player_ = 0;         // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_workers_placed_ = 0;
  int num_moves_ = 0;
  std::vector<Action> legal_actions_ = std::vector<Action>(0);
  std::vector<std::pair<CellState, CellState>> worker_positions_ = std::vector<std::pair<CellState, CellState>>(kNumPlayers);
};

// Game object.
class SantoriniGame : public Game {
 public:
  explicit SantoriniGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new SantoriniState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kNumPlayers * 2 + kNumCells * (kNumFloors + 1); }
  std::string ActionToString(Player player, Action action_id) const override;
};

}  // namespace santorini
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SANTORINI_H_