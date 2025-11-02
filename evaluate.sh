#!/bin/bash

# AlphaZero Evaluation Script for OpenSpiel (Interactive Version)
# Usage: ./evaluate.sh
# This script will prompt you for all necessary parameters

set -e

echo "=============================================="
echo "🎯 AlphaZero Evaluation Setup"
echo "=============================================="
echo ""

# Function to prompt for input with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local result
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " result
        echo "${result:-$default}"
    else
        read -p "$prompt: " result
        echo "$result"
    fi
}

# Game selection
echo "📋 Available games:"
echo "   Simple:     tic_tac_toe, connect_four, hex, othello"
echo "   Complex:    chess, go, breakthrough, amazons"
echo "   Card games: leduc_poker, kuhn_poker"
echo "   Others:     quoridor, santorini, kriegspiel"
echo ""
GAME_NAME=$(prompt_with_default "Enter game name" "connect_four")

# Function to list available models for the selected game
list_available_models() {
    local game="$1"
    echo ""
    echo "🎯 Available models for $game:"
    
    local models_found=0
    if [ -d "models/$game" ]; then
        for model_dir in models/$game/*/; do
            if [ -d "$model_dir" ] && [ -f "$model_dir/config.json" ]; then
                local model_name=$(basename "$model_dir")
                echo "   - $model_name"
                models_found=1
            fi
        done
    fi
    
    if [ $models_found -eq 0 ]; then
        echo "   (No trained models found for $game)"
    fi
}

# Show available models for the selected game
list_available_models "$GAME_NAME"

# Model name selection
echo ""
echo "💾 Model Selection:"
# Find the most recently modified model directory
LATEST_MODEL=""
if [ -d "models/$GAME_NAME" ]; then
    LATEST_MODEL=$(find "models/$GAME_NAME" -maxdepth 1 -type d -name "*" ! -name "." ! -name ".." -exec stat -c "%Y %n" {} \; 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- | xargs basename 2>/dev/null || echo "")
fi

MODEL_NAME=$(prompt_with_default "Model name" "${LATEST_MODEL}")

# Construct model directory
MODEL_DIR="models/${GAME_NAME}/${MODEL_NAME}"

# Check if the model exists in the new structure, fallback to legacy or custom path
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo ""
    echo "⚠️  Model not found at: $MODEL_DIR"
    echo ""
    echo "🔍 Searching for alternative locations..."
    
    # Look for the model in legacy locations
    LEGACY_FOUND=""
    if [ -d "/tmp/az_${GAME_NAME}" ] && [ -f "/tmp/az_${GAME_NAME}/config.json" ]; then
        LEGACY_FOUND="/tmp/az_${GAME_NAME}"
    elif [ -d "/tmp/${MODEL_NAME}" ] && [ -f "/tmp/${MODEL_NAME}/config.json" ]; then
        LEGACY_FOUND="/tmp/${MODEL_NAME}"
    fi
    
    if [ -n "$LEGACY_FOUND" ]; then
        echo "   Found legacy model at: $LEGACY_FOUND"
        USE_LEGACY=$(prompt_with_default "Use this legacy model? [Y/n]" "Y")
        if [[ ! "$USE_LEGACY" =~ ^[Nn] ]]; then
            MODEL_DIR="$LEGACY_FOUND"
        else
            MODEL_DIR=$(prompt_with_default "Enter custom model directory path" "")
        fi
    else
        echo "   No alternative found."
        MODEL_DIR=$(prompt_with_default "Enter custom model directory path" "")
    fi
fi

# Validate model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory '$MODEL_DIR' does not exist!"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ Error: No config.json found in '$MODEL_DIR'!"
    echo "   This doesn't appear to be a valid AlphaZero model directory."
    exit 1
fi

# Opponent selection
echo ""
echo "🤖 Opponent Types:"
echo "   1. human  - Play interactively against AlphaZero"
echo "   2. mcts   - Evaluate AlphaZero vs MCTS bot"
echo "   3. random - Evaluate AlphaZero vs random player"
echo ""
OPPONENT_CHOICE=$(prompt_with_default "Select opponent type (1-3)" "3")

case "$OPPONENT_CHOICE" in
    "1") OPPONENT="human" ;;
    "2") OPPONENT="mcts" ;;
    "3") OPPONENT="random" ;;
    *) OPPONENT="mcts" ;;
esac

# Evaluation parameters
echo ""
echo "⚙️ Evaluation Parameters:"

# Number of games
if [ "$OPPONENT" = "human" ]; then
    NUM_GAMES=1
    echo "ℹ️  Interactive mode - playing 1 game"
else
    # Suggest number of games based on opponent type
    NUM_GAMES=$(prompt_with_default "Number of games to play" "10")
fi

# MCTS simulations for evaluation (if not playing against human)
if [ "$OPPONENT" = "mcts" ]; then
    echo ""
    echo "🔄 MCTS Configuration:"
    echo "   - Higher simulations = stronger play but slower evaluation"
    echo "   - Lower simulations = faster evaluation but potentially weaker play"
    

    DEFAULT_SIMULATIONS="10"
    
    MCTS_SIMULATIONS=$(prompt_with_default "MCTS simulations per move" "10")
fi

echo ""
echo "=============================================="
echo "📋 Evaluation Summary"
echo "=============================================="
echo "Game:           $GAME_NAME"
echo "Model Name:     $(basename "$MODEL_DIR")"
echo "Model Path:     $MODEL_DIR"
echo "Opponent:       $OPPONENT"
echo "Games:          $NUM_GAMES"
if [ "$OPPONENT" = "mcts" ]; then
    echo "MCTS Sims:      $MCTS_SIMULATIONS"
fi
echo "=============================================="
echo ""

# read -p "Start evaluation with these settings? [Y/n]: " CONFIRM
# if [[ "$CONFIRM" =~ ^[Nn] ]]; then
#     echo "Evaluation cancelled."
#     exit 0
# fi

echo ""
echo "🎯 Starting AlphaZero Evaluation..."
echo ""

run_game() {
    local player1="$1"
    local player2="$2"
    local cmd_args="$3"

    # echo "Running game: $player1 vs $player2. cmd_args: $cmd_args"
    
    RESULT=$(./build/examples/alpha_zero_torch_game_example \
        --player1="$player1" \
        --player2="$player2" \
        $cmd_args 2>&1 | tail -n 1)
    
    # Parse the reward from the result
    if [[ "$RESULT" =~ Overall\ returns:\ ([^ ,]+),\ ([^ ,]+) ]]; then
        left_reward="${BASH_REMATCH[1]}"
        right_reward="${BASH_REMATCH[2]}"
        echo " $player1: $left_reward | $player2: $right_reward"

        # Determine winner based on rewards and which player is AlphaZero
        if [ "$left_reward" = "$right_reward" ]; then
            echo "0"  # Draw
        elif [ "$left_reward" -gt "$right_reward" ]; then
            # Player 1 won
            if [ "$player1" = "az" ]; then
                echo "1"  # AlphaZero won
            else
                echo "-1"  # AlphaZero lost
            fi
        else
            # Player 2 won
            if [ "$player2" = "az" ]; then
                echo "1"  # AlphaZero won
            else
                echo "-1"  # AlphaZero lost
            fi
        fi
    else
        echo "Could not parse result!"
        echo "0"  # Default to draw on error
    fi
}

WINS=0
LOSSES=0
DRAWS=0

if [ "$OPPONENT" = "human" ]; then
    echo "🎮 Starting interactive play against AlphaZero!"
    echo "   You are Player 1, AlphaZero is Player 2"
    echo "   Follow the prompts to make your moves."
    echo ""
    # Set variables PLAYER1 and PLAYER2 by tossing a coin
    if [ $((RANDOM % 2)) -eq 0 ]; then
        PLAYER1="human"
        PLAYER2="az"
    else
        PLAYER1="az"
        PLAYER2="human"
    fi

    # Run the game and get the winner
    echo "Player 1: $PLAYER1 | Player 2: $PLAYER2"
    RESULT=$(run_game "$PLAYER1" "$PLAYER2" "--game=$GAME_NAME --az_path=$MODEL_DIR --az_checkpoint=-1")
    SCORES=$(echo "$RESULT" | head -n 1)
    echo "Scores: $SCORES"
    WINNER=$(echo "$RESULT" | tail -n 1)
    
    if [ "$WINNER" = "1" ]; then
        WINS=$((WINS + 1))
        echo "WIN"
    elif [ "$WINNER" = "-1" ]; then
        LOSSES=$((LOSSES + 1))
        echo "LOSS"
    else
        DRAWS=$((DRAWS + 1))
        echo "DRAW"
    fi

elif [ "$OPPONENT" = "random" ] || [ "$OPPONENT" = "mcts" ]; then
    echo "🎲 Evaluating AlphaZero vs ${OPPONENT} opponent..."
    echo "   Running $NUM_GAMES games..."
    if [ "$OPPONENT" = "mcts" ]; then
        echo "   Using $MCTS_SIMULATIONS MCTS simulations..."
    fi
    echo ""
    
    # Run multiple games and collect results for random or mcts opponent
    WINS=0
    LOSSES=0
    DRAWS=0

    CMD_ARGS="--game=$GAME_NAME --az_path=$MODEL_DIR --az_checkpoint=-1"
    if [ "$OPPONENT" = "mcts" ]; then
        CMD_ARGS="$CMD_ARGS --max_simulations=$MCTS_SIMULATIONS"
    fi
    
    for ((i=1; i<=NUM_GAMES; i++)); do
        echo ""
        echo "Game $i/$NUM_GAMES... "

        # Alternate who goes first
        if [ $((i % 2)) -eq 1 ]; then
            PLAYER1="az"
            PLAYER2="$OPPONENT"
        else
            PLAYER1="$OPPONENT"
            PLAYER2="az"
        fi
        echo "Player 1: $PLAYER1 | Player 2: $PLAYER2"
        RESULT=$(run_game "$PLAYER1" "$PLAYER2" "$CMD_ARGS")
        SCORES=$(echo "$RESULT" | head -n 1)
        echo "Scores: $SCORES"
        WINNER=$(echo "$RESULT" | tail -n 1)
        
        if [ "$WINNER" = "1" ]; then
            WINS=$((WINS + 1))
            echo "WIN"
        elif [ "$WINNER" = "-1" ]; then
            LOSSES=$((LOSSES + 1))
            echo "LOSS"
        else
            DRAWS=$((DRAWS + 1))
            echo "DRAW"
        fi
    done
fi

# Show final results
echo ""
echo "=============================================="
echo "📊 Evaluation Results vs ${OPPONENT^}"
echo "=============================================="
echo "Total Games:    $NUM_GAMES"
echo "Wins:           $WINS ($(( (WINS * 100) / NUM_GAMES ))%)"
echo "Losses:         $LOSSES ($(( (LOSSES * 100) / NUM_GAMES ))%)"
echo "Draws:          $DRAWS ($(( (DRAWS * 100) / NUM_GAMES ))%)"
echo "Win Rate:       $(( (WINS * 100) / NUM_GAMES ))%"
echo "=============================================="

echo ""
echo "=============================================="
echo "✅ Evaluation completed!"
echo "Game: $GAME_NAME | Model: $(basename "$MODEL_DIR")"
echo ""
echo "Next steps:"
echo "  Train more:       ./train.sh"
echo "  Try different:    ./evaluate.sh (different opponent/settings)"
echo "=============================================="

echo ""
echo "✅ Evaluation completed!"