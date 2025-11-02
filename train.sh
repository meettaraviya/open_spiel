#!/bin/bash

# AlphaZero Training Script for OpenSpiel (Interactive Version)
# Usage: ./train.sh
# This script will prompt you for all necessary parameters

set -e

echo "=============================================="
echo "🚀 AlphaZero Training Setup"
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
GAME_NAME=$(prompt_with_default "Enter game name" "tic_tac_toe")

# Model name
echo ""
echo "💾 Model Configuration:"
MODEL_NAME=$(prompt_with_default "Model name (for organization)" "model_$(date +%Y%m%d_%H%M%S)")

# Construct output directory
OUTPUT_DIR="models/${GAME_NAME}/${MODEL_NAME}"

# Training parameters
echo ""
echo "🎯 Training Parameters:"
MAX_STEPS=$(prompt_with_default "Maximum training steps (0 = unlimited)" "1000")
ACTORS=$(prompt_with_default "Number of actors (parallel processes)" "4")
MAX_SIMULATIONS=$(prompt_with_default "MCTS simulations per move" "300")

# Network architecture selection
echo ""
echo "🧠 Network Architecture:"
echo "   mlp:     Simple multi-layer perceptron (fast, good for simple games)"
echo "   resnet:  Residual network (best performance, slower)"
echo ""

# Auto-suggest network based on game

DEFAULT_NN_MODEL="mlp"
DEFAULT_NN_WIDTH="64"
DEFAULT_NN_DEPTH="4"

NN_MODEL=$(prompt_with_default "Network architecture" "$DEFAULT_NN_MODEL")
NN_WIDTH=$(prompt_with_default "Network width" "$DEFAULT_NN_WIDTH")
NN_DEPTH=$(prompt_with_default "Network depth" "$DEFAULT_NN_DEPTH")

# Show summary and confirm
echo ""
echo "=============================================="
echo "📋 Training Summary"
echo "=============================================="
echo "Game:           $GAME_NAME"
echo "Model Name:     $MODEL_NAME"
echo "Output Path:    $OUTPUT_DIR"
echo "Max Steps:      $MAX_STEPS"
echo "Actors:         $ACTORS"
echo "Simulations:    $MAX_SIMULATIONS"
echo "Network:        $NN_MODEL ($NN_WIDTH x $NN_DEPTH)"
echo "=============================================="
echo ""

# read -p "Start training with these settings? [Y/n]: " CONFIRM
# if [[ "$CONFIRM" =~ ^[Nn] ]]; then
#     echo "Training cancelled."
#     exit 0
# fi

echo ""
echo "🚀 Starting AlphaZero Training..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the command
CMD="./build/examples/alpha_zero_torch_example"
CMD="$CMD --game=$GAME_NAME"
CMD="$CMD --path=$OUTPUT_DIR"
CMD="$CMD --max_steps=$MAX_STEPS"
CMD="$CMD --actors=$ACTORS"
CMD="$CMD --max_simulations=$MAX_SIMULATIONS"
CMD="$CMD --nn_model=$NN_MODEL"
CMD="$CMD --nn_width=$NN_WIDTH"
CMD="$CMD --nn_depth=$NN_DEPTH"
CMD="$CMD --verbose"

echo "Running: $CMD"
echo ""

# Run the training
$CMD

echo ""
echo "=============================================="
echo "✅ Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "Game: $GAME_NAME | Model: $MODEL_NAME"
echo ""
echo "Next steps:"
echo "  Resume training:  ./train.sh (choose same game and model name)"
echo "  Evaluate model:   ./evaluate.sh"
echo "=============================================="