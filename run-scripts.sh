#!/bin/bash
set -e

# Run the first script
python pull-ollama-models.py

# Run the second script
python initial-llama-bot.py
