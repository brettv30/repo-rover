#!/bin/bash

if [ "$RUN_MODE" = "development" ]; then
    # Run with uvicorn for hot-reloading in development
    exec uvicorn basic-chatbot:app --host 0.0.0.0 --port 420 --reload
else
    # Run the original script in production
    exec ./run-scripts.sh
fi