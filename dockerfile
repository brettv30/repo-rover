FROM python:3.12-slim-bullseye

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pull-ollama-models.py .
COPY basic-chatbot.py .
COPY run-scripts.sh .
COPY entrypoint.sh .

# Give execution permissions to the shell script
RUN chmod +x run-scripts.sh entrypoint.sh

# Use an environment variable to determine whether to run in development or production mode
ENV RUN_Mode=development

ENTRYPOINT ["./entrypoint.sh"]