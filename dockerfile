FROM python:3.12-slim-bullseye

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pull-ollama-models.py .
COPY basic-chatbot.py .
COPY run-scripts.sh .

# Give execution permissions to the shell script
RUN chmod +x run-scripts.sh

CMD ["./run-scripts.sh"]
