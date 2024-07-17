import os
from ollama import Client
import time

# Get the Ollama host from environment variable
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = Client(host=ollama_host)


def is_model_available(model_name):
    try:
        # Try to get model info
        client.show(model_name)
        return True
    except Exception:
        return False


def pull_model_with_retry(model_name, max_retries=5, delay=60):
    if is_model_available(model_name):
        print(f"Model {model_name} is already available. Skipping pull.")
        return

    for attempt in range(max_retries):
        try:
            print(f"Attempting to pull {model_name}...")
            client.pull(model_name)
            print(f"Successfully pulled {model_name}")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to pull {model_name} after {max_retries} attempts")


# List of models to pull
models = ["phi3:3.8b"]

# Pull the models from the Ollama server if not already available
for model in models:
    pull_model_with_retry(model)

print("All model checks and pulls completed")
