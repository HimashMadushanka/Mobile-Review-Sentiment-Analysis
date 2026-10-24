FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing bytecode files
ENV PYTHONDONTWRITEBYTECODE=1

# Show logs immediately
ENV PYTHONUNBUFFERED=1

# Keep Hugging Face from trying to download the model
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Copy the Docker-specific requirements
COPY requirements-docker.txt .

# Install the API and Transformers dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the API source code
COPY app/ ./app/
COPY src/ ./src/

# Copy the trained BERT model and tokenizer
COPY models/bert/fine-tuned/ ./models/bert/fine-tuned/

# Document the port used by the API
EXPOSE 8000

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]