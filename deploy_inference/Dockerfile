# Use a lightweight Python base image
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install third-party Python deps (cache keyed only by requirements.txt)
COPY deploy_inference/inference_requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy project metadata and source files for editable install
COPY deploy_inference/setup.py ./
COPY src/* ./src/
# Install the project in editable mode
RUN pip install -e .

# Copy model artifacts and late-changing assets
COPY data/extracted_model/model.pkl ./data/extracted_model/model.pkl
COPY data/extracted_model/params.json ./data/extracted_model/params.json
COPY main_inference.py ./main_inference.py

EXPOSE 9696

# Define the default command
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "--timeout", "60", "main_inference:app"]
