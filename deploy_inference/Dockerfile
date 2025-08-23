# Use a lightweight Python base image
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Copy project metadata and source files for editable install
COPY deploy_inference/setup.py ./
COPY deploy_inference/inference_requirements.txt ./requirements.txt
COPY src/* ./src/
# Install the project in editable mode
RUN pip install -r requirements.txt -e .

# Copy model artifacts
COPY data/extracted_model/model.pkl ./data/extracted_model/model.pkl
COPY data/extracted_model/params.json ./data/extracted_model/params.json

# Copy inference entry point
COPY main_inference.py ./main_inference.py

EXPOSE 9696

# Define the default command
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "main_inference:app"]
