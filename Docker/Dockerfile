# Use a lightweight Python base image
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Copy project metadata and source files for editable install
COPY Docker/setup.py ./
COPY Docker/inference_requirements.txt ./requirements.txt
COPY src/__init__.py ../src/data.py ./src/
# Install the project in editable mode
RUN pip install -r requirements.txt -e .

# Copy model artifacts
COPY deployment/model.pkl ./deployment/model.pkl
COPY deployment/params.json ./deployment/params.json

# Copy inference entry point
COPY main_inference.py ./main_inference.py

EXPOSE 9696

# Define the default command
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "main_inference:app"]
