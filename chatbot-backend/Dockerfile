# Dockerfile for the Chatbot Backend (FastAPI service)

# --- Base Image ---
# Use an official Python runtime as a parent image.
# The slim-bullseye version is a good balance of size and functionality.
FROM python:3.11-slim-bullseye

# --- Environment Variables ---
# Set environment variables to prevent Python from writing .pyc files to disk
# and to ensure output is sent straight to the terminal without buffering.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Working Directory ---
# Set the working directory inside the container.
WORKDIR /app

# --- Install Dependencies ---
# First, copy only the requirements.txt file to leverage Docker's layer caching.
# This way, dependencies are only re-installed if the requirements.txt file changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Unprivileged user ---
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app

# --- Copy Application Code ---
# Application, plus what the maintenance scripts (ingestion, replay, evaluation,
# dataset export) and the MCP prototype import.
COPY --chown=app:app ./app /app/app
COPY --chown=app:app ./scripts /app/scripts
COPY --chown=app:app ./eval /app/eval
COPY --chown=app:app ./training /app/training
COPY --chown=app:app ./mcp_server /app/mcp_server

USER app

# --- Expose Port ---
# Expose the port the app will run on.
EXPOSE 8000

# --- Run Application ---
# The command to run the FastAPI application using Uvicorn.
# --host 0.0.0.0 makes the server accessible from outside the container.
# --port 8000 matches the exposed port.
# No --reload here: development auto-reload is enabled in infra/docker-compose.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
