# Use a lightweight Python base image that matches the environment where your MLflow model was logged
# Airflow's default is often Python 3.12, so let's stick with that for consistency.
FROM python:3.12-slim-bookworm

# Set labels for image metadata (optional but good practice)
LABEL maintainer="mich910116@gmail.com"
LABEL version="1.0.0"
LABEL description="FastAPI service for Popularity-Based Recommendations"

# Set working directory inside the container
WORKDIR /app

# 1. Copy dependency files first to leverage Docker's build cache
# This means if only your code changes, but dependencies don't, Docker can reuse the layer.
COPY requirements.txt .

# 2. Install Python dependencies
# --no-cache-dir: Don't store build cache, reducing image size.
# --upgrade pip: Ensure pip is up-to-date.
# Set global PATH for pip installed packages in a standard user location
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip # Clean up pip cache to minimize image size

# 3. Copy your application code
# This copies main.py and the config folder/file relative to your project structure.
# Adjust the COPY source path if your 'config' folder is not directly next to 'main.py'
COPY main.py .

# 4. Expose the port FastAPI will run on
EXPOSE 8000

# 5. Command to run the application using Uvicorn
# Use the "exec" form for CMD to ensure signals are handled correctly (e.g., for graceful shutdown)
# --host 0.0.0.0 makes the API accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]