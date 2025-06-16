# Use a lightweight Python base image
FROM python:3.12-slim-bookworm

# Set working directory in the container
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't store build cache
# --upgrade pip: Ensure pip is up-to-date
# Set global PATH for pip installed packages
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the application code
COPY main.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]