# Stage 1: Build dependencies
FROM python:3.9-slim AS builder
WORKDIR /WasteApp

# Install system dependencies including build tools
RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies with increased timeout & retries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim
WORKDIR /WasteApp

# Install necessary system dependencies in the final image
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and executables from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of your application code
COPY . .

# Expose the port and run the application
EXPOSE 5000
CMD ["python", "app.py"]