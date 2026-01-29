# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app/SharedBackend/src"

# Install system dependencies (needed for psycopg2/asyncpg sometimes, but slim usually ok for wheels)
# If reportlab needs fonts or libs, we might need:
# RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SharedBackend dependencies
COPY SharedBackend/requirements.txt ./requirements-shared.txt
RUN pip install --no-cache-dir -r requirements-shared.txt

# Copy the rest of the application
COPY . .



# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
