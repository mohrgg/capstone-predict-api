# Use Python 3.9-slim as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt /app/requirements.txt

# Install the necessary packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for protocol buffers
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Set environment variable to use tf_keras with TensorFlow
ENV TF_USE_LEGACY_KERAS=1

# Copy the entire project to the working directory
COPY . /app

# Expose port 8080 to the outside world
EXPOSE 8080

# Define the command to run the application
CMD ["python", "api/app.py"]
