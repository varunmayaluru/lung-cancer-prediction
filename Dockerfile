# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install supervisor
RUN apt-get update && apt-get install -y supervisor

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requierments.txt

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Start the supervisor to manage FastAPI and Streamlit
CMD ["/usr/bin/supervisord"]
