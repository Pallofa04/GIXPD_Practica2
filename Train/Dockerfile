# Dockerfile: Use an official Python runtime as a parent image
FROM python:3.6-slim
# Set the working directory to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
ADD . /app
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.pyhton.org -r requirements-train.txt
# Define environment variable
ENV MODEL_PATH=/app/model.npy
# Run main-train.py when the container launches
CMD ["python", "main-train.py"]
