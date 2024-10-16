# Dockerfile: Use an official Python runtime as a parent image
FROM python:3.6-slim
# Set the working directory to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
ADD . /app
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.pyhton.org -r requirements-server.txt
# Define environment variable
ENV MODEL_PATH=/app/model.npy
# Export port 8000, as it's where Flask's listening
EXPOSE 8000
# Run main-server.py when the container launches
CMD ["python", "main-server.py"]
