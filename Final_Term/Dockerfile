# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Make port 9696 available to the world outside this container
EXPOSE 9696

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "./predict.py"]
