# Use the official image as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean

# Copy the requirements file into the container
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install the dependencies using --use-pep517
RUN pip install --use-pep517 --no-cache-dir -r requirements.txt
RUN pip install --use-pep517 --no-cache-dir -r requirements_dev.txt

# Copy the rest of the application code
COPY . .

# Install the Selector package using PEP 517 standards-based tools
RUN pip install --use-pep517 .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.enableXsrfProtection=false"]
