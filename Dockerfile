# Use the official image as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && \
    apt-get install -y build-essential git && \
    apt-get clean

# Upgrade pip
RUN pip install --upgrade pip

# Install streamlit
RUN pip install streamlit

# Install the Selector package from the source code
RUN pip install git+https://github.com/theochem/Selector.git

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.enableXsrfProtection=false"]
