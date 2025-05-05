
FROM python:3.11

# Set the working directory
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY requirements.txt /app

RUN python -m pip install --no-cache-dir --upgrade pip==20.0.2
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

COPY mistral.py /app

# Run the application
CMD ["streamlit", "run", "mistral.py", "--server.address=0.0.0.0", "--server.port=8000"]