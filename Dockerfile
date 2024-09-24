# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch, Sentence Transformers, and required dependencies in one go
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy scikit-learn huggingface-hub tqdm sentence-transformers "transformers<5.0.0"

# Copy the 'app' directory and everything inside it into the container
COPY app/ /app/

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable to specify the entry point
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]