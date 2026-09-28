FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files to disk and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_KEY="prod-secret-key-12345"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train and build the model pipeline binary into model/loan_model.joblib
RUN python train_model.py

# Expose container application port
EXPOSE 5000

# Run WSGI production server Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "app:app"]