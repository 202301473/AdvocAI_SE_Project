# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
# libpq-dev is for PostgreSQL, others might be needed for PDF generation (xhtml2pdf/reportlab)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirement.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

# Copy project
COPY . /app/

# Run the application (defaulting to daphne for ASGI support since channels is used)
# CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "legal_doc_generator.asgi:application"]
