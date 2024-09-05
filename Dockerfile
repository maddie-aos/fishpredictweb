# Stage 1: Build stage
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5 AS builder

# Set BASE_DIR for your application
ENV BASE_DIR=/opt/fishprediction
ENV FLASK_ENV=developmen


# Install necessary tools in the builder stage
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget gnupg \
    && apt-get clean

# Install Python dependencies in the builder stage
RUN pip3 install --no-cache Flask pandas numpy geopandas rasterio tensorflow keras folium

# Copy the application source code (non-mapped directories)
WORKDIR $BASE_DIR
COPY app.py $BASE_DIR
COPY ml_bio_mean $BASE_DIR/ml_bio_mean
COPY results $BASE_DIR/results
COPY templates $BASE_DIR/templates

# Stage 2: Runtime stage
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5 AS final

# Set BASE_DIR in the final image
ENV BASE_DIR=/opt/fishprediction

# Copy the application files from the builder stage
WORKDIR $BASE_DIR
COPY app.py $BASE_DIR
COPY ml_bio_mean $BASE_DIR/ml_bio_mean
COPY results $BASE_DIR/results
COPY templates $BASE_DIR/templates

# Install Python in the final stage
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean

# Install the necessary Python dependencies in the final stage
RUN pip3 install --no-cache Flask pandas numpy geopandas rasterio tensorflow keras folium gunicorn

# Set environment variables for Flask
ENV FLASK_APP=$BASE_DIR/app.py
ENV FLASK_ENV=production

# Expose Flask app port
EXPOSE 5000

# Run the Flask app using python -m flask to ensure it uses the correct Python environment
#CMD ["flask", "run", "--host=0.0.0.0"]
#CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
