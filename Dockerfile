# Stage 1: Build stage
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5 as builder

# Set BASE_DIR for your application
ENV BASE_DIR=/opt/fishprediction/

# Install necessary tools in the builder stage
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget gnupg \
    && apt-get clean

# Install Python dependencies
RUN pip3 install --no-cache Flask pandas numpy geopandas rasterio tensorflow keras folium

# Create and switch to the application directory
WORKDIR $BASE_DIR

# Copy the application source code (non-mapped directories)
COPY app.py $BASE_DIR
COPY ml_bio_mean $BASE_DIR/ml_bio_mean
COPY results $BASE_DIR/results
COPY templates $BASE_DIR/templates

# Stage 2: Runtime stage (minimal image)
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5 as final

# Set BASE_DIR in the final image as well
ENV BASE_DIR=/opt/fishprediction/

# Copy necessary files from the builder stage to the final stage
WORKDIR $BASE_DIR
COPY --from=builder $BASE_DIR $BASE_DIR
COPY --from=builder /usr/local/lib/python3.*/site-packages /usr/local/lib/python3.*/site-packages

# Ensure Python and Flask are installed in the runtime stage
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean

# Install Flask in the runtime stage to ensure it works
RUN pip3 install --no-cache Flask

# Set environment variables for Flask
ENV FLASK_APP=$BASE_DIR/app.py

# Expose Flask app port
EXPOSE 5000

# Run the Flask app using python -m flask to ensure it uses the correct Python environment
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
