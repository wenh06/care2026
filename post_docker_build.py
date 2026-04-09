"""
Post-docker-build script: downloads and caches our own trained model
weights (uploaded to cloud storage) into the Docker image at build time,
so they are available at inference without being included as large files
in the repository.
"""
