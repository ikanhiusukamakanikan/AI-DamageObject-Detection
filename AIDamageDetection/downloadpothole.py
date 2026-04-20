import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/pothole-detection")

print("Path to dataset files:", path)