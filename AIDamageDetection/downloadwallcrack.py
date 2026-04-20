import kagglehub

# Download latest version
path = kagglehub.dataset_download("yatata1/crack-dataset")

print("Path to dataset files:", path)