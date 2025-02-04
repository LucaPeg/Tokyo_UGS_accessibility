import requests

zenodo_url = "https://zenodo.org/record/YOUR_RECORD_ID/files/dataset.zip?download=1"

output_file = "tokyo_ugs_accessibility.zip"

# Download the file
response = requests.get(zenodo_url, stream=True)
with open(output_file, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)

print("Download complete.")