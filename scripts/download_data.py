import gdown
import os

output_dir = os.path.join('..')
os.makedirs(output_dir, exist_ok=True)

# The folder URL from Google Drive
folder_url = 'https://drive.google.com/drive/folders/1IX-OQY1NuDNeU4vJ-hwdcs1klzi_1aff'

# Download the folder; gdown will download all files into the specified directory.
gdown.download_folder(url=folder_url, output=output_dir, quiet=False, use_cookies=False)