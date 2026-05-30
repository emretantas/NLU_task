"""
Download assisting binary files, and save to binary file path.
"""

import requests
import zipfile
from pathlib import Path


def download_and_extract_glove_6B(download_folder):
    """
    Download and extract GloVe 6B embeddings.

    Args:
        download_folder: Path to folder where GloVe will be downloaded

    Returns:
        Path to download folder

    Example:
        >>> download_and_extract_glove_6B('bin/glove')
    """
    URL = "http://nlp.stanford.edu/data/glove.6B.zip"

    download_folder = Path(download_folder)
    download_folder.mkdir(parents=True, exist_ok=True)

    zip_path = download_folder / "glove.6B.zip"

    # Check if zip file already exists
    if zip_path.exists():
        print(f"Zip file already exists at {zip_path}")
        print("Skipping download.")
    else:
        print(f"Downloading GloVe embeddings from {URL}")
        print("This may take a few minutes...")

        # Download the file
        response = requests.get(URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)

        print(f"\nDownload complete: {zip_path}")

    # Extract the zip file
    print(f"Extracting files to {download_folder}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_folder)

    print(f"Extraction complete. GloVe files available in {download_folder}")
    return download_folder

def elmo_downloader(download_folder):
    """
    Download ELMo model files (options.json and weights.hdf5).

    Args:
        download_folder: Path to folder where ELMo files will be downloaded

    Returns:
        Path to download folder

    Example:
        >>> elmo_downloader('bin/elmo')
    """
    ELMO_JSON_URL = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    ELMO_HDF5_URL = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    download_folder = Path(download_folder)
    download_folder.mkdir(parents=True, exist_ok=True)

    json_path = download_folder / "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    hdf5_path = download_folder / "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # Download JSON options file
    if json_path.exists():
        print(f"ELMo options file already exists at {json_path}")
        print("Skipping download.")
    else:
        print(f"Downloading ELMo options file from {ELMO_JSON_URL}")
        response = requests.get(ELMO_JSON_URL, stream=True)
        response.raise_for_status()

        with open(json_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Download complete: {json_path}")

    # Download HDF5 weights file
    if hdf5_path.exists():
        print(f"ELMo weights file already exists at {hdf5_path}")
        print("Skipping download.")
    else:
        print(f"Downloading ELMo weights file from {ELMO_HDF5_URL}")
        print("This is a large file (~600MB), it may take several minutes...")

        response = requests.get(ELMO_HDF5_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(hdf5_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)

        print(f"\nDownload complete: {hdf5_path}")

    print(f"ELMo model files available in {download_folder}")
    return download_folder