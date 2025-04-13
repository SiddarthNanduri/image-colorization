import os
import sys
import urllib.request
import zipfile
import tarfile

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete!")

def extract_file(filename, extract_path):
    print(f"Extracting {filename}...")
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif filename.endswith('.tar.gz'):
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    print("Extraction complete!")

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # COCO 2017 dataset URLs
    urls = {
        'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017': 'http://images.cocodataset.org/zips/val2017.zip'
    }
    
    # Download and extract files
    for name, url in urls.items():
        filename = os.path.join('data', f'{name}.zip')
        download_file(url, filename)
        extract_file(filename, 'data')
        # Remove the zip file after extraction
        os.remove(filename)
    
    print("COCO dataset download and extraction complete!")

if __name__ == '__main__':
    main() 