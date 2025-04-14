import os
import requests
from PIL import Image
from io import BytesIO
import json

def download_test_images(num_images=5, output_dir='test_images'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO validation images URLs (these are public domain images)
    image_urls = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',  # A person riding a horse
        'http://images.cocodataset.org/val2017/000000037777.jpg',  # A dining table with food
        'http://images.cocodataset.org/val2017/000000252219.jpg',  # A city street scene
        'http://images.cocodataset.org/val2017/000000087038.jpg',  # A beach scene
        'http://images.cocodataset.org/val2017/000000174482.jpg'   # A group of people
    ]
    
    print(f"Downloading {num_images} test images to {output_dir}...")
    
    for i, url in enumerate(image_urls[:num_images]):
        try:
            # Download image
            response = requests.get(url)
            response.raise_for_status()
            
            # Open and save image
            img = Image.open(BytesIO(response.content))
            img_path = os.path.join(output_dir, f'test_image_{i+1}.jpg')
            img.save(img_path)
            
            print(f"Downloaded: {img_path}")
            
        except Exception as e:
            print(f"Error downloading image {i+1}: {str(e)}")
    
    print("Download complete!")

if __name__ == '__main__':
    download_test_images() 