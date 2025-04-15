import os
import requests
from PIL import Image
from io import BytesIO
import json

def download_test_images(num_images=5, output_dir='test_images'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO validation images URLs 
    image_urls = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',  
        'http://images.cocodataset.org/val2017/000000037777.jpg',  
        'http://images.cocodataset.org/val2017/000000252219.jpg',  
        'http://images.cocodataset.org/val2017/000000087038.jpg',  
        'http://images.cocodataset.org/val2017/000000174482.jpg'   
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
