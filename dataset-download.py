import os
from roboflow import download_dataset

try:
    print("Attempting to download dataset...")
    
    # Using the correct URL format from the working Google Colab example
    dataset = download_dataset(
        "https://universe.roboflow.com/rf100-vl/mahjong-vtacs-mexax-m4vyu-sjtd/dataset/2",
        "coco"
    )
    
    print("Dataset downloaded successfully!")
    print(f"Dataset location: {os.path.abspath('mahjong-vtacs-mexax-m4vyu-sjtd-2')}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nPlease make sure you have:")
    print("1. Have a stable internet connection")
    print("2. Have enough disk space")
    print("3. Have write permissions in the current directory")
