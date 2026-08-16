import urllib.request
import os

def download_file(url, dest_path):
    """
    Downloads a single file from a URL to a local destination path.
    """
    # Get the file name from the path for cleaner print logs
    filename = os.path.basename(dest_path)
    print(f"Downloading {filename} from {url}...")
    
    try:
        # urllib.request.urlretrieve downloads a network resource directly to a local file
        urllib.request.urlretrieve(url, dest_path)
        # Check size of downloaded file to confirm it downloaded content
        file_size = os.path.getsize(dest_path) / (1024 * 1024) # Convert bytes to Megabytes
        print(f"Successfully saved {filename} ({file_size:.2f} MB)")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def main():
    # Ensure the target dataset directory exists
    os.makedirs("dataset", exist_ok=True)
    
    # Define the raw GitHub URLs containing the original ISOT CSV files
    true_url = "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/True.csv"
    fake_url = "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/Fake.csv"
    
    # Define local file destination paths
    true_dest = os.path.join("dataset", "True.csv")
    fake_dest = os.path.join("dataset", "Fake.csv")
    
    print("=======================================================")
    print("ISOT DATASET DOWNLOAD HELPER")
    print("=======================================================")
    print("Downloading the original real/fake CSV files (approx. 110MB total).")
    print("This might take a minute depending on your internet connection...")
    print("-" * 55)
    
    # Run download for True news (approx 45MB)
    download_file(true_url, true_dest)
    
    # Run download for Fake news (approx 62MB)
    download_file(fake_url, fake_dest)
    
    print("-" * 55)
    print("Download process completed!")
    print("You can now run 'python train_model.py' to train your models on real data.")
    print("=======================================================")

if __name__ == "__main__":
    main()
