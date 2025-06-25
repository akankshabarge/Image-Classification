import tensorflow as tf
import os
import shutil
import zipfile

def download_cats_dogs_dataset():
    print("Downloading cats and dogs dataset...")
    
    # Download the dataset
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    
    # Find the extracted folder
    base_dir = os.path.dirname(path_to_zip)
    print(f"Base directory: {base_dir}")
    
    # List contents to see the actual structure
    print("Contents of base directory:")
    for item in os.listdir(base_dir):
        print(f"  {item}")
        if os.path.isdir(os.path.join(base_dir, item)):
            sub_path = os.path.join(base_dir, item)
            print(f"    Contents of {item}:")
            for sub_item in os.listdir(sub_path):
                print(f"      {sub_item}")
    
    # Try different possible paths
    possible_paths = [
        os.path.join(base_dir, 'cats_and_dogs_filtered'),
        os.path.join(base_dir, 'cats_and_dogs'),
        base_dir
    ]
    
    extracted_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'train')):
            extracted_path = path
            break
    
    if not extracted_path:
        # Manual extraction
        print("Manually extracting zip file...")
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # Check again
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'train')):
                extracted_path = item_path
                break
    
    if not extracted_path:
        print("Could not find extracted dataset. Contents of base directory:")
        for item in os.listdir(base_dir):
            print(f"  {item}")
        return
    
    print(f"Found dataset at: {extracted_path}")
    
    # Set up source paths
    source_train = os.path.join(extracted_path, 'train')
    source_validation = os.path.join(extracted_path, 'validation')
    
    print(f"Source train: {source_train}")
    print(f"Source validation: {source_validation}")
    
    # Create our data directories
    os.makedirs('data', exist_ok=True)
    
    # Copy training data
    if os.path.exists(source_train):
        print("Copying training data...")
        if os.path.exists('data/train'):
            shutil.rmtree('data/train')
        shutil.copytree(source_train, 'data/train')
        print("✅ Training data copied")
    else:
        print(f"❌ Training data not found at {source_train}")
    
    # Copy validation data
    if os.path.exists(source_validation):
        print("Copying validation data...")
        if os.path.exists('data/validation'):
            shutil.rmtree('data/validation')
        shutil.copytree(source_validation, 'data/validation')
        print("✅ Validation data copied")
        
        # Create test data from validation (copy validation to test)
        print("Creating test data...")
        if os.path.exists('data/test'):
            shutil.rmtree('data/test')
        shutil.copytree('data/validation', 'data/test')
        print("✅ Test data created")
    else:
        print(f"❌ Validation data not found at {source_validation}")
    
    # Count images
    if os.path.exists('data/train'):
        train_cats = len([f for f in os.listdir('data/train/cats') if f.endswith(('.jpg', '.jpeg', '.png'))])
        train_dogs = len([f for f in os.listdir('data/train/dogs') if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\n📊 Dataset Summary:")
        print(f"   Training cats: {train_cats}")
        print(f"   Training dogs: {train_dogs}")
        
        if os.path.exists('data/validation'):
            val_cats = len([f for f in os.listdir('data/validation/cats') if f.endswith(('.jpg', '.jpeg', '.png'))])
            val_dogs = len([f for f in os.listdir('data/validation/dogs') if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   Validation cats: {val_cats}")
            print(f"   Validation dogs: {val_dogs}")
    
    print("\n✅ Dataset ready!")
    print("Next steps:")
    print("1. python train_models.py")
    print("2. cd ui && python app.py")

if __name__ == "__main__":
    download_cats_dogs_dataset()