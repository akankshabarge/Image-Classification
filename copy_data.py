import os
import shutil

def copy_existing_data():
    # The data is already extracted here
    source_base = r"C:\Users\Akanksha\.keras\datasets\cats_and_dogs_extracted\cats_and_dogs_filtered"
    
    source_train = os.path.join(source_base, "train")
    source_validation = os.path.join(source_base, "validation")
    
    print("Copying data from existing download...")
    
    # Create our data directory
    os.makedirs('data', exist_ok=True)
    
    # Copy training data
    if os.path.exists(source_train):
        print("Copying training data...")
        if os.path.exists('data/train'):
            shutil.rmtree('data/train')
        shutil.copytree(source_train, 'data/train')
        print("✅ Training data copied")
    
    # Copy validation data  
    if os.path.exists(source_validation):
        print("Copying validation data...")
        if os.path.exists('data/validation'):
            shutil.rmtree('data/validation')
        shutil.copytree(source_validation, 'data/validation')
        print("✅ Validation data copied")
        
        # Create test data (copy from validation)
        print("Creating test data...")
        if os.path.exists('data/test'):
            shutil.rmtree('data/test')
        shutil.copytree('data/validation', 'data/test')
        print("✅ Test data created")
    
    # Count images
    try:
        train_cats = len([f for f in os.listdir('data/train/cats') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_dogs = len([f for f in os.listdir('data/train/dogs') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Training cats: {train_cats}")
        print(f"   Training dogs: {train_dogs}")
        
        val_cats = len([f for f in os.listdir('data/validation/cats') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_dogs = len([f for f in os.listdir('data/validation/dogs') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   Validation cats: {val_cats}")
        print(f"   Validation dogs: {val_dogs}")
        
        test_cats = len([f for f in os.listdir('data/test/cats') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        test_dogs = len([f for f in os.listdir('data/test/dogs') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   Test cats: {test_cats}")
        print(f"   Test dogs: {test_dogs}")
        
    except Exception as e:
        print(f"Error counting images: {e}")
    
    print("\n✅ Dataset ready!")
    print("Next steps:")
    print("1. python train_models.py")
    print("2. cd ui && python app.py")

if __name__ == "__main__":
    copy_existing_data()