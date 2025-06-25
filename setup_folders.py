import os

def create_folders():
    folders = [
        'data/train/cats',
        'data/train/dogs',
        'data/validation/cats', 
        'data/validation/dogs',
        'data/test/cats',
        'data/test/dogs',
        'models',
        'utils',
        'ui',
        'saved_models'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")
    
    # Create __init__.py files
    init_files = ['models/__init__.py', 'utils/__init__.py']
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('')
        print(f"Created: {init_file}")
    
    print("All folders created successfully!")

if __name__ == "__main__":
    create_folders()