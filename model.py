import os
import numpy as np
import pandas as pd
import face_recognition
from PIL import Image, ImageDraw
from fastai.vision.all import *
from fastai.vision.widgets import *
from pathlib import Path
import matplotlib.pyplot as plt
import zipfile
import random
import time


DATA_PATH = Path('./42amman-profiles')
print(f"Number of profile images: {len(list(DATA_PATH.glob('*.jpg')))}")

# Create processed data directory
PROCESSED_PATH = Path('./processed_faces')
PROCESSED_PATH.mkdir(exist_ok=True)

def process_image(img_path, output_path):
    """Detect face in image, crop, and save to output path"""
    try:
        # Load image using face_recognition
        image = face_recognition.load_image_file(img_path)
        
        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image, model="hog")
        
        if not face_locations:
            print(f"No face found in {img_path}")
            return False
        
        # For simplicity, we'll use the first face found
        top, right, bottom, left = face_locations[0]
        
        # Add some margin to the face crop (20% of face size)
        height, width = bottom - top, right - left
        margin_h, margin_w = int(height * 0.2), int(width * 0.2)
        
        # Adjust boundaries with margins and ensure they're within image bounds
        img_h, img_w = image.shape[:2]
        top = max(0, top - margin_h)
        bottom = min(img_h, bottom + margin_h)
        left = max(0, left - margin_w)
        right = min(img_w, right + margin_w)
        
        # Crop the image to focus on the face
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        
        # Resize to a standard size (224x224 for resnet34)
        pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Save the processed image
        pil_image.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False
    
# Process all images in the dataset
def process_dataset():
    successful = 0
    failed = 0
    
    # Get all jpg files
    all_images = list(DATA_PATH.glob('*.jpg'))
    total = len(all_images)
    
    print(f"Processing {total} profile images...")
    
    for i, img_path in enumerate(all_images):
        if i % 50 == 0:
            print(f"Progress: {i}/{total}")
            
        output_path = PROCESSED_PATH / img_path.name
        if process_image(img_path, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"Finished processing {successful} images successfully, {failed} failed")
    return successful, failed

# Run the processing
successful, failed = process_dataset()


# Display a random selection of processed faces
processed_images = list(PROCESSED_PATH.glob('*.jpg'))
sample_size = min(8, len(processed_images))
samples = random.sample(processed_images, sample_size)

# Create a grid to display the sample images
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

for i, img_path in enumerate(samples):
    if i < len(axes):
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(img_path.stem)
        axes[i].axis('off')

plt.tight_layout()
plt.show()


# Function to create training dataset with image pairs
def create_siamese_dataset():
    # Create directories for the dataset
    siamese_data_path = Path('../working/siamese_data')
    siamese_data_path.mkdir(exist_ok=True)
    
    # Create 'same' and 'different' directories
    same_dir = siamese_data_path/'same'
    diff_dir = siamese_data_path/'different'
    same_dir.mkdir(exist_ok=True)
    diff_dir.mkdir(exist_ok=True)
    
    # Get all processed images
    processed_images = list(PROCESSED_PATH.glob('*.jpg'))
    
    # Create positive pairs (same student, different augmentations)
    print("Creating positive pairs...")
    for i, img_path in enumerate(processed_images):
        # For each image, create 5 augmented versions
        img = PILImage.create(img_path)
        
        for j in range(5):
            # Apply random augmentations
            aug_img = img.apply_tfms(
                [*aug_transforms(do_flip=True, flip_vert=False, max_rotate=15.0, 
                                 max_zoom=1.1, max_lighting=0.2, max_warp=0.2)], 
                size=224)
            
            # Save original and augmented image as pair
            pair_name = f"{img_path.stem}_pair_{j}.jpg"
            aug_img.save(same_dir/pair_name)
            
        if i % 50 == 0:
            print(f"Created positive pairs for {i}/{len(processed_images)} images")
    
    # Create negative pairs (different students)
    print("\nCreating negative pairs...")
    num_neg_pairs = len(processed_images) * 5  # Same number as positive pairs
    
    for i in range(num_neg_pairs):
        # Select two different random images
        img1, img2 = random.sample(processed_images, 2)
        
        # Create the pair name
        pair_name = f"{img1.stem}_vs_{img2.stem}_{i}.jpg"
        
        # Load and resize both images
        img1_pil = PILImage.create(img1).resize((224, 224))
        img2_pil = PILImage.create(img2).resize((224, 224))
        
        # Save the pair
        img1_pil.save(diff_dir/pair_name)
        
        if i % 100 == 0:
            print(f"Created {i}/{num_neg_pairs} negative pairs")
    
    # Verify dataset creation
    same_count = len(list(same_dir.glob('*.jpg')))
    diff_count = len(list(diff_dir.glob('*.jpg')))
    
    print(f"\nDataset creation complete!")
    print(f"Same class (positive) pairs: {same_count}")
    print(f"Different class (negative) pairs: {diff_count}")
    
    return siamese_data_path, same_count, diff_count

# Create the Siamese dataset
siamese_data_path, same_count, diff_count = create_siamese_dataset()



import os
import numpy as np
import pandas as pd
import face_recognition
from PIL import Image, ImageDraw
# We will still import fastai, but won't rely on its apply_tfms here for augmentation
from fastai.vision.all import * # Keep fastai for other potential uses and PILImage class
from fastai.vision.widgets import *
from pathlib import Path
import matplotlib.pyplot as plt
import zipfile
import random
import time
import albumentations as A # Import albumentations

# Assuming other necessary imports like requests, json, concurrent.futures, tqdm
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Rest of your initial code (download functions, process_image, process_dataset)
# ... (paste download_image, download_all_images, process_image, process_dataset here)
# Make sure PROCESSED_PATH and DATA_PATH are defined as in your notebook


# Create processed data directory (if not already done)
PROCESSED_PATH = Path('./processed_faces')
PROCESSED_PATH.mkdir(exist_ok=True)

# Assuming DATA_PATH is defined elsewhere or from the download step
# Example placeholder if not defined:
# DATA_PATH = Path(DEFAULT_OUTPUT_DIR)


# Re-define process_image and process_dataset if they were in the original code block
# ... (paste your process_image and process_dataset functions here)


# --- Start of Modified create_siamese_dataset ---
def create_siamese_dataset():
    """
    Function to create Siamese training dataset with image pairs
    using albumentations for augmentation.
    """
    # Create directories for the dataset
    siamese_data_path = Path('./siamese_data')
    siamese_data_path.mkdir(exist_ok=True)

    # Create 'same' and 'different' directories
    same_dir = siamese_data_path/'same'
    diff_dir = siamese_data_path/'different'
    same_dir.mkdir(exist_ok=True)
    diff_dir.mkdir(exist_ok=True)

    # Get all processed images
    processed_images = list(PROCESSED_PATH.glob('*.jpg'))

    if not processed_images:
        print("No processed images found. Run process_dataset first.")
        return siamese_data_path, 0, 0

    # Define the augmentations using Albumentations
    aug_albumentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2), # Added some noise augmentation
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.2), # Added dropout
        A.Resize(224, 224) # Albumentations applies resize as a transform
    ])

    # Create positive pairs (same student, different augmentations)
    print("Creating positive pairs using Albumentations...")
    positive_samples_created = 0
    positive_pairs_failed = 0
    num_positive_pairs_per_image = 3 # Create 3 augmented versions per image for more data

    for i, img_path in enumerate(processed_images):
        try:
            # Load image using PIL and convert to NumPy array (H, W, C), values 0-255
            img_pil_std = Image.open(img_path).convert('RGB')
            img_np = np.array(img_pil_std)

            for j in range(num_positive_pairs_per_image):
                 # Apply Albumentations transforms
                 augmented_np = aug_albumentations(image=img_np)['image']

                 # Convert augmented NumPy array back to PIL Image
                 augmented_pil = Image.fromarray(augmented_np.astype(np.uint8))

                 # Save the augmented image
                 pair_name = f"{img_path.stem}_aug_{j}.jpg" # Name indicating it's an augmentation
                 augmented_pil.save(same_dir/pair_name)

            positive_samples_created += num_positive_pairs_per_image

            if (i + 1) % 20 == 0: # Progress update more frequently
                print(f"Processed {i+1}/{len(processed_images)} images for positive pairs. Created {positive_samples_created} samples so far.")

        except Exception as e:
            print(f"Error creating positive pair for {img_path}: {e}")
            positive_pairs_failed += 1

    print(f"\nFinished creating positive pairs. {positive_samples_created} samples created, {positive_pairs_failed} source images failed.")


    # Create negative pairs (different students)
    print("\nCreating negative pairs...")
    # Let's create a similar number of negative pairs as positive pairs
    num_neg_pairs = positive_samples_created

    neg_successful = 0
    neg_failed = 0

    # Check if there are enough images to create negative pairs
    if len(processed_images) < 2:
        print("Need at least 2 images to create negative pairs. Skipping negative pair creation.")
        num_neg_pairs = 0

    # Use tqdm for a progress bar on negative pair creation
    for i in tqdm(range(num_neg_pairs), desc="Creating negative pairs"):
        try:
            # Select two different random images
            # Use a loop to ensure different images are selected
            img1_path = random.choice(processed_images)
            while True:
                img2_path = random.choice(processed_images)
                if img1_path != img2_path:
                    break

            # Create the pair name including both source image stems
            # The name format allows potentially reconstructing the pair source later
            pair_name = f"{img1_path.stem}_vs_{img2_path.stem}_{i}.jpg"

            # Load and resize both images using PIL to the target size
            img1_pil = Image.open(img1_path).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
            img2_pil = Image.open(img2_path).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)

            # Combine images side-by-side to create a single image representing the pair
            combined_width = img1_pil.width + img2_pil.width
            combined_height = max(img1_pil.height, img2_pil.height)
            combined_img = Image.new('RGB', (combined_width, combined_height))
            combined_img.paste(img1_pil, (0, 0))
            combined_img.paste(img2_pil, (img1_pil.width, 0))

            combined_img.save(diff_dir/pair_name)

            neg_successful += 1

        except Exception as e:
            # Print error but don't stop the loop
            print(f"\nError creating negative pair {i}: {e}")
            neg_failed += 1
            # It might be useful to know which files caused the error
            # print(f"Error with files: {img1_path}, {img2_path}")


    print(f"\nFinished creating negative pairs. {neg_successful} samples created, {neg_failed} pairs failed.")


    # Verify dataset creation
    same_count = len(list(same_dir.glob('*.jpg')))
    diff_count = len(list(diff_dir.glob('*.jpg')))

    print(f"\nDataset creation complete!")
    print(f"Files created:")
    print(f"  '{same_dir}': {same_count} files")
    print(f"  '{diff_dir}': {diff_count} files")

    return siamese_data_path, same_count, diff_count

# Create the Siamese dataset
# Ensure PROCESSED_PATH is populated by running process_dataset first if needed
# successful_processed, failed_processed = process_dataset() # Uncomment if needed

siamese_data_path, same_count, diff_count = create_siamese_dataset()



# Create a FastAI DataBlock for the Siamese network
class SiameseImage(fastuple):
    @classmethod
    def create(cls, fnames):
        return cls(PILImage.create(fnames))

# Set up the dataloader with explicit NumPy check
def get_data(bs=32):
    """Create FastAI dataloaders for the Siamese network with error handling."""
    # First explicitly check if numpy is available
    try:
        import numpy as np
        print("NumPy is available, version:", np.__version__)
    except ImportError:
        print("Error: NumPy is not available. Please install it with: pip install numpy")
        return None
    
    # Now proceed with dataloader creation
    try:
        # Define a transform that ensures all images are resized to the same dimensions
        # This addresses the error where images have inconsistent sizes
        item_tfms = [Resize(224)]
        
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=parent_label,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            item_tfms=item_tfms,  # Apply resize to each item individually
            batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
        )
        
        # Verify siamese_data_path exists
        if not siamese_data_path.exists():
            print(f"Warning: {siamese_data_path} does not exist. Run create_siamese_dataset() first.")
            return None
            
        print(f"Loading data from {siamese_data_path}...")
        return dblock.dataloaders(siamese_data_path, bs=bs)
    
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return None

# Try to create dataloaders with proper error handling
print("Creating dataloaders...")
dls = get_data()

if dls is not None:
    print(f"Successfully created dataloaders")
    # Safely show a batch with error handling
    try:
        print("Displaying sample batch:")
        dls.show_batch(max_n=8, figsize=(10, 10))
    except Exception as e:
        print(f"Error showing batch: {e}")
        
    # Create and train the model with proper error handling
    try:
        # Explicitly import cnn_learner if it's not already available
        try:
            from fastai.vision.all import cnn_learner
            print("Successfully imported cnn_learner")
        except ImportError:
            print("Attempting alternative import path...")
            from fastai.vision.learner import cnn_learner
            print("Imported cnn_learner from fastai.vision.learner")
            
        print("Creating model...")
        learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy])
        print("Model created successfully")
        
        # Find optimal learning rate
        print("Finding optimal learning rate...")
        learn.lr_find()
        
        # Train the model
        print("Training model for 5 epochs...")
        learn.fit_one_cycle(5, 1e-3)
        
        # Save the model
        print("Saving model...")
        learn.save('siamese_model')
        print("Model saved as 'siamese_model'")
        
    except Exception as e:
        print(f"Error in model creation or training: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Failed to create dataloaders. Please resolve the issues above.")
    # You might want to install numpy if it's missing
    # !pip install numpy


# Fine-tune the model by unfreezing
learn.unfreeze()
learn.lr_find()

learn.fit_one_cycle(5, slice(1e-5, 1e-4))

# Evaluate the model on the validation set
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,10))

# Display the most confused images
interp.plot_top_losses(9, figsize=(15,11))



# Create a feature extractor
def extract_features(learn, img_path):
    """Extract features from the penultimate layer of the model"""
    import torch
    
    # Load and transform the image
    img = PILImage.create(img_path)
    
    # Create a test batch with a single image
    batch = learn.dls.test_dl([img])
    
    # Determine the device and move data appropriately
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the model's feature extractor (everything except the final layer)
    feature_extractor = learn.model[:-1].to(device)
    
    # Ensure the model is in eval mode
    feature_extractor.eval()
    
    # Extract features with gradient calculation disabled for efficiency
    with torch.no_grad():
        # Get the first batch from the dataloader
        batch_data = first(batch)
        
        # Debug the batch data structure
        print(f"Batch data type: {type(batch_data)}")
        
        # Handle different batch data structures
        if isinstance(batch_data, tuple):
            # Always take the first element from the tuple, which should be the input tensor
            x = batch_data[0]
            print(f"Extracted tensor from tuple, shape: {x.shape}")
        else:
            # It's already a tensor
            x = batch_data
            print(f"Using tensor directly, shape: {x.shape}")
        
        # Move tensor to device
        x = x.to(device)
        
        # Get the embeddings/activations from the feature extractor
        activations = feature_extractor(x)
    
    # Convert to numpy array (first taking it back to CPU if needed)
    return activations[0].cpu().numpy()

# Test feature extraction on a few images with better error handling
try:
    test_imgs = random.sample(list(PROCESSED_PATH.glob('*.jpg')), 3)
    for img_path in test_imgs:
        try:
            print(f"\nProcessing image: {img_path.name}")
            features = extract_features(learn, img_path)
            print(f"Image: {img_path.name}, Feature vector shape: {features.shape}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
except NameError as e:
    print(f"Error: {e}. Make sure the model 'learn' is defined before running feature extraction.")


# Save the model
learn.export('./42amman_face_model.pkl')
print("Model saved to './42amman_face_model.pkl'")

# Precompute and save feature vectors for all processed faces
feature_vectors = {}
processed_images = list(PROCESSED_PATH.glob('*.jpg'))

print(f"Computing feature vectors for {len(processed_images)} images...")
for i, img_path in enumerate(processed_images):
    if i % 50 == 0:
        print(f"Progress: {i}/{len(processed_images)}")
        
    features = extract_features(learn, img_path)
    feature_vectors[img_path.stem] = features

# Save feature vectors
np.save('./42amman_face_features.npy', feature_vectors)
print("Feature vectors saved to './42amman_face_features.npy'")
