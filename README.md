# 42 Amman Face Recognition System

A comprehensive guide to the face recognition system built with FastAI and ResNet34.

---

## Project Overview

This documentation explains the 42 Amman Face Recognition System, a deep learning-based solution that can identify whether an uploaded photo matches a student from 42 Amman based on their profile picture. The system uses state-of-the-art face detection, feature extraction, and similarity matching techniques.

### Workflow

1. **Data Collection and Preparation**: Process profile pictures to extract and align faces
2. **Dataset Creation**: Generate pairs of faces (same/different students) for training
3. **Model Training**: Train a ResNet34 model to distinguish between same and different faces
4. **Feature Extraction**: Extract face embeddings from the trained model
5. **Recognition System**: Compare new faces against known profiles using cosine similarity

---

## System Requirements

The face recognition system requires the following dependencies:

- **Core Libraries**: Python 3.8+, NumPy, Pandas, PIL (Pillow), Matplotlib
- **Deep Learning**: FastAI 2.8.1, PyTorch, face_recognition, dlib
- **Data Augmentation**: Albumentations
- **Interface (Optional)**: Gradio (for interactive UI)

```bash
pip install fastai==2.7.12 face-recognition dlib albumentations numpy pandas pillow matplotlib gradio
```

---

## Step 1: Data Preparation and Face Detection

The first step in building the face recognition system is to prepare the dataset by detecting and cropping faces from profile pictures. This ensures that the model focuses on facial features rather than backgrounds or other irrelevant details.

### Key Processing Steps

1. **Face Detection**: Using the face_recognition library with HOG-based detection to locate faces in images
2. **Face Cropping**: Extracting the face region with a 20% margin for context
3. **Standardization**: Resizing all faces to 224×224 pixels (standard input size for ResNet34)

> **Note:** The face detection process filters out images where no face is detected, ensuring that only valid face images are used for training.

#### Example Code for Face Detection

```python
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
```

---

## Step 2: Creating a Siamese Network Dataset

For face recognition when we have only one image per person, a Siamese network approach is ideal. This requires creating pairs of images labeled as either "same" (same student with different augmentations) or "different" (different students).

### Creating Training Pairs

- **Positive Pairs (Same Class):**
  - For each processed face image:
    1. Apply multiple random augmentations using Albumentations
    2. Each original image + augmented version creates a "same" pair
    3. Transformations include rotation, flipping, brightness/contrast changes, and more

- **Negative Pairs (Different Class):**
  - For each negative example:
    1. Randomly select two different student images
    2. Resize both to 224×224 pixels
    3. Label the pair as "different"

> **Note:** The system creates a balanced dataset with approximately equal numbers of positive and negative pairs to prevent bias in training.

#### Data Augmentation Details

```python
# Define the augmentations using Albumentations
aug_albumentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.2), # Added some noise augmentation
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.2), # Added dropout
    A.Resize(224, 224) # Albumentations applies resize as a transform
])
```

---

## Step 3: Model Architecture and Training

The face recognition system uses a ResNet34 model pretrained on ImageNet as the backbone. This approach leverages transfer learning to achieve good results even with limited training data.

### Model Architecture

- **During Training:** As a classifier that learns to distinguish between "same" and "different" face pairs
- **For Inference:** As a feature extractor that generates face embeddings (feature vectors)

### Training Process

- **Initial Training:**
  - Data split: 80% training, 20% validation
  - Learning rate: 1e-3
  - 5 epochs with frozen weights (transfer learning)
  - Metrics: error_rate and accuracy
- **Fine-tuning:**
  - Unfreeze all layers
  - Learning rate: slice(1e-5, 1e-4) for discriminative learning rates
  - 5 additional epochs
  - One-cycle learning policy

#### FastAI DataBlock API

```python
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=[Resize(224)],  # Ensure consistent size
    batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
)

dls = dblock.dataloaders(siamese_data_path, bs=32)
```

#### Model Training Code

```python
# Create and train the model
learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy])

# Find optimal learning rate
learn.lr_find()

# Train the model
learn.fit_one_cycle(5, 1e-3)

# Fine-tune the model by unfreezing
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5, 1e-4))
```

---

## Step 4: Feature Extraction and Embeddings

After training, the model is used to extract face embeddings - high-dimensional vector representations of faces that capture distinctive features. These embeddings are then used for face comparison and recognition.

### Feature Extraction Process

1. **Input Preprocessing**: Load and transform face image to model's required format
2. **Forward Pass**: Pass the image through the ResNet34 model, excluding the final classification layer
3. **Feature Extraction**: Capture the output of the penultimate layer (typically 512-dimensional vector)
4. **Storage**: Save the feature vectors for all known faces for later comparison

#### Feature Extraction Code

```python
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
        # Handle different batch data structures
        if isinstance(batch_data, tuple):
            x = batch_data[0]
        else:
            x = batch_data
        # Move tensor to device
        x = x.to(device)
        # Get the embeddings/activations from the feature extractor
        activations = feature_extractor(x)
    # Convert to numpy array
    return activations[0].cpu().numpy()
```

> **Important:** The feature vectors are normalized before comparison to ensure that similarities are based on the angle between vectors (cosine similarity) rather than their magnitudes.

#### Precomputing Feature Vectors

```python
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
```

---

## Step 5: Face Recognition and Matching

The face recognition process compares a query face with all known faces to find potential matches. This is done by calculating the cosine similarity between feature vectors.

### Recognition Process

1. **Query Image Processing**: Detect, crop, and align the face in the query image
2. **Feature Extraction**: Extract the feature vector from the query face
3. **Similarity Calculation**: Calculate cosine similarity with all known faces
4. **Threshold Filtering**: Filter matches based on a similarity threshold (e.g., 0.7)
5. **Results Ranking**: Rank potential matches by similarity score

#### Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, providing a similarity score between -1 (completely different) and 1 (identical). In practice, face embedding similarities typically range from 0 to 1.

#### Recognition Function

```python
def recognize_face(img_path, threshold=0.7):
    """Recognize a face in the given image"""
    # Process the query image to extract face
    query_face_path = Path('./query_face.jpg')
    face_found = process_image(img_path, query_face_path)
    if not face_found:
        return "No face detected in the query image", None
    # Extract features from the query face
    query_features = extract_features(learn, query_face_path)
    # Calculate cosine similarity with all known faces
    similarities = {}
    for name, features in feature_vectors.items():
        similarity = np.dot(query_features, features) / (
            np.linalg.norm(query_features) * np.linalg.norm(features))
        similarities[name] = float(similarity)
    # Sort by similarity score (highest first)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # Filter by threshold
    matches = [match for match in sorted_similarities if match[1] >= threshold]
    if not matches:
        return "No matches found above the similarity threshold", sorted_similarities[:5]
    return matches, query_face_path
```

> **Similarity Threshold:**
> - Higher threshold (e.g., 0.8+): More precision, fewer false positives, but might miss some true matches
> - Lower threshold (e.g., 0.6-0.7): Higher recall, more potential matches, but might include some false positives

---

## Interactive Interface

The face recognition system can be used through an interactive interface built with Gradio. This allows users to:

- **Face Recognition**: Upload a photo and find matching students from the database
- **Face Verification**: Compare two faces to check if they belong to the same person

#### Interface Setup

```python
# Create the Gradio interface for face recognition
iface = gr.Interface(
    fn=recognize_face_gradio,
    inputs=[
        gr.Image(type="numpy", label="Upload a face image"),
        gr.Slider(minimum=0.5, maximum=0.95, value=0.7, step=0.05, label="Similarity Threshold")
    ],
    outputs=gr.Textbox(label="Recognition Results"),
    title="42 Amman Face Recognition",
    description="Upload a photo to check if it matches a student from 42 Amman."
)

# Launch the interface
iface.launch(share=True)
```

---

## Performance Optimization and Deployment

The face recognition system can be optimized for better performance and deployed in various environments.

### Model Export

The trained model can be exported for deployment in production environments:

```python
# Export model
learn.export('./42amman_face_model.pkl')

# Export feature vectors as a compressed file
np.savez_compressed('42amman_face_features.npz', feature_vectors=feature_vectors)
```

### Performance Considerations

- **Speed Optimization**
  - Use CPU or GPU inference based on available hardware
  - Batch processing for multiple faces when possible
  - Precompute and store feature vectors for known faces
- **Accuracy Optimization**
  - Adjust face detection parameters for different scenarios
  - Fine-tune the similarity threshold based on use case
  - Regular model retraining with new images for improved performance

> **Best Practices:**
> - Use clear, well-lit photos for best recognition results
> - Ensure the face is clearly visible and not obscured
> - For profile updates, periodically recompute feature vectors

---

## Conclusion and Future Improvements

The 42 Amman Face Recognition System demonstrates how modern deep learning techniques can be applied to create an effective face recognition solution. The system uses a combination of face detection, feature extraction, and similarity matching to identify faces with high accuracy.

### Key Achievements

- Reliable face detection and alignment
- High-quality feature extraction using ResNet34
- Effective similarity-based matching
- User-friendly interface for face recognition and verification

### Potential Improvements

- **Technical Enhancements**
  - Implement more advanced face alignment techniques
  - Explore newer model architectures (e.g., EfficientNet, Vision Transformer)
  - Use triplet loss or contrastive loss for direct embedding learning
- **Feature Additions**
  - Support for video-based face recognition
  - Multi-face detection and recognition in group photos
  - Integration with access control systems

---

© 2025 42-Segfault Face Recognition Project | Created with FastAI and PyTorch
