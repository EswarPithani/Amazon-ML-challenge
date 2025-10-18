# ML Challenge Project Documentation

## Overview
This project appears to be a machine learning challenge focused on price prediction using multimodal data (text and images). The project involves downloading images, generating embeddings using CLIP (Contrastive Language-Image Pre-training), concatenating embeddings, and training ensemble models for price prediction.

## Project Structure
```
MLChallange/
├── concat.py              # Concatenates embeddings for test data
├── concat2.py             # Concatenates embeddings for training data
├── download_driver.py     # Downloads images from URLs
├── dummy.py               # Generates embeddings for test data
├── dummy2.py              # Generates embeddings for training data
└── train_test.py          # Main training and prediction pipeline
```

## File Descriptions

### 1. `download_driver.py`
**Purpose**: Downloads images from URLs in parallel for the test dataset.

**Key Features**:
- Parallel image downloading using 64 workers
- Downloads up to 75,000 images from test dataset
- Saves results to `download_results_test_75000.csv`
- Handles failed downloads gracefully

**Configuration**:
- Input CSV: `D:/DA/68e8d1d70b66d_student_resource/student_resource/dataset/test.csv`
- Output directory: `images_test_all`
- Max images: 75,000

**Dependencies**:
- `pandas`
- `os`
- `utils` module (contains `download_all_images` function)

### 2. `dummy.py` and `dummy2.py`
**Purpose**: Generate text and image embeddings using CLIP model for test and training data respectively.

**Key Features**:
- Uses `sentence-transformers/clip-ViT-B-16` model
- Generates 512-dimensional embeddings for both text and images
- Handles missing images gracefully
- Creates compact ML datasets with embeddings

**Process**:
1. Loads CSV data with image paths
2. Generates text embeddings from `catalog_content`
3. Generates image embeddings from image files
4. Creates compact dataset with embeddings
5. Saves to `embed_test_16.csv` (dummy.py) or `embed_train_16.csv` (dummy2.py)

**Dependencies**:
- `sentence_transformers`
- `pandas`
- `numpy`
- `PIL` (Pillow)
- `tqdm`

### 3. `concat.py` and `concat2.py`
**Purpose**: Concatenate text and image embeddings into single 1024-dimensional vectors for test and training data respectively.

**Key Features**:
- Parses string embeddings to numpy arrays
- Concatenates 512-dim text + 512-dim image = 1024-dim vector
- Validates embedding dimensions before concatenation
- Handles malformed embeddings gracefully

**Process**:
1. Loads embedding CSV files
2. Parses string embeddings to arrays
3. Concatenates text and image embeddings
4. Saves results with concatenated embeddings

**Output**:
- `test_embed_16.csv` (concat.py)
- `train_embed_16.csv` (concat2.py)

### 4. `train_test.py`
**Purpose**: Main training and prediction pipeline using ensemble of neural networks.

**Key Features**:
- Custom SMAPE (Symmetric Mean Absolute Percentage Error) loss function
- Ensemble of 5 different MLP architectures
- Advanced training with early stopping and learning rate scheduling
- Robust prediction using median + mean combination

## Model Architecture

### SMAPE Loss Function
```python
class SMAPELoss(nn.Module):
    def forward(self, y_pred, y_true):
        abs_diff = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2 + self.eps
        return torch.mean(abs_diff / denominator)
```

### Improved MLP Model
- Multiple hidden layers with BatchNorm and LeakyReLU
- Dropout for regularization
- Kaiming weight initialization
- Configurable architecture and dropout rates

### Ensemble Architecture
The ensemble consists of 5 different MLP models:
1. `[2048, 1024, 512, 256, 128]` with dropouts `[0.4, 0.4, 0.3, 0.2, 0.1]`
2. `[1536, 768, 384, 192, 96]` with dropouts `[0.35, 0.35, 0.25, 0.15, 0.05]`
3. `[1024, 1024, 512, 256, 128, 64]` with dropouts `[0.3, 0.3, 0.25, 0.2, 0.15, 0.1]`
4. `[2560, 1280, 640, 320, 160]` with dropouts `[0.45, 0.4, 0.35, 0.25, 0.15]`
5. `[1024, 512, 256, 128, 64, 32]` with dropouts `[0.3, 0.25, 0.2, 0.15, 0.1, 0.05]`

## Data Processing Pipeline

### 1. Image Download
- Downloads images from URLs in parallel
- Handles failures and missing images
- Saves image paths to CSV

### 2. Embedding Generation
- Text embeddings: 512-dimensional using CLIP
- Image embeddings: 512-dimensional using CLIP
- Handles missing images by setting embeddings to None

### 3. Embedding Concatenation
- Concatenates text and image embeddings
- Results in 1024-dimensional feature vectors
- Validates dimensions before concatenation

### 4. Feature Engineering
- Adds extra features: L2 norm, mean, and standard deviation
- Final feature dimension: 1027 (1024 + 3)
- Standard scaling applied to all features

### 5. Target Processing
- Log transformation: `y = log1p(price)`
- Inverse transformation for predictions: `price = expm1(y_pred)`

## Training Process

### Data Split
- 85% training, 15% validation
- Random split using PyTorch's `random_split`

### Training Configuration
- Optimizer: AdamW with weight decay
- Learning rate: 1e-3
- Batch size: 64
- Max epochs: 150
- Early stopping: 25 epochs patience
- Learning rate scheduling: ReduceLROnPlateau

### Training Features
- Gradient clipping (max norm: 1.0)
- Batch normalization
- Dropout regularization
- Early stopping based on validation SMAPE
- Model checkpointing

## Prediction Process

### Ensemble Prediction
- Uses median + mean combination for robustness
- Weight: 70% median + 30% mean
- Reduces outlier effects

### Final Output
- Converts log predictions back to original scale
- Saves predictions to `final_test_out.csv`
- Format: `sample_id, price`

## Dependencies

### Core Libraries
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `torch`: Deep learning framework
- `scikit-learn`: Preprocessing and metrics

### Specialized Libraries
- `sentence-transformers`: CLIP model for embeddings
- `PIL`: Image processing
- `tqdm`: Progress bars
- `ast`: String parsing

## Configuration Files

### Input Paths
- Training data: `D:/DA/AAA/train_embed_16.csv`
- Test data: `D:/DA/AAA/test_embed_16.csv`
- Original dataset: `D:/DA/68e8d1d70b66d_student_resource/student_resource/dataset/test.csv`

### Output Files
- `final_test_out.csv`: Final predictions
- `download_results_test_75000.csv`: Download status
- `best_mlp_*.pt`: Saved model checkpoints

## Usage Instructions

### 1. Data Preparation
```bash
# Download images
python download_driver.py

# Generate embeddings for training data
python dummy2.py

# Generate embeddings for test data
python dummy.py

# Concatenate embeddings for training data
python concat2.py

# Concatenate embeddings for test data
python concat.py
```

### 2. Training and Prediction
```bash
python train_test.py
```

## Performance Metrics

### Primary Metric
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- Formula: `100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))`

### Model Performance
- Individual model SMAPEs are tracked during training
- Ensemble validation SMAPE is reported
- Early stopping prevents overfitting

## Key Features

### Robustness
- Handles missing images gracefully
- Validates embedding dimensions
- Uses median + mean for ensemble predictions
- Gradient clipping prevents exploding gradients

### Scalability
- Parallel image downloading
- Efficient batch processing
- GPU support with automatic device detection

### Reproducibility
- Fixed random seeds (implicit in PyTorch)
- Deterministic data splits
- Consistent preprocessing pipeline

## Notes

### Potential Issues
1. Hard-coded file paths may need adjustment for different environments
2. Missing `utils.py` file for image downloading functionality
3. Some commented-out code in dummy files suggests iterative development

### Recommendations
1. Use relative paths instead of absolute paths
2. Add configuration files for easy path management
3. Implement proper error handling and logging
4. Add data validation and quality checks
5. Consider using more recent CLIP models or fine-tuning

## File Dependencies

```
download_driver.py → utils.py (missing)
dummy.py → updated_test.csv
dummy2.py → updated_train.csv
concat.py → embed_test_16.csv
concat2.py → embed_train_16.csv
train_test.py → train_embed_16.csv, test_embed_16.csv
```

This documentation provides a comprehensive overview of the ML Challenge project, including its purpose, structure, implementation details, and usage instructions.
