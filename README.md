# Chest X-Ray Disease Classification — Part 1: Lung Segmentation Using U-Net

## Overview

This project is the first stage of a deep learning-based chest X-ray disease classification system. It uses a U-Net architecture to identify and segment lung regions from chest X-ray images.

The purpose of lung segmentation is to focus on the relevant lung regions and reduce the influence of irrelevant image backgrounds before disease classification.

This notebook implements dataset preparation, class balancing, dataset splitting, U-Net training, segmentation evaluation using the Dice coefficient, and generation of segmented lung images.

## Objectives

- Prepare a balanced chest X-ray dataset containing four classes.
- Load chest X-ray images and their corresponding lung masks.
- Preprocess images and masks for U-Net training.
- Train a U-Net model for lung segmentation.
- Evaluate segmentation performance using the Dice coefficient.
- Generate lung-segmented images for the subsequent classification stage.

## Dataset

**Dataset:** COVID-19 Radiography Database

The dataset is organized into four classes:

|Class|
COVID 
Normal
Viral Pneumonia
Lung Opacity 

The notebook selects **1,345 images per class**, giving a balanced subset of **5,380 images**.

Each selected X-ray image is paired with its corresponding lung mask.

### Dataset source

[Kaggle — COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- Scikit-learn
- U-Net for image segmentation

## Methodology

### 1. Dataset Preparation

The notebook loads the dataset, verifies image and mask availability, and matches images with their corresponding masks.

A fixed random seed is used for reproducibility, and 1,345 samples are selected from each class to create a balanced dataset.

### 2. Image Preprocessing

- Convert chest X-ray images to grayscale.
- Resize images to 224 × 224 pixels.
- Convert images and masks into tensors.
- Convert masks into binary values for segmentation.

The grayscale X-ray images are represented using three channels to match the U-Net input configuration.

### 3. Dataset Splitting

The dataset is divided into development and independent test sets using stratified sampling.

- Development set: used for model training and validation.
- Validation set: used to monitor segmentation performance and select the best checkpoint.
- Independent test set: reserved for evaluating the trained segmentation model.

The notebook uses a 15% independent test split and an 80:20 training-validation split within the development set.

### 4. U-Net Architecture

U-Net is an encoder-decoder convolutional neural network designed for image segmentation.

**Encoder:** Extracts image features through convolutional blocks and max pooling.

**Bottleneck:** Learns deeper representations of the input image.

**Decoder:** Reconstructs the spatial resolution using transposed convolutions and skip connections.

**Output layer:** Produces a single-channel segmentation map representing the predicted lung region.

### 5. Model Training

The notebook trains U-Net using the following configuration:

| Parameter | Configuration |
|---|---|
| Input image size | 224 × 224 |
| Input channels | 3 |
| Output channels | 1 |
| Batch size | 16 |
| Number of epochs | 15 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Segmentation loss | Defined by `segmentation_loss` in the notebook |
| Evaluation metric | Dice coefficient |

The model learns to predict lung masks by comparing its predictions against the corresponding ground-truth masks.

### 6. Segmentation Evaluation

The Dice coefficient measures the overlap between the predicted lung mask and the ground-truth mask.

A Dice score closer to 1 indicates greater overlap between the predicted and actual lung regions.

The notebook records training and validation loss and Dice scores, saves the best segmentation checkpoint, and calculates Dice performance on the independent test set.

### 7. Generation of Segmented Images

After training, the best U-Net checkpoint is loaded and used to predict lung masks.

The predicted masks are applied to the original X-ray tensors to suppress pixels outside the predicted lung regions. The resulting segmented images are saved for the next stage of the project.

## Workflow

1. Load the chest X-ray dataset.
2. Match X-ray images with their lung masks.
3. Create a balanced four-class dataset.
4. Preprocess images and masks.
5. Split the data into development and independent test sets.
6. Train the U-Net segmentation model.
7. Evaluate segmentation using the Dice coefficient.
8. Generate and save segmented lung images.
9. Use the segmented images in the subsequent classification stage.

## Expected Outputs

The notebook is configured to produce:

- A balanced dataset CSV file.
- An independent train, validation, and test split CSV file.
- A trained U-Net model checkpoint.
- Training and validation segmentation metrics.
- An independent test Dice score.
- Segmented chest X-ray images for subsequent classification.

The generated files are saved under `/kaggle/working/` when the notebook is executed in Kaggle.

## Project Status

**Completed in this notebook:** Dataset preparation, U-Net segmentation training, Dice-based evaluation, and segmented image generation.

**Planned for subsequent stages:** ConvNeXt-Tiny and RepViT-M1 classification, feature fusion, Grad-CAM explainability, five-fold cross-validation, and final classification evaluation.

## How to Run

1. Open the notebook in Kaggle.
2. Add the COVID-19 Radiography Database to the notebook's input datasets.
3. Enable GPU acceleration if available.
4. Run the notebook cells sequentially.
5. Check the segmentation metrics and generated output files.

The dataset path used in the notebook is configured for Kaggle and may need to be updated if running in another environment.

