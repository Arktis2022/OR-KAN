import os
import sys
import argparse
import warnings
import numpy as np
import torch
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchio as tio
from scipy.ndimage import zoom
from skimage.transform import resize
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Suppress warnings
warnings.filterwarnings("ignore")

# Import local modules
from model import OR_KAN
from dataloader import *

# Command line argument setup
parser = argparse.ArgumentParser(description='MRI Quality Control with OR-KAN Model')
parser.add_argument('--input_dir', type=str, default='./data_example_with_mask', 
                    help='Directory containing input MRI data')
parser.add_argument('--model_path', type=str, default='./checkpoint/OR_KAN_weight.pth',
                    help='Path to the pretrained model weights')
parser.add_argument('--slice_used', type=int, default=14, 
                    help='Number of slices to use for inference')
parser.add_argument('--rotation', type=int, default=1,
                    help='Whether to use rotation augmentation (0=No, 1=Yes)')
parser.add_argument('--quality_threshold', type=float, default=0.4,
                    help='Threshold for determining high/low quality (>threshold = high quality)')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def rotate_image(input_img):
    """
    Create multiple rotated versions of the input image.
    
    Args:
        input_img: Input tensor image
        
    Returns:
        Tensor containing rotated versions of the input image
    """
    angles = torch.linspace(0, 360, 8)
    rotated_tensors = []
    rotate_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor()
    ])
    
    for _ in angles:  # We don't use angle values since RandomRotation handles it
        transformed_tensor = rotate_transform(input_img.squeeze(0))
        rotated_tensors.append(transformed_tensor.unsqueeze(0))
    
    return torch.cat(rotated_tensors)

def calculate_predictive_entropy(predictions):
    """
    Calculate normalized predictive entropy from a list of predictions.
    
    Args:
        predictions: List of prediction tensors
        
    Returns:
        Normalized entropy value
    """
    epsilon = sys.float_info.epsilon
    predictions = torch.stack(predictions)
    
    # Calculate predictive entropy
    mean_predictions = torch.mean(predictions, dim=0)
    predictive_entropy = -torch.sum(mean_predictions * torch.log(mean_predictions + epsilon), dim=-1)
    
    # Normalize entropy
    max_entropy = torch.log(torch.tensor(predictions.size(-1), dtype=torch.float32))
    normalized_entropy = predictive_entropy / max_entropy
    
    return normalized_entropy.item()

def main():
    """Main function to run the quality assessment pipeline."""
    # Load model
    model = OR_KAN().to(device)
    model.load_state_dict(torch.load(args.model_path)['net'])
    model.eval()
    
    # Process input data
    image_names, images = change_and_save_image_inference(args.input_dir)
    print("Processing images:")
    for name, img in zip(image_names, images):
        print(f"Image: {name}, Shape: {img.shape}")
    
    processed_images, processed_names = traverse_and_modify(images, image_names)
    
    # Create dataset and dataloader
    test_dataset = MRIDatasetQuality_inference(processed_images, processed_names, transform=data_transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Perform quality assessment
    quality_scores = []
    image_paths = []
    quality_labels = []
    
    with torch.no_grad():
        for batch, img_path in test_dataloader:
            image = batch.to(device)
            output_list = []
            
            for i in range(image.shape[1]):
                img = image[:, i:i+1, :, :].float()
                
                if args.rotation == 0:
                    output = model(img)
                    output_list.append(output.cpu())
                else:
                    input_tensor = img.cpu()
                    rotated_images = rotate_image(input_tensor)
                    rotated_images = rotated_images.to(device)
                    output = torch.mean(model(rotated_images), dim=0)
                    output_list.append(output.cpu())
            
            entropy = calculate_predictive_entropy(output_list)
            quality_score = 1 - entropy
            quality_scores.append(quality_score)
            image_paths.append(img_path[0])
            
            quality_label = "high_quality" if quality_score > args.quality_threshold else "low_quality"
            quality_labels.append(quality_label)
    
    # Display results
    print("Quality Assessment Results:")
    print("--------------------------")
    print("Image Paths:")
    for path in image_paths:
        print(f"  - {path}")
    
    print("Quality Scores:")
    for i, score in enumerate(quality_scores):
        print(f"  - {image_paths[i]}: {score:.4f}")
    
    print("Quality Classification:")
    for i, label in enumerate(quality_labels):
        print(f"  - {image_paths[i]}: {label}")

if __name__ == "__main__":
    main()