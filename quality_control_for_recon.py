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
import shutil  
from scipy.ndimage import zoom  
from skimage.transform import resize  
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report  
from torch import nn  
from torch.utils.data import Dataset, DataLoader  
from torch.utils.tensorboard import SummaryWriter  
from torchvision import transforms  
from collections import Counter  

# Suppress warnings  
warnings.filterwarnings("ignore")  

# Import local modules  
from model import OR_KAN  
from dataloader import *  

# Command line argument setup  
parser = argparse.ArgumentParser(description='MRI Quality Control and Best View Selection')  
parser.add_argument('--input_dir', type=str, default='./data_example_with_mask',   
                    help='Directory containing input MRI data')  
parser.add_argument('--output_dir', type=str, default='./selected_best_views',  
                    help='Directory to save selected best views')  
parser.add_argument('--model_path', type=str, default='./checkpoint/OR_KAN_weight.pth',  
                    help='Path to the pretrained model weights')  
parser.add_argument('--slice_used', type=int, default=14,   
                    help='Number of slices to use for inference')  
parser.add_argument('--rotation', type=int, default=1,  
                    help='Whether to use rotation augmentation (0=No, 1=Yes)')  
args = parser.parse_args()  

# Device configuration  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

# View orientations  
views = ['cor', 'tra', 'sag']  # coronal, transverse, sagittal  

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
    """Main function to select best views and copy to output directory"""  
    # Load model  
    model = OR_KAN().to(device)  
    model.load_state_dict(torch.load(args.model_path)['net'])  
    model.eval()  
    
    # Create output directory  
    if not os.path.exists(args.output_dir):  
        os.makedirs(args.output_dir)  
    else:  
        # Clear output directory  
        for item in os.listdir(args.output_dir):  
            item_path = os.path.join(args.output_dir, item)  
            if os.path.isfile(item_path):  
                os.remove(item_path)  
            elif os.path.isdir(item_path):  
                shutil.rmtree(item_path)  
    
    # Process input data using original preprocessing logic  
    image_names, images = change_and_save_image_inference(args.input_dir)  
    print("Processing images:")  
    for name, img in zip(image_names, images):  
        print(f"Image: {name}, Shape: {img.shape}")  
    
    processed_images, processed_names = traverse_and_modify(images, image_names)  
    
    # Create dataset and dataloader  
    test_dataset = MRIDatasetQuality_inference(processed_images, processed_names, transform=data_transform_test)  
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  
    
    # Store best images for each orientation  
    best_images = {  
        'cor': {'path': None, 'score': -1},  
        'tra': {'path': None, 'score': -1},  
        'sag': {'path': None, 'score': -1}  
    }  
    
    # Process each image to determine orientation and quality  
    with torch.no_grad():  
        for batch, img_path in test_dataloader:  
            print(f"Processing image: {img_path[0]}")  
            
            # Skip if image doesn't have enough slices  
            if batch.shape[1] != args.slice_used:  
                print(f"  Skipping: Invalid number of slices ({batch.shape[1]})")  
                continue  
            
            image = batch.to(device)  
            output_list = []  
            class_outputs = []  
            
            for i in range(image.shape[1]):  
                img = image[:, i:i+1, :, :].float()  
                
                if args.rotation == 0:  
                    output = model(img)  
                    output_list.append(output.cpu())  
                    class_outputs.append(torch.argmax(output).item())  
                else:  
                    input_tensor = img.cpu()  
                    rotated_images = rotate_image(input_tensor)  
                    rotated_images = rotated_images.to(device)  
                    output = torch.mean(model(rotated_images), dim=0)  
                    output_list.append(output.cpu())  
                    class_outputs.append(torch.argmax(output).item())  
            
            # Calculate quality score using entropy  
            entropy = calculate_predictive_entropy(output_list)  
            quality_score = 1 - entropy  
            
            # Determine orientation based on most common prediction  
            orientation_counts = Counter(class_outputs)  
            most_common_orientation = orientation_counts.most_common(1)[0][0]  
            orientation = views[most_common_orientation]  
            
            print(f"  Orientation: {orientation}, Quality score: {quality_score:.4f}")  
            
            # Update best images  
            if quality_score > best_images[orientation]['score']:  
                best_images[orientation]['path'] = img_path[0]  
                best_images[orientation]['score'] = quality_score  
    
    # Copy best images to output directory  
    print("\nSelected best views:")  
    for orientation, data in best_images.items():  
        if data['path'] is not None:  
            file_name = os.path.basename(data['path'])  
            output_path = os.path.join(args.output_dir, f"{orientation}_{file_name}")  
            

            shutil.copy2(args.input_dir+'/'+data['path'], output_path)  
            print(f"  {orientation}: {data['path']} (Score: {data['score']:.4f})")  
        else:  
            print(f"  {orientation}: No valid image found")  

if __name__ == "__main__":  
    main()  