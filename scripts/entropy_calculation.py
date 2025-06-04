import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
from PIL import Image

# Add your src path - adjust as needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import UNET
from utils.utils import get_loaders_with_augmentation, Config

class MultiClassEntropyCalculator:
    def __init__(self, model_path, num_classes=2, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the entropy calculator with a trained model.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            num_classes (int): Number of segmentation classes (C in Equation 3). Default=2 for binary segmentation
            device (str): Device to run inference on
        """
        self.device = device
        self.num_classes = num_classes
        self.model = self._load_model(model_path)
        
        print(f"Initialized for {self.num_classes}-class segmentation")
        if self.num_classes == 2:
            print("Using binary segmentation mode (background + foreground)")
        else:
            print(f"Using multi-class segmentation mode with {self.num_classes} classes")
        
    def _load_model(self, model_path):
        """Load the trained U-Net model from checkpoint."""
        print(f"Loading model from: {model_path}")
        
        # Initialize model based on number of classes
        if self.num_classes == 2:
            # Binary segmentation: single output channel with sigmoid
            model = UNET(in_channels=3, out_channels=1).to(self.device)
        else:
            # Multi-class segmentation: C output channels with softmax
            model = UNET(in_channels=3, out_channels=self.num_classes).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("Model loaded successfully!")
        return model
    
    def calculate_entropy_equation3(self, predicted_probs, epsilon=1e-8):
        """
        Calculate entropy using Equation 3 from the paper:
        H_n = -∑(j=1 to C) P_n,j(x_n) · log(P_n,j(x_n))
        
        Args:
            predicted_probs (torch.Tensor): Predicted probabilities [H, W, C]
            epsilon (float): Small value to avoid log(0)
            
        Returns:
            float: Total entropy value for the image
        """
        # Ensure probabilities are in range [0, 1] and add epsilon for numerical stability
        predicted_probs = torch.clamp(predicted_probs, epsilon, 1.0 - epsilon)
        
        # Calculate entropy: H = -∑ p * log(p)
        log_probs = torch.log(predicted_probs)
        entropy = -torch.sum(predicted_probs * log_probs)
        
        return entropy.item()
    
    def calculate_pixel_wise_entropy(self, predicted_probs, epsilon=1e-8):
        """
        Calculate pixel-wise entropy and return mean entropy for the image.
        This calculates entropy per pixel and averages across all pixels.
        
        Args:
            predicted_probs (torch.Tensor): Predicted probabilities [H, W, C]
            epsilon (float): Small value to avoid log(0)
            
        Returns:
            float: Mean entropy value across all pixels
        """
        # Clamp probabilities for numerical stability
        predicted_probs = torch.clamp(predicted_probs, epsilon, 1.0 - epsilon)
        
        # Calculate entropy per pixel: H = -∑ p * log(p) across classes (dim=-1)
        log_probs = torch.log(predicted_probs)
        pixel_entropy = -torch.sum(predicted_probs * log_probs, dim=-1)  # Sum over classes
        
        # Return mean entropy across all pixels
        return torch.mean(pixel_entropy).item()
    
    def calculate_class_wise_entropy(self, predicted_probs, epsilon=1e-8):
        """
        Calculate entropy for each class separately.
        
        Args:
            predicted_probs (torch.Tensor): Predicted probabilities [H, W, C]
            epsilon (float): Small value to avoid log(0)
            
        Returns:
            dict: Entropy values for each class
        """
        predicted_probs = torch.clamp(predicted_probs, epsilon, 1.0 - epsilon)
        
        class_entropies = {}
        for c in range(self.num_classes):
            class_prob = predicted_probs[:, :, c]
            # Calculate entropy for this class: -p*log(p) - (1-p)*log(1-p)
            log_prob = torch.log(class_prob)
            log_one_minus_prob = torch.log(1 - class_prob)
            class_entropy = -torch.mean(class_prob * log_prob + (1 - class_prob) * log_one_minus_prob)
            class_entropies[f'class_{c}_entropy'] = class_entropy.item()
        
        return class_entropies
    
    def predict_and_calculate_entropy(self, image_tensor):
        """
        Generate prediction and calculate entropy for a single image.
        
        Args:
            image_tensor (torch.Tensor): Input image tensor [1, 3, H, W]
            
        Returns:
            tuple: (predicted_mask, entropy_dict)
        """
        with torch.no_grad():
            # Get model prediction (logits)
            predictions, _ = self.model(image_tensor)
            
            if self.num_classes == 2:
                # Binary segmentation: use sigmoid
                probs = torch.sigmoid(predictions)  # Shape: [1, 1, H, W]
                
                # Create [background_prob, foreground_prob]
                background_prob = 1 - probs  # Probability of background (class 0)
                foreground_prob = probs      # Probability of foreground (class 1)
                
                # Stack to create [H, W, 2] tensor
                multi_class_probs = torch.stack([background_prob.squeeze(), foreground_prob.squeeze()], dim=-1)
                
                # Convert prediction to binary mask
                predicted_mask = (probs > 0.5).float()
                
            else:
                # Multi-class segmentation: use softmax
                probs = torch.softmax(predictions, dim=1)  # Shape: [1, C, H, W]
                
                # Rearrange to [H, W, C]
                multi_class_probs = probs.squeeze(0).permute(1, 2, 0)  # [H, W, C]
                
                # Convert prediction to class mask (argmax)
                predicted_mask = torch.argmax(probs, dim=1).float()  # [1, H, W]
            
            # Calculate different entropy measures
            entropy_results = {}
            
            # Equation 3 entropy (total entropy)
            entropy_results['entropy_equation3'] = self.calculate_entropy_equation3(multi_class_probs)
            
            # Pixel-wise entropy (mean per-pixel entropy)
            entropy_results['pixel_wise_entropy'] = self.calculate_pixel_wise_entropy(multi_class_probs)
            
            # Class-wise entropy
            class_entropies = self.calculate_class_wise_entropy(multi_class_probs)
            entropy_results.update(class_entropies)
            
            # Additional statistics (for internal use only - not saved to CSV)
            pixel_entropies = -torch.sum(multi_class_probs * torch.log(torch.clamp(multi_class_probs, 1e-8, 1.0)), dim=-1)
            max_pixel_entropy = float(torch.max(pixel_entropies))
            min_pixel_entropy = float(torch.min(pixel_entropies))
            std_pixel_entropy = float(torch.std(pixel_entropies))
            
            # Store for internal calculations but don't include in final results
            # entropy_results['max_pixel_entropy'] = max_pixel_entropy
            # entropy_results['min_pixel_entropy'] = min_pixel_entropy  
            # entropy_results['std_pixel_entropy'] = std_pixel_entropy
            
            return predicted_mask, entropy_results
    
    def process_dataset(self, train_img_dir, val_img_dir, batch_size=1, save_results=True, output_file="entropy_results.csv"):
        """
        Process entire dataset and calculate entropy for all images.
        
        Args:
            train_img_dir (str): Directory containing training images
            val_img_dir (str): Directory containing validation images (can be same as train)
            batch_size (int): Batch size for processing (recommend 1 for individual image analysis)
            save_results (bool): Whether to save results to CSV
            output_file (str): Output CSV filename
            
        Returns:
            pandas.DataFrame: Results containing image names and entropy values
        """
        # Create transforms (no augmentation for inference)
        transforms = A.Compose([
            A.Resize(height=256, width=256),
            ToTensorV2(),
        ])
        
        # Setup data loader without augmentation
        config = Config(
            flip_rate=0.0,
            local_rate=0.0,
            nonlinear_rate=0.0,
            paint_rate=0.0,
            inpaint_rate=0.0
        )
        
        # Get data loader
        train_loader, _ = get_loaders_with_augmentation(
            train_img_dir,
            val_img_dir,
            batch_size,
            transforms,
            transforms,
            num_workers=1,
            pin_memory=True,
            config=config,
            shuffle_train=False,  # Don't shuffle to maintain order
        )
        
        results = []
        
        print(f"Processing images and calculating entropy for {self.num_classes}-class segmentation...")
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(self.device)
            
            # Process each image in the batch
            for i in range(data.shape[0]):
                image_tensor = data[i:i+1]  # Keep batch dimension
                
                # Get prediction and entropy
                predicted_mask, entropy_dict = self.predict_and_calculate_entropy(image_tensor)
                
                # Calculate sequential image index (since batch_size=1, this is just batch_idx)
                sequential_idx = batch_idx * batch_size + i
                
                # Get image filename if available
                if hasattr(train_loader.dataset, 'img_list'):
                    if sequential_idx < len(train_loader.dataset.img_list):
                        img_name = os.path.basename(train_loader.dataset.img_list[sequential_idx])
                    else:
                        img_name = f"image_{sequential_idx}"
                else:
                    img_name = f"image_{sequential_idx}"
                
                # Store results (removed batch_idx, kept only image_idx as sequential counter)
                result_dict = {
                    'image_name': img_name,
                    'num_classes': self.num_classes,
                    'image_idx': sequential_idx  # Sequential order: 0, 1, 2, 3, ...
                }
                result_dict.update(entropy_dict)
                results.append(result_dict)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Keep original order - NO SORTING by entropy
        # Results maintain the same order as images appear in the directory
        
        print(f"\nEntropy Statistics for {self.num_classes}-class segmentation:")
        print(f"Total images processed: {len(results_df)}")
        # print(f"Mean Equation 3 entropy: {results_df['entropy_equation3'].mean():.4f}")
        # print(f"Mean pixel-wise entropy: {results_df['pixel_wise_entropy'].mean():.4f}")
        # print(f"Max pixel-wise entropy: {results_df['pixel_wise_entropy'].max():.4f}")
        # print(f"Min pixel-wise entropy: {results_df['pixel_wise_entropy'].min():.4f}")
        
        # Show class-wise entropy statistics if available
        for c in range(self.num_classes):
            class_col = f'class_{c}_entropy'
            if class_col in results_df.columns:
                print(f"Mean {class_col}: {results_df[class_col].mean():.4f}")
        
        # print(f"\nFirst 10 images (in directory order):")
        display_cols = ['image_name', 'pixel_wise_entropy', 'entropy_equation3']
        # print(results_df[display_cols].head(10))
        
        # print(f"\nLast 10 images (in directory order):")
        # print(results_df[display_cols].tail(10))
        
        # Save results if requested - maintaining original order
        if save_results:
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file} (maintaining directory order)")
        
        return results_df

def main():
    """
    Main function to run entropy calculation.
    Update the paths and num_classes according to your setup.
    """
    # Configuration - UPDATE THESE PATHS AND PARAMETERS
    MODEL_PATH = 'results/ssl_trained_model/2025-04-19_01-19-21/self_supervised_learning.pt'
    TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
    VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/"
    OUTPUT_FILE = "entropy_results.csv"
    
    # SET NUMBER OF CLASSES HERE
    NUM_CLASSES = 2  # Default: 2 for binary segmentation (background + lesion)
    # For multi-class: set to 3, 4, 5, etc. depending on your task
    
    print(f"Setting up entropy calculation for {NUM_CLASSES}-class segmentation")
    
    # Initialize entropy calculator
    entropy_calc = MultiClassEntropyCalculator(MODEL_PATH, num_classes=NUM_CLASSES)
    
    # Process dataset and calculate entropy
    results_df = entropy_calc.process_dataset(
        train_img_dir=TRAIN_IMG_DIR,
        val_img_dir=VAL_IMG_DIR,
        batch_size=1,
        save_results=True,
        output_file=OUTPUT_FILE
    )
    
    # print(f"\nProcessing complete! Results available in '{OUTPUT_FILE}'")
    
    return results_df

# Example usage for different number of classes
def example_usage():
    """Examples showing how to use for different numbers of classes."""
    
    # Example 1: Binary segmentation (C=2) - Default
    print("=== Binary Segmentation Example (C=2) ===")
    entropy_calc_binary = MultiClassEntropyCalculator(
        model_path='your_model.pt', 
        num_classes=2  # background + lesion
    )
    
    # Example 2: 3-class segmentation (C=3)
    print("\n=== 3-Class Segmentation Example (C=3) ===")
    entropy_calc_3class = MultiClassEntropyCalculator(
        model_path='your_model.pt', 
        num_classes=3  # background + benign + malignant
    )
    
    # Example 3: 5-class segmentation (C=5)
    print("\n=== 5-Class Segmentation Example (C=5) ===")
    entropy_calc_5class = MultiClassEntropyCalculator(
        model_path='your_model.pt', 
        num_classes=5  # multiple lesion types
    )

if __name__ == "__main__":
    results = main()