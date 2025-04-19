# Standard library imports
import os
import random
import copy
import numpy as np

# Third-party imports
import PIL.Image as Image_PIL
import PIL.ImageDraw as ImageDraw_PIL
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.special import comb

# =================================================
#          Train and validation data loader
# =================================================

def get_loaders_with_augmentation(
    train_dir,
    val_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    config=None,
    shuffle_training=True
):
    """
    Create data loaders for training and validation with augmentation.
    
    Args:
        train_dir: Directory containing training images
        val_dir: Directory containing validation images
        batch_size: Batch size for data loaders
        train_transform: Albumentations transforms for training images
        val_transform: Albumentations transforms for validation images
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        config: Configuration for augmentation parameters
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    class Config:
        def __init__(self, flip_rate=0.5, local_rate=0.5, nonlinear_rate=0.5, paint_rate=0.5, inpaint_rate=0.5):
            self.flip_rate = flip_rate
            self.local_rate = local_rate
            self.nonlinear_rate = nonlinear_rate
            self.paint_rate = paint_rate
            self.inpaint_rate = inpaint_rate

    if config is None:
        config = Config()
    
    def local_pixel_shuffling(x, prob=0.5):
        if random.random() >= prob:
            return x
        image_temp = x.clone()
        orig_image = x.clone()
        img_rows, img_cols, img_channels = x.shape
        num_block = 10000
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows//10)
            block_noise_size_y = random.randint(1, img_cols//10)
            noise_x = random.randint(0, img_rows-block_noise_size_x)
            noise_y = random.randint(0, img_cols-block_noise_size_y)
            window = orig_image[noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :]
            window = window.flatten()
            window = window[torch.randperm(len(window))]
            window = window.reshape((block_noise_size_x, block_noise_size_y, img_channels))
            image_temp[noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :] = window
        return image_temp

    def bernstein_poly(i, n, t):
        return torch.tensor(comb(n, i)) * (t**(n-i)) * (1 - t)**i

    def bezier_curve(points, nTimes=1000):
        nPoints = len(points)
        xPoints = torch.tensor([p[0] for p in points], dtype=torch.float32)
        yPoints = torch.tensor([p[1] for p in points], dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, nTimes)
        polynomial_array = torch.stack([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
        xvals = torch.matmul(xPoints, polynomial_array)
        yvals = torch.matmul(yPoints, polynomial_array)
        return xvals, yvals

    def nonlinear_transformation(x, prob=0.5):
        if random.random() >= prob:
            return x
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xvals, yvals = bezier_curve(points, nTimes=100000)
        if random.random() < 0.5:
            xvals, _ = torch.sort(xvals)
        else:
            xvals, _ = torch.sort(xvals)
            yvals, _ = torch.sort(yvals)
        xvals_np = xvals.cpu().numpy()
        yvals_np = yvals.cpu().numpy()
        x_np = x.cpu().numpy()
        nonlinear_x_np = np.interp(x_np, xvals_np, yvals_np)
        nonlinear_x = torch.tensor(nonlinear_x_np, dtype=torch.float32)
        return nonlinear_x

    def image_in_painting(x):
        img_rows, img_cols, img_channels = x.shape
        cnt = 5
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows//6, img_rows//3)
            block_noise_size_y = random.randint(img_cols//6, img_cols//3)
            noise_x = random.randint(3, img_rows-block_noise_size_x-3)
            noise_y = random.randint(3, img_cols-block_noise_size_y-3)
            x[noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :] = torch.rand(block_noise_size_x, block_noise_size_y, img_channels)
            cnt -= 1
        return x

    def image_out_painting(x):
        img_rows, img_cols, img_channels = x.shape
        image_temp = x.clone()
        x = torch.rand(x.shape)
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        x[noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :] = image_temp[noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :]
        cnt = 4
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
            block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
            noise_x = random.randint(3, img_rows-block_noise_size_x-3)
            noise_y = random.randint(3, img_cols-block_noise_size_y-3)
            x[noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :] = image_temp[noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :]
            cnt -= 1
        return x
    
    def normalize_minus1_to_1(tensor):
        """
        Normalize tensor values from [0, 1] to [-1, 1] using (pixel_value-0.5)/0.5
        """
        return (tensor - 0.5) / 0.5

    class AutoencoderTrainDataset(Dataset):
        def __init__(self, image_dir, transform=None, config=None):
            self.image_dir = image_dir
            self.transform = transform
            self.config = config if config else Config()
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".png", ".jpg", ".jpeg"))])
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            
            resize_only = [t for t in train_transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])   
                
            if len(self.images) == 0:
                raise ValueError(f"No images found in {image_dir}")
                    
            print(f"Found {len(self.images)} training images in {image_dir}")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load and transform image
            img_path = os.path.join(self.image_dir, self.images[idx])
            image = Image.open(img_path).convert("RGB")               
                            
            # Apply album transforms
            if self.transform:
                transformed = self.transform(image=np.array(image))
                image_tensor = transformed["image"] / 255.0 # This is already a tensor from ToTensorV2()

                transformed_prediction = self.resize_only_transform(image=np.array(image))
                image_tensor_prediction = transformed_prediction["image"] / 255.0 # This is already a tensor from ToTensorV2()
            else:
                image_tensor = transforms.ToTensor()(image)
                image_tensor_prediction = transforms.ToTensor()(image)

            
            
            # Create grayscale version of the original image
            grayscale_tensor = self.rgb_to_grayscale(image_tensor_prediction)
            
            # Apply augmentation
            augmented_tensor = self.augment_image(image_tensor.clone())
            
            # Normalize to [-1, 1] range
            augmented_tensor = normalize_minus1_to_1(augmented_tensor)
            grayscale_tensor = normalize_minus1_to_1(grayscale_tensor)
            
            return augmented_tensor, grayscale_tensor

        def rgb_to_grayscale(self, rgb_tensor):
            """Convert RGB tensor to grayscale"""
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
            grayscale = torch.sum(rgb_tensor * weights.view(3, 1, 1), dim=0, keepdim=True)
            return grayscale
        
        def augment_image(self, image_tensor):
            x = image_tensor.permute(1, 2, 0).clone()
            if random.random() < self.config.flip_rate:
                flip_count = 0
                max_flips = 3
                while random.random() < 0.5 and flip_count < max_flips:
                    degree = random.choice([0, 1])
                    x = torch.flip(x, [degree])
                    flip_count += 1
            x = local_pixel_shuffling(x, prob=self.config.local_rate)
            x = nonlinear_transformation(x, self.config.nonlinear_rate)
            if random.random() < self.config.paint_rate:
                if random.random() < self.config.inpaint_rate:
                    x = image_in_painting(x)
                else:
                    x = image_out_painting(x)
            return x.permute(2, 0, 1)

    class AutoencoderValDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".png", ".jpg", ".jpeg"))])
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            
            resize_only = [t for t in val_transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])        

            if len(self.images) == 0:
                raise ValueError(f"No images found in {image_dir}")
                    
            print(f"Found {len(self.images)} validation images in {image_dir}")               

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load and transform image
            img_path = os.path.join(self.image_dir, self.images[idx])
            image = Image.open(img_path).convert("RGB")
                                        
            # Apply album transforms
            if self.transform:
                transformed = self.transform(image=np.array(image))
                image_tensor = transformed["image"] / 255.0  # This is already a tensor from ToTensorV2()

                transformed_prediction = self.resize_only_transform(image=np.array(image))
                image_tensor_prediction = transformed_prediction["image"] / 255.0 # This is already a tensor from ToTensorV2()
            else:
                image_tensor = transforms.ToTensor()(image)
                image_tensor_prediction = transforms.ToTensor()(image)            
            
            # Create grayscale version of the original image
            grayscale_tensor = self.rgb_to_grayscale(image_tensor_prediction)
            
            # Normalize to [-1, 1] range
            image_tensor = normalize_minus1_to_1(image_tensor)
            grayscale_tensor = normalize_minus1_to_1(grayscale_tensor)
            
            return image_tensor, grayscale_tensor
    
        def rgb_to_grayscale(self, rgb_tensor):
            """Convert RGB tensor to grayscale"""
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
            grayscale = torch.sum(rgb_tensor * weights.view(3, 1, 1), dim=0, keepdim=True)
            return grayscale
        
        

    # Initialize datasets
    train_ds = AutoencoderTrainDataset(train_dir, transform=train_transform, config=config)
    val_ds = AutoencoderValDataset(val_dir, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=shuffle_training,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=True,  # Drop the last incomplete batch to avoid shape issues
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=True,  # Drop the last incomplete batch to avoid shape issues
    )

    return train_loader, val_loader

def inspect_pixel_value_range(loader, name="Loader"):
    print(f"Inspecting pixel value ranges for {name}")
    max_input = float('-inf')
    max_target = float('-inf')
    min_input = float('inf')
    min_target = float('inf')

    for inputs, targets in loader:
        max_input = max(max_input, inputs.max().item())
        min_input = min(min_input, inputs.min().item())
        max_target = max(max_target, targets.max().item())
        min_target = min(min_target, targets.min().item())
        break  # Remove this line if you want to scan the full loader

    print(f"{name} Input  - min: {min_input:.4f}, max: {max_input:.4f}")
    print(f"{name} Target - min: {min_target:.4f}, max: {max_target:.4f}")

# =================================================
#             Save predictions as images
# =================================================
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """
    Save the input images, predictions, and targets as a single combined image with labels.
    
    Args:
        loader: DataLoader containing the images to make predictions on
        model: The UNET model that returns both segmentation and bottleneck features
        folder: Folder to save the images to
        device: Device to use for predictions
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        
        x = x.to(device=device)
        y = y.to(device=device)
        y = (y + 1) / 2 # Normalize this from [-1, 1] to [0, 1]

        with torch.no_grad():
            # Handle the dual output of UNET model
            preds, _ = model(x)
            preds = torch.sigmoid(preds) # Convert logits to probabilities [0, 1]
        
        # Normalize this from [-1, 1] to [0, 1] for visualization
        normalized_x = (x + 1) / 2        
        
        # Check channel dimensions
        input_channels = normalized_x.size(1)
        pred_channels = preds.size(1)
        target_channels = y.size(1)                      
    
        # Combine input, prediction, and target horizontally
        batch_size = x.size(0)
        for b in range(batch_size):
            # Get single images from batch
            input_img = normalized_x[b:b+1]
            pred_img = preds[b:b+1]
            target_img = y[b:b+1]
            
            # Convert single-channel predictions and targets to 3-channel if needed
            if pred_channels == 1 and input_channels == 3:
                pred_img = pred_img.repeat(1, 3, 1, 1)
            
            if target_channels == 1 and input_channels == 3:
                target_img = target_img.repeat(1, 3, 1, 1)
            
            # Get image dimensions
            _, _, h, w = input_img.shape
            
            # Create label bands (20 pixels high black bars with white text)
            label_height = 20
            label_tensor = torch.zeros(1, 3, label_height, w * 3).to(device)
            
            # Create combined image with labels
            combined_with_labels = torch.zeros(1, 3, h + label_height, w * 3).to(device)
            
            # Add images to the combined tensor
            combined_with_labels[0, :, label_height:, 0:w] = input_img[0]
            combined_with_labels[0, :, label_height:, w:2*w] = pred_img[0]
            combined_with_labels[0, :, label_height:, 2*w:3*w] = target_img[0]
            
            # Add label bar at the top
            combined_with_labels[0, :, 0:label_height, :] = label_tensor[0]
            
            # Save the combined image with labels
            torchvision.utils.save_image(
                combined_with_labels, f"{folder}/batch{idx}_img{b}_combined.png"
            )         
                     
            # Convert tensor to PIL Image
            pil_img = Image_PIL.open(f"{folder}/batch{idx}_img{b}_combined.png")
            draw = ImageDraw_PIL.Draw(pil_img)
            
            # Add text labels (using default font)
            font = None  # Use default font
            
            # Label positions (center of each image section)
            label_positions = [
                (w//2, label_height//2),                  # Input
                (w + w//2, label_height//2),              # Prediction
                (2*w + w//2, label_height//2)             # Target
            ]
            
            # Labels
            labels = ["Input", "Prediction", "Target"]
            
            # Add text
            for pos, label in zip(label_positions, labels):
                draw.text(pos, label, fill="white", font=font, anchor="mm")
            
            # Save the image with labels
            pil_img.save(f"{folder}/batch{idx}_img{b}_combined.png")
        
        # Only save a few batches to avoid filling disk
        # if idx >= 2:
        #     break
    
    model.train()

# =================================================
#             Checkpoint handling
# =================================================
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save a checkpoint of the model.
    
    Args:
        state: Dictionary containing the model state and other training info
        filename: Path to save the checkpoint to
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
    Load a checkpoint into the model.
    
    Args:
        checkpoint: Dictionary containing the model state
        model: Model to load the weights into
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

# =================================================
#             Autoencoder specific utilities
# =================================================
def calculate_dice_score(pred, target):
    """
    Calculate the Dice score between prediction and target.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        
    Returns:
        float: Dice score
    """
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def load_trained_model(model, optimizer, scheduler, model_path):
    """
    Load a trained model from a checkpoint.
    
    Args:
        model: Model to load the weights into
        optimizer: Optimizer to load the state into
        scheduler: Scheduler to load the state into
        model_path: Path to the checkpoint file
        
    Returns:
        tuple: Updated model, optimizer, scheduler, and epoch
    """
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return model, optimizer, scheduler, 0

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded. Trained for {epoch} epochs.")
    print_current_lr(scheduler)
    return model, optimizer, scheduler, epoch

def print_current_lr(scheduler):
    """
    Print the current learning rate from the scheduler.
    
    Args:
        scheduler: Learning rate scheduler
    """
    current_lr = scheduler.optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")

# =================================================
#             Visualization utilities
# =================================================
def visualize_autoencoder_results(model, test_loader, device, num_samples=5):
    """
    Visualize the results of the autoencoder on test samples.
    
    Args:
        model: Trained autoencoder model
        test_loader: DataLoader containing test images
        device: Device to use for predictions
        num_samples: Number of samples to visualize
    """
    model.eval()
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            x = x.to(device)
            # Get reconstruction and bottleneck features
            reconstruction, bottleneck = model(x)
            
            # Move tensors back to CPU and convert to numpy
            orig_img = x[0].cpu().permute(1, 2, 0).numpy()
            recon_img = reconstruction[0].cpu().permute(1, 2, 0).numpy()
            target_img = y[0].cpu().numpy()
            
            # Normalize for display
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
            
            # Display images
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(target_img, cmap='gray')
            axes[2, i].set_title('Target Mask')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('autoencoder_results.png')
    plt.close()
    model.train()
    print("Autoencoder visualization saved to 'autoencoder_results.png'")

def visualize_bottleneck(model, sample_image, device, save_path='bottleneck_features.png'):
    """
    Visualize the bottleneck features of the autoencoder.
    
    Args:
        model: Trained autoencoder model
        sample_image: Sample image tensor to visualize bottleneck for
        device: Device to use for predictions
        save_path: Path to save the visualization to
    """
    model.eval()
    
    with torch.no_grad():
        # Move image to device and add batch dimension if necessary
        if len(sample_image.shape) == 3:
            sample_image = sample_image.unsqueeze(0)
        sample_image = sample_image.to(device)
        
        # Get bottleneck features
        _, bottleneck = model(sample_image)
        
        # Get the number of feature channels
        num_channels = bottleneck.size(1)
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        # Flatten axes for easy indexing
        axes = axes.flatten()
        
        # Plot each channel
        for i in range(num_channels):
            if i < len(axes):
                feature = bottleneck[0, i].cpu().numpy()
                feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
                axes[i].imshow(feature, cmap='viridis')
                axes[i].set_title(f'Channel {i+1}')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    model.train()
    print(f"Bottleneck visualization saved to '{save_path}'")

class Config:
    def __init__(self, flip_rate=0.5, local_rate=0.5, nonlinear_rate=0.5, paint_rate=0.5, inpaint_rate=0.5):
        self.flip_rate = flip_rate
        self.local_rate = local_rate
        self.nonlinear_rate = nonlinear_rate
        self.paint_rate = paint_rate
        self.inpaint_rate = inpaint_rate


def check_dataloader_sizes(loader):
    total_batches = 0
    total_images = 0
    image_shape = None
    mask_shape = None

    for batch in loader:
        images, masks = batch
        total_batches += 1
        total_images += images.shape[0]

        if image_shape is None:
            image_shape = images.shape
            mask_shape = masks.shape
    print(f"Batch size: {image_shape[0] if image_shape else 'N/A'}")
    print(f"Total number of batches: {total_batches}")
    print(f"Total number of images: {total_images}")
    print(f"Input Image batch shape: {image_shape}")
    print(f"Target Prediction batch shape: {mask_shape}")
    
def review_batch(loader, title, num_images=5, figsize=(25, 15)):
    # Get a batch from the loader
    input_images, target_predictions = next(iter(loader))
    num_images = min(num_images, input_images.shape[0])
    
    # Squeeze the channel dimension of the targets if needed (e.g., [B, 1, H, W] -> [B, H, W])
    if target_predictions.dim() == 4 and target_predictions.shape[1] == 1:
        target_predictions = target_predictions.squeeze(1)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_images, 2, figsize=figsize, squeeze=False)
    
    for i in range(num_images):
        # Handle RGB input image display
        input_image = input_images[i].permute(1, 2, 0).cpu().numpy()
        input_image = (input_image + 1) / 2  # Convert from [-1, 1] to [0, 1] range
        axes[i, 0].imshow(input_image)
        axes[i, 0].set_title(f"Input Image {i+1}")
        axes[i, 0].axis('off')
        
        # Handle grayscale target display
        target_prediction = target_predictions[i].cpu().numpy()
        target_prediction = (target_prediction + 1) / 2  # Convert from [-1, 1] to [0, 1]
        axes[i, 1].imshow(target_prediction, cmap='gray')
        axes[i, 1].set_title(f"Target Prediction {i+1}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.draw() 
    plt.pause(20)
    plt.close()
