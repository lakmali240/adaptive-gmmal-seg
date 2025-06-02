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
import torch.serialization
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
    shuffle_train=True
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
        shuffle=shuffle_train,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
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


def data_loader_for_fully_supervised_learning(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    test_img_dir,
    test_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
    config=None,
    shuffle_train=True
):
    """
    Create data loaders for fully supervised learning with image and mask pairs.
    
    Args:
        train_img_dir: Directory containing training images
        train_mask_dir: Directory containing training masks
        val_img_dir: Directory containing validation images
        val_mask_dir: Directory containing validation masks
        batch_size: Batch size for data loaders
        train_transform: Albumentations transforms for training images
        val_transform: Albumentations transforms for validation images
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        config: Configuration for augmentation parameters
        shuffle_train: Whether to shuffle training data
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    import os
    import random
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torchvision import transforms
    from scipy.special import comb
    
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

    class SupervisedTrainDataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None, config=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform = transform
            self.config = config if config else Config()
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".jpg", ".jpeg"))])
            
            # Ensure mask files exist for each image
            self.masks = []
            for img_file in self.images:
                # Get base filename without extension
                base_name = os.path.splitext(img_file)[0]
                # Create mask filename with _segmentation suffix and .png extension
                mask_file = base_name + "_segmentation.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_file} not found for image {img_file}")
                    continue
                
                self.masks.append(mask_file)
            
            # Update images list to only include those with matching masks
            self.images = self.images[:len(self.masks)]
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            self.mask_list = [os.path.join(mask_dir, mask) for mask in self.masks]
            
            # Extract resize transform for masks
            resize_only = [t for t in transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])
            
            if len(self.images) == 0:
                raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
                    
            print(f"Found {len(self.images)} training image-mask pairs")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load image and mask
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load as grayscale
                            
            # Apply full transforms to image
            if self.transform:
                # Convert to numpy arrays for albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"] / 255.0  # Already tensor from ToTensorV2
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Apply only resize to mask
            mask_np = np.array(mask)
            transformed_mask = self.resize_only_transform(image=mask_np)
            mask_tensor = transformed_mask["image"] / 255.0
            
            # Apply augmentation to image only
            augmented_tensor = self.augment_image(image_tensor.clone())
            
            # Ensure mask is binary (0 or 1)
            mask_tensor = (mask_tensor > 0.5).float()
            
            # Normalize image to [-1, 1] range
            augmented_tensor = normalize_minus1_to_1(augmented_tensor)
            
            return augmented_tensor, mask_tensor
        
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

    class SupervisedValDataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform = transform
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".jpg", ".jpeg"))])
            
            # Ensure mask files exist for each image
            self.masks = []
            for img_file in self.images:
                # Get base filename without extension
                base_name = os.path.splitext(img_file)[0]
                # Create mask filename with _segmentation suffix and .png extension
                mask_file = base_name + "_segmentation.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_file} not found for image {img_file}")
                    continue
                
                self.masks.append(mask_file)
            
            # Update images list to only include those with matching masks
            self.images = self.images[:len(self.masks)]
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            self.mask_list = [os.path.join(mask_dir, mask) for mask in self.masks]
            
            # Extract resize transform for masks
            resize_only = [t for t in transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])
            
            if len(self.images) == 0:
                raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
                    
            print(f"Found {len(self.images)} validation image-mask pairs")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load image and mask
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load as grayscale
                            
            # Apply full transforms to image
            if self.transform:
                # Convert to numpy arrays for albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"] / 255.0  # Already tensor from ToTensorV2
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Apply only resize to mask
            mask_np = np.array(mask)
            transformed_mask = self.resize_only_transform(image=mask_np)
            mask_tensor = transformed_mask["image"] / 255.0
            
            # Ensure mask is binary (0 or 1)
            mask_tensor = (mask_tensor > 0.5).float()
            
            # Normalize image to [-1, 1] range
            image_tensor = normalize_minus1_to_1(image_tensor)
            
            return image_tensor, mask_tensor
    
    class SupervisedTestDataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform = transform
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".jpg", ".jpeg"))])
            
            # Ensure mask files exist for each image
            self.masks = []
            for img_file in self.images:
                # Get base filename without extension
                base_name = os.path.splitext(img_file)[0]
                # Create mask filename with _segmentation suffix and .png extension
                mask_file = base_name + "_segmentation.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_file} not found for image {img_file}")
                    continue
                
                self.masks.append(mask_file)
            
            # Update images list to only include those with matching masks
            self.images = self.images[:len(self.masks)]
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            self.mask_list = [os.path.join(mask_dir, mask) for mask in self.masks]
            
            # Extract resize transform for masks
            resize_only = [t for t in transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])
            
            if len(self.images) == 0:
                raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
                    
            print(f"Found {len(self.images)} test image-mask pairs")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load image and mask
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load as grayscale
                            
            # Apply full transforms to image
            if self.transform:
                # Convert to numpy arrays for albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"] / 255.0  # Already tensor from ToTensorV2
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Apply only resize to mask
            mask_np = np.array(mask)
            transformed_mask = self.resize_only_transform(image=mask_np)
            mask_tensor = transformed_mask["image"] / 255.0
            
            # Ensure mask is binary (0 or 1)
            mask_tensor = (mask_tensor > 0.5).float()
            
            # Normalize image to [-1, 1] range
            image_tensor = normalize_minus1_to_1(image_tensor)
            
            return image_tensor, mask_tensor
        
    # Initialize datasets
    train_ds = SupervisedTrainDataset(train_img_dir, train_mask_dir, transform=train_transform, config=config)
    val_ds = SupervisedValDataset(val_img_dir, val_mask_dir, transform=val_transform)
    test_ds = SupervisedTestDataset(test_img_dir, test_mask_dir, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
    )

    return train_loader, val_loader, test_loader



def data_loader_for_self_supervised_assisted_active_learning(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    test_img_dir,
    test_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
    config=None,
    shuffle_train=True,
    ranked_clusters_file=None,
    prev_selected_cluster_file=None,  # New parameter for previously selected samples
    top_n_samples=300,  # Parameter to control how many top samples to select
    image_ranking_ascending=True # Paratemter to rankg the images based on the cluster probability
    
):
    """
    Create data loaders for self-supervised assisted active learning with image files
    and optional cluster ranking information.
    
    Args:
        train_img_dir: Directory containing training images
        train_mask_dir: Directory containing training masks
        val_img_dir: Directory containing validation images
        val_mask_dir: Directory containing validation masks
        test_img_dir: Directory containing test images
        test_mask_dir: Directory containing test masks
        batch_size: Batch size for data loaders
        train_transform: Albumentations transforms for training images
        val_transform: Albumentations transforms for validation images
        test_transform: Albumentations transforms for test images
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        config: Configuration for augmentation parameters
        shuffle_train: Whether to shuffle training data
        ranked_clusters_file: Path to the ranked_cluster_assignments.csv file
        prev_selected_cluster_file: Path to previously selected clusters file
        top_n_samples: Number of top-ranked samples to include
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, updated_ranked_clusters_file, selected_clusters_file)
    """
    import os
    import random
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torchvision import transforms
    from scipy.special import comb
    import pandas as pd
    from collections import Counter
    
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

    class SelfSupervisedAssistActiveLearnTrainDataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None, config=None, ranked_clusters_file=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform = transform
            self.config = config if config else Config()
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".jpg", ".jpeg"))])
            
            # Ensure mask files exist for each image
            self.masks = []
            for img_file in self.images:
                # Get base filename without extension
                base_name = os.path.splitext(img_file)[0]
                # Create mask filename with _segmentation suffix and .png extension
                mask_file = base_name + "_segmentation.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_file} not found for image {img_file}")
                    continue
                
                self.masks.append(mask_file)
            
            # Update images list to only include those with matching masks
            self.images = self.images[:len(self.masks)]
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            self.mask_list = [os.path.join(mask_dir, mask) for mask in self.masks]
            
            # Load cluster assignments and rankings if provided
            self.clusters = {}
            self.ranks = {}
            self.likelihoods = {}
            self.filename_to_idx = {filename: idx for idx, filename in enumerate(self.images)}
            
            # Create a map from index to filename
            self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.images)}
            
            if ranked_clusters_file and os.path.exists(ranked_clusters_file):
                try:
                    self.clusters_df = pd.read_csv(ranked_clusters_file)
                    # Create dictionaries mapping filenames to cluster, rank, and likelihood
                    for _, row in self.clusters_df.iterrows():
                        filename = row['filename']
                        if filename in self.filename_to_idx:  # Only consider files that exist in the directory
                            self.clusters[filename] = int(row['cluster'])
                            self.ranks[filename] = int(row['rank'])
                            self.likelihoods[filename] = float(row['likelihood'])
                    print(f"Loaded cluster assignments for {len(self.clusters)} images")
                except Exception as e:
                    print(f"Error loading cluster assignments: {e}")
            
            # Extract resize transform for masks
            resize_only = [t for t in transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])
            
            if len(self.images) == 0:
                raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
                    
            print(f"Found {len(self.images)} training image-mask pairs")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load image and mask
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load as grayscale            
                                    
            # Apply full transforms to image
            if self.transform:
                # Convert to numpy arrays for albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"] / 255.0  # Already tensor from ToTensorV2
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Apply only resize to mask
            mask_np = np.array(mask)
            transformed_mask = self.resize_only_transform(image=mask_np)
            mask_tensor = transformed_mask["image"] / 255.0
            
            # Apply augmentation to image only
            augmented_tensor = self.augment_image(image_tensor.clone())
            
            # Ensure mask is binary (0 or 1)
            mask_tensor = (mask_tensor > 0.5).float()
            
            # Normalize image to [-1, 1] range
            augmented_tensor = normalize_minus1_to_1(augmented_tensor)
            
            # Return cluster info along with the image and mask
            return augmented_tensor, mask_tensor
        
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
       
    # Load the ranked clusters file and process for active learning
    updated_ranked_clusters_file = ranked_clusters_file
    newly_selected_clusters_file = None
    all_selected_clusters_file = prev_selected_cluster_file
    train_indices = None
    
    # Initialize sets for tracking selected filenames
    previously_selected_filenames = set()
    newly_selected_filenames = set()
    
    # Step 1: First load previously selected samples if the file exists
    if prev_selected_cluster_file and os.path.exists(prev_selected_cluster_file):
        try:
            prev_selected_df = pd.read_csv(prev_selected_cluster_file)
            print(f"Loaded {len(prev_selected_df)} previously selected samples")
            
            # Get the list of previously selected filenames that exist in training directory
            train_files = set([img for img in os.listdir(train_img_dir) 
                      if img.endswith((".jpg", ".jpeg"))])
            previously_selected_filenames = set(prev_selected_df['filename'].values)
            
            # Filter to include only files that exist in training directory
            previously_selected_filenames = previously_selected_filenames.intersection(train_files)
            print(f"Found {len(previously_selected_filenames)} previously selected samples in training directory")
            
        except Exception as e:
            print(f"Error loading previously selected samples: {e}")
            previously_selected_filenames = set()
    else:
        print("No previously selected samples file provided or file does not exist")
        previously_selected_filenames = set()

    # Step 2: Select additional new samples to reach top_n_samples total
    remaining_samples_needed = top_n_samples - len(previously_selected_filenames)
    print(f"Need to select {remaining_samples_needed} additional samples to reach total of {top_n_samples}")
    
    if remaining_samples_needed > 0 and ranked_clusters_file and os.path.exists(ranked_clusters_file):
        try:
            clusters_df = pd.read_csv(ranked_clusters_file)
            print(f"Loaded {len(clusters_df)} entries from ranked clusters file")
            
            # Filter to include only files that exist in training directory and are not previously selected
            train_files = set([img for img in os.listdir(train_img_dir) 
                      if img.endswith((".jpg", ".jpeg"))])
            
            # Remove previously selected samples from consideration
            clusters_df = clusters_df[
                (clusters_df['filename'].isin(train_files)) & 
                (~clusters_df['filename'].isin(previously_selected_filenames))
            ]
            
            # Get the distribution of clusters
            cluster_counts = Counter(clusters_df['cluster'])
            total_samples = len(clusters_df)
            
            if total_samples > 0:
                # Calculate how many samples to take from each cluster proportionally
                samples_per_cluster = {}
                for cluster, count in cluster_counts.items():
                    # Calculate the proportion and make sure each cluster gets at least 1 sample
                    proportion = count / total_samples
                    samples_per_cluster[cluster] = max(1, int(remaining_samples_needed * proportion))
                
                # Adjust to ensure we get exactly remaining_samples_needed
                total_allocated = sum(samples_per_cluster.values())
                if total_allocated < remaining_samples_needed:
                    # Distribute remaining samples to largest clusters
                    remaining = remaining_samples_needed - total_allocated
                    for cluster in sorted(cluster_counts, key=cluster_counts.get, reverse=True):
                        if remaining <= 0:
                            break
                        samples_per_cluster[cluster] += 1
                        remaining -= 1
                elif total_allocated > remaining_samples_needed:
                    # Remove samples from smallest clusters
                    excess = total_allocated - remaining_samples_needed
                    for cluster in sorted(cluster_counts, key=cluster_counts.get):
                        if excess <= 0:
                            break
                        if samples_per_cluster[cluster] > 1:  # Ensure at least 1 sample per cluster
                            samples_per_cluster[cluster] -= 1
                            excess -= 1
                
                # Select top-ranked samples from each cluster based on allocated count
                selected_samples = []
                for cluster, sample_count in samples_per_cluster.items():
                    # Sort cluster samples based on the ranking preference
                    if image_ranking_ascending:
                        # Sort by rank in ascending order (high probablilty (low rank) values first)
                        # print("Sort by rank in ascending order (high probablilty (low rank) values first)")
                        cluster_samples = clusters_df[clusters_df['cluster'] == cluster].sort_values('rank')
                    elif image_ranking_ascending is False:  
                        # Sort by rank in desceding order (low probablilty (high rank) values first)
                        print("Sort by rank in desceding order (low probablilty (high rank) values first)")
                        cluster_samples = clusters_df[clusters_df['cluster'] == cluster].sort_values('rank', ascending=False)
                    else:
                        # Handle unexpected value
                        raise ValueError("Invalid ordering request. 'image_ranking_ascending' must be True or False")
                    # Take the top n samples
                    selected_cluster_samples = cluster_samples.head(sample_count)
                    selected_samples.append(selected_cluster_samples)
                
                # Combine all selected samples
                newly_selected_df = pd.concat(selected_samples) if selected_samples else pd.DataFrame()
                
                # Get the list of newly selected filenames
                newly_selected_filenames = set(newly_selected_df['filename'].tolist())
                print(f"Selected {len(newly_selected_filenames)} new samples")
            else:
                print("No additional samples available for selection after removing previously selected samples")
                newly_selected_filenames = set()
            
        except Exception as e:
            print(f"Error processing ranked clusters file: {e}")
            newly_selected_filenames = set()
    else:
        print("No ranked clusters file provided, file does not exist, or no additional samples needed")
        newly_selected_filenames = set()
    
    # Combine previously selected and newly selected filenames
    # all_selected_filenames = list(previously_selected_filenames.union(newly_selected_filenames)) # all_selected_filenames is not sorted
    all_selected_filenames = sorted(previously_selected_filenames.union(newly_selected_filenames)) # all_selected_filenames is sorted
    
    print(f"Total selected samples: {len(all_selected_filenames)} (Previous: {len(previously_selected_filenames)}, New: {len(newly_selected_filenames)})")
    
    # Create a new DataFrame with all selected samples
    if ranked_clusters_file and os.path.exists(ranked_clusters_file):
        try:
            # Read the original clusters file to get full information
            all_clusters_df = pd.read_csv(ranked_clusters_file)
            
            # Filter to only include selected filenames
            selected_df = all_clusters_df[all_clusters_df['filename'].isin(all_selected_filenames)]
            
            # Save selected samples to a new file
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            # newly_selected_clusters_file = ranked_clusters_file.replace('.csv', f'_newly_selected_{timestamp}.csv')
            newly_selected_clusters_file = f'results/gmm_results/dynamic_features/ranked_cluster_assignments_newly_selected_{timestamp}.csv'
            selected_df.to_csv(newly_selected_clusters_file, index=False)
            print(f"Newly selected clusters information saved to {newly_selected_clusters_file}")
            
            # Create updated clusters file by removing all selected samples
            updated_clusters_df = all_clusters_df[~all_clusters_df['filename'].isin(all_selected_filenames)]
            
            # Save updated clusters file to a new path
            # updated_ranked_clusters_file = ranked_clusters_file.replace('.csv', f'_updated_{timestamp}.csv')
            updated_ranked_clusters_file = f'results/gmm_results/dynamic_features/ranked_cluster_assignments_updated_{timestamp}.csv'
            updated_clusters_df.to_csv(updated_ranked_clusters_file, index=False)
            print(f"Updated ranked clusters file saved to {updated_ranked_clusters_file}")

            # Save all selected samples to a new path
            # all_selected_clusters_file = ranked_clusters_file.replace('.csv', f'_all_selected_{timestamp}.csv')
            all_selected_clusters_file = f'results/gmm_results/dynamic_features/ranked_cluster_assignments_all_selected_{timestamp}.csv'
            all_selected_df = pd.DataFrame({'filename': all_selected_filenames})
            all_selected_df.to_csv(all_selected_clusters_file, index=False)
            print(f"All selected clusters information saved to {all_selected_clusters_file}")
            
            # Create a subset of indices to include in training
            train_files_list = sorted([img for img in os.listdir(train_img_dir) 
                        if img.endswith((".jpg", ".jpeg"))])
            train_indices = [i for i, filename in enumerate(train_files_list) if filename in all_selected_filenames]
            
        except Exception as e:
            print(f"Error creating selected, updated, all files files: {e}")
            newly_selected_clusters_file = None
            all_selected_clusters_file = prev_selected_cluster_file
            updated_ranked_clusters_file = ranked_clusters_file
            train_indices = None
    else:
        print("No ranked clusters file provided or file does not exist")
        newly_selected_clusters_file = None
        all_selected_clusters_file = prev_selected_cluster_file
        updated_ranked_clusters_file = ranked_clusters_file
        train_indices = None

    # Initialize datasets
    train_ds = SelfSupervisedAssistActiveLearnTrainDataset(
        train_img_dir, 
        train_mask_dir,
        transform=train_transform, 
        config=config, 
        ranked_clusters_file=ranked_clusters_file
    )
    
    class SupervisedValDataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform = transform
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".jpg", ".jpeg"))])
            
            # Ensure mask files exist for each image
            self.masks = []
            for img_file in self.images:
                # Get base filename without extension
                base_name = os.path.splitext(img_file)[0]
                # Create mask filename with _segmentation suffix and .png extension
                mask_file = base_name + "_segmentation.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_file} not found for image {img_file}")
                    continue
                
                self.masks.append(mask_file)
            
            # Update images list to only include those with matching masks
            self.images = self.images[:len(self.masks)]
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            self.mask_list = [os.path.join(mask_dir, mask) for mask in self.masks]
            
            # Extract resize transform for masks
            resize_only = [t for t in transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])
            
            if len(self.images) == 0:
                raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
                    
            print(f"Found {len(self.images)} validation image-mask pairs")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load image and mask
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load as grayscale
                            
            # Apply full transforms to image
            if self.transform:
                # Convert to numpy arrays for albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"] / 255.0  # Already tensor from ToTensorV2
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Apply only resize to mask
            mask_np = np.array(mask)
            transformed_mask = self.resize_only_transform(image=mask_np)
            mask_tensor = transformed_mask["image"] / 255.0
            
            # Ensure mask is binary (0 or 1)
            mask_tensor = (mask_tensor > 0.5).float()
            
            # Normalize image to [-1, 1] range
            image_tensor = normalize_minus1_to_1(image_tensor)
            
            return image_tensor, mask_tensor
    
    class SupervisedTestDataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform = transform
            
            # Get all image files from directory in alphabetically sorted order
            self.images = sorted([img for img in os.listdir(image_dir) 
                        if img.endswith((".jpg", ".jpeg"))])
            
            # Ensure mask files exist for each image
            self.masks = []
            for img_file in self.images:
                # Get base filename without extension
                base_name = os.path.splitext(img_file)[0]
                # Create mask filename with _segmentation suffix and .png extension
                mask_file = base_name + "_segmentation.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_file} not found for image {img_file}")
                    continue
                
                self.masks.append(mask_file)
            
            # Update images list to only include those with matching masks
            self.images = self.images[:len(self.masks)]
            
            # Create img_list attribute with full paths in the same sorted order
            self.img_list = [os.path.join(image_dir, img) for img in self.images]
            self.mask_list = [os.path.join(mask_dir, mask) for mask in self.masks]
            
            # Extract resize transform for masks
            resize_only = [t for t in transform if isinstance(t, A.Resize)][0]
            self.resize_only_transform = A.Compose([
                resize_only,  # This maintains the original IMAGE_HEIGHT and IMAGE_WIDTH
                ToTensorV2(),
            ])
            
            if len(self.images) == 0:
                raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
                    
            print(f"Found {len(self.images)} test image-mask pairs")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Load image and mask
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load as grayscale
                            
            # Apply full transforms to image
            if self.transform:
                # Convert to numpy arrays for albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"] / 255.0  # Already tensor from ToTensorV2
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Apply only resize to mask
            mask_np = np.array(mask)
            transformed_mask = self.resize_only_transform(image=mask_np)
            mask_tensor = transformed_mask["image"] / 255.0
            
            # Ensure mask is binary (0 or 1)
            mask_tensor = (mask_tensor > 0.5).float()
            
            # Normalize image to [-1, 1] range
            image_tensor = normalize_minus1_to_1(image_tensor)
            
            return image_tensor, mask_tensor
    
    val_ds = SupervisedValDataset(val_img_dir, val_mask_dir, transform=val_transform)
    test_ds = SupervisedTestDataset(test_img_dir, test_mask_dir, transform=test_transform)
    
    # Create a subset of the training dataset with only the selected files if we have indices
    if train_indices is not None and len(train_indices) > 0:
        train_ds = Subset(train_ds, train_indices)
        print(f"Using subset of {len(train_indices)} training samples instead of {len(os.listdir(train_img_dir))}")

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
    )
    
    # Keep val_loader and test_loader unchanged from the original function
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False,  # Drop the last incomplete batch to avoid shape issues
    )

    return train_loader, val_loader, test_loader, updated_ranked_clusters_file, newly_selected_clusters_file, all_selected_clusters_file

# =================================================
#             Save predictions as images
# =================================================
def save_ssl_predictions_as_imgs(
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

def save_fss_predictions_images(
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
        x = x.to(device=device) # [-1, 1]
        y = y.to(device=device) # {0,1}

        with torch.no_grad():
            # Handle the dual output of UNET model
            preds, _ = model(x)
            preds = torch.sigmoid(preds) # Convert logits to probabilities [0, 1]
            preds = (preds > 0.5).float() # Convert to binary {0, 1} for metrics/visualization
        
        # Normalize this from [-1, 1] to [0, 1] for visualization
        normalized_x = (x + 1) / 2       # [0,1]
        
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


def save_ssaal_predictions_images(
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
        x = x.to(device=device) # [-1, 1]
        y = y.to(device=device) # {0,1}

        with torch.no_grad():
            # Handle the dual output of UNET model
            preds, _ = model(x)
            preds = torch.sigmoid(preds) # Convert logits to probabilities [0, 1]
            preds = (preds > 0.5).float() # Convert to binary {0, 1} for metrics/visualization
        
        # Normalize this from [-1, 1] to [0, 1] for visualization
        normalized_x = (x + 1) / 2       # [0,1]
        
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

def save_ssaal_test_images(
    loader, model, folder="saved_images/", device="cuda", iteration=None
):
    """
    Save the input images, predictions, and targets as a single combined image with labels.
    If iteration is provided, also save the three images separately with iteration in the filename.
    
    Args:
        loader: DataLoader containing the images to make predictions on
        model: The UNET model that returns both segmentation and bottleneck features
        folder: Folder to save the images to
        device: Device to use for predictions
        iteration: Optional iteration number to include in the filenames of separate images
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    model.eval()
    for idx, (x, y) in enumerate(loader):        
        x = x.to(device=device) # [-1, 1]
        y = y.to(device=device) # {0,1}

        with torch.no_grad():
            # Handle the dual output of UNET model
            preds, _ = model(x)
            preds = torch.sigmoid(preds) # Convert logits to probabilities [0, 1]
            preds = (preds > 0.5).float() # Convert to binary {0, 1} for metrics/visualization
        
        # Normalize this from [-1, 1] to [0, 1] for visualization
        normalized_x = (x + 1) / 2       # [0,1]
        
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
            
            # If iteration is provided, save individual images with iteration in filename
            if iteration is not None:
                # Save input image
                torchvision.utils.save_image(
                    input_img, f"{folder}/batch{idx}_img{b}_input_{iteration}.png"
                )
                
                # Save prediction image
                torchvision.utils.save_image(
                    pred_img, f"{folder}/batch{idx}_img{b}_prediction_{iteration}.png"
                )
                
                # Save target image
                torchvision.utils.save_image(
                    target_img, f"{folder}/batch{idx}_img{b}_target_{iteration}.png"
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
    # Add the classes you need to safely load    
    torch.serialization.add_safe_globals({"torch.optim.adam.Adam": torch.optim.Adam})
    torch.serialization.add_safe_globals({"torch.optim.lr_scheduler.ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau})
    # Add more classes as needed if you use other optimizer or scheduler types
    
    print("\n+++ Model Load starts")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return model, optimizer, scheduler, 1

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu') # weights_only=True
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded. Trained for {epoch} epochs.")
    print_current_lr(scheduler)
    print("+++ Model Load ends successfully\n")
    return model, optimizer, scheduler, epoch


def load_trained_model_without_decoder(model, optimizer, scheduler, model_path):
    """
    Load a trained model from a checkpoint, but only load encoder weights.
    
    Args:
        model: Model to load the weights into
        optimizer: Optimizer to load the state into
        scheduler: Scheduler to load the state into
        model_path: Path to the checkpoint file
        
    Returns:
        tuple: Updated model, optimizer, scheduler, and epoch
    """
    # Add the classes you need to safely load
    torch.serialization.add_safe_globals({"torch.optim.adam.Adam": torch.optim.Adam})
    torch.serialization.add_safe_globals({"torch.optim.lr_scheduler.ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau})
    
    print("\n+++ Model Load starts")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return model, optimizer, scheduler, 1
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load model weights but exclude decoder parts
    pretext_model = checkpoint['state_dict']
    
    # Get the current model state
    model_dict = model.state_dict()
    
    # Filter to keep only encoder parameters
    encoder_keys = []
    for key in model_dict.keys():
        if 'downs.' in key or 'bottleneck.' in key:
            encoder_keys.append(key)
    
    # Create a filtered state dict with only encoder parameters
    filtered_state_dict = {k: pretext_model[k] for k in encoder_keys if k in pretext_model}
    
    # Update the model dictionary with the filtered weights
    model_dict.update(filtered_state_dict)
    
    # Load the updated dictionary into the model
    model.load_state_dict(model_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded with encoder weights only. Original model was trained for {epoch} epochs.")
    print_current_lr(scheduler)
    print("+++ Model Load ends successfully\n")
    return model, optimizer, scheduler, epoch


def load_trained_model_without_output_layer(model, optimizer, scheduler, model_path):
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
    # Add the classes you need to safely load    
    torch.serialization.add_safe_globals({"torch.optim.adam.Adam": torch.optim.Adam})
    torch.serialization.add_safe_globals({"torch.optim.lr_scheduler.ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau})
    # Add more classes as needed if you use other optimizer or scheduler types
    
    print("\n+++ Model Load starts")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return model, optimizer, scheduler, 1

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu') # weights_only=True
    
    # Load model weights but exclude output layer
    pretext_model = checkpoint['state_dict']
    # Filter out final_conv parameters
    state_dict = {k:v for k,v in pretext_model.items() if 'final_conv' not in k}
    # Get the current model state
    model_dict = model.state_dict()
    # Update the model dictionary with the filtered weights
    model_dict.update(state_dict)
    # Load the updated dictionary into the model
    model.load_state_dict(model_dict)


    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded. Trained for {epoch} epochs.")
    print_current_lr(scheduler)
    print("+++ Model Load ends successfully\n")
    return model, optimizer, scheduler, epoch

def load_trained_model_without_scheduler(model, model_path):
    """
    Load a trained model from a checkpoint.
    
    Args:
        model: Model to load the weights into
        model_path: Path to the checkpoint file
        
    Returns:
        tuple: Updated model, optimizer, scheduler, and epoch
    """
    # Add the classes you need to safely load    
    torch.serialization.add_safe_globals({"torch.optim.adam.Adam": torch.optim.Adam})
    torch.serialization.add_safe_globals({"torch.optim.lr_scheduler.ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau})
    # Add more classes as needed if you use other optimizer or scheduler types
    
    print("\n+++ Model Load starts")
    print("\n+++ Model Load starts")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return model, 1

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu') # weights_only=True
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']

    print("+++ Model Load ends successfully\n")
    return model, epoch

def load_trained_model_without_scheduler_and_output_layer(model, model_path):
    """
    Load pre-trained model weights but exclude the output layer.
    
    Args:
        model: Model to load the weights into
        model_path: Path to the checkpoint file
        
    Returns:
        tuple: Updated model and start epoch
    """
    print("\n+++ Loading pre-trained model without output layer")
    
    if model_path is None or not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return model, 1

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get the current model state
    model_dict = model.state_dict()
    
    # Get the weights from the checkpoint but exclude output conv layer
    pretext_model = checkpoint['state_dict']
    state_dict = {k:v for k,v in pretext_model.items() if 'final_conv' not in k}
    
    # Update the model dictionary with the filtered weights
    model_dict.update(state_dict)
    
    # Load the updated dictionary into the model
    model.load_state_dict(model_dict)
    epoch = checkpoint['epoch']
    
    print(f"Model loaded with all weights except output layer. Original model was trained for {checkpoint['epoch']} epochs.")
    print("+++ Model load completed successfully\n")
    
    return model, epoch  # Return start epoch as 1 to train the new output layer from scratch

# def load_trained_model(model, optimizer, scheduler, model_path):
#     """
#     Load a trained model from a checkpoint.
    
#     Args:
#         model: Model to load the weights into
#         optimizer: Optimizer to load the state into
#         scheduler: Scheduler to load the state into
#         model_path: Path to the checkpoint file
        
#     Returns:
#         tuple: Updated model, optimizer, scheduler, and epoch
#     """
#     print("\n+++ Model Load starts")
#     if not os.path.exists(model_path):
#         print(f"No model found at {model_path}")
#         return model, optimizer, scheduler, 0

#     print(f"Loading model from {model_path}")
#     checkpoint = torch.load(model_path, map_location='cpu') # weights_only=True
    
#     # Load model weights
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
    
#     if 'scheduler_state_dict' in checkpoint:
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
#     print(f"Model loaded. Trained for {epoch} epochs.")
#     print_current_lr(scheduler)
#     print("+++ Model Load ends successfully\n")
#     return model, optimizer, scheduler, epoch

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
    plt.pause(5)
    plt.close()
