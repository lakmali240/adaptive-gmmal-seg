import pandas as pd

def count_images_in_csv(csv_file_path):
    """
    Count the number of image filenames in a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        int: Number of image filenames in the CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if 'filename' column exists
        if 'filename' not in df.columns:
            print(f"Error: 'filename' column not found in {csv_file_path}")
            return 0
        
        # Count rows where the filename ends with '.jpg'
        image_count = df[df['filename'].str.endswith('.jpg', na=False)].shape[0]
        
        # Print all filenames for verification
        # print("Image filenames found:")
        # for filename in df['filename']:
        #     print(filename)
        
        return image_count
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0

def count_images_in_csv(csv_file_path):
    """
    Count the number of image filenames in a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        int: Number of image filenames in the CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if 'filename' column exists
        if 'filename' not in df.columns:
            print(f"Error: 'filename' column not found in {csv_file_path}")
            return 0
        
        # Count rows where the filename ends with '.jpg'
        image_count = df[df['filename'].str.endswith('.jpg', na=False)].shape[0]
        
        return image_count
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0

def analyze_csv_files(file1_path, file2_path):
    """
    Analyze two CSV files to count images and find common filenames.
    
    Args:
        file1_path (str): Path to the first CSV file
        file2_path (str): Path to the second CSV file
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Check if 'filename' column exists in both files
        if 'filename' not in df1.columns:
            print(f"Error: 'filename' column not found in {file1_path}")
            return
        
        if 'filename' not in df2.columns:
            print(f"Error: 'filename' column not found in {file2_path}")
            return
        
        # Filter for only .jpg files
        jpg_files1 = df1[df1['filename'].str.endswith('.jpg', na=False)]['filename'].tolist()
        jpg_files2 = df2[df2['filename'].str.endswith('.jpg', na=False)]['filename'].tolist()
        
        # Count images in each file
        count1 = len(jpg_files1)
        count2 = len(jpg_files2)
        
        # Find common images
        common_images = list(set(jpg_files1).intersection(set(jpg_files2)))
        common_count = len(common_images)
        
        # Print results
        print(f"\n===== ANALYSIS RESULTS =====")
        print(f"File 1: {file1_path}")
        print(f"  - Total images: {count1}")
        
        print(f"\nFile 2: {file2_path}")
        print(f"  - Total images: {count2}")
        
        print(f"\nCommon Images: {common_count}")
        
        # Print common images if there are any
        if common_count > 0:
            print("\nList of common images:")
            for image in sorted(common_images):
                # print(f"  - {image}")\
                pass
            
            # Calculate percentage of overlap
            overlap_percentage1 = (common_count / count1) * 100 if count1 > 0 else 0
            overlap_percentage2 = (common_count / count2) * 100 if count2 > 0 else 0
            
            print(f"\nOverlap statistics:")
            print(f"  - {overlap_percentage1:.2f}% of images in File 1 are also in File 2")
            print(f"  - {overlap_percentage2:.2f}% of images in File 2 are also in File 1")
            
    except Exception as e:
        print(f"Error analyzing CSV files: {e}")

# Part 1: Count images in a single file
file_path = 'results/ssaal_trained_model/2025-05-05_04-41-17/gmm_results/ranked_cluster_assignments_updated.csv'
image_count = count_images_in_csv(file_path)
print(f"\nTotal number of images in {file_path}: {image_count}")

print("\n----------------------------------------\n")

# Part 2: Compare two files
file1_path = file_path
file2_path = 'results/ssaal_trained_model/2025-05-05_04-41-17/gmm_results/ranked_cluster_assignments_updated_20250505_053813.csv'

analyze_csv_files(file1_path, file2_path)