import os
import csv

def extract_and_save_filenames(input_csv_path, output_csv_path):
    """
    Extract filenames from the input CSV file and save them to a new CSV 
    with only 'filename' as the header.
    
    Parameters:
    input_csv_path (str): Path to the input CSV file
    output_csv_path (str): Path where the output CSV file will be saved
    
    Returns:
    bool: True if successful, False otherwise
    """
    filenames = []
    
    try:
        # Read filenames from input CSV
        with open(input_csv_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                if 'filename' in row:
                    filenames.append(row['filename'])
                else:
                    print("Warning: 'filename' column not found in CSV file.")
                    return False
        
        # Write filenames to output CSV
        with open(output_csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            
            # Write header
            csv_writer.writerow(['filename'])
            
            # Write filenames
            for filename in filenames:
                csv_writer.writerow([filename])
        
        print(f"Successfully saved {len(filenames)} filenames to {output_csv_path}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_csv_path}' not found.")
        return False
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return False

def extract_matching_records(first_csv_file, second_csv_file, output_csv_file):
    """
    Extract records from first CSV that match filenames in second CSV
    
    Parameters:
    first_csv_file (str): Path to first CSV with filename,cluster,likelihood,rank data
    second_csv_file (str): Path to second CSV with filenames to extract
    output_csv_file (str): Path to output CSV file
    
    Returns:
    int: Number of matches found
    """
    # Read all records from first CSV into a dictionary with filename as key
    first_csv_data = {}
    with open(first_csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            first_csv_data[row['filename']] = row
    
    # Read filenames from second CSV
    filenames_to_extract = []
    with open(second_csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames_to_extract.append(row['filename'])
    
    # Extract matching records and write to output file
    matches_found = 0
    with open(output_csv_file, 'w', newline='') as f:
        # Use the same fieldnames as the first CSV
        fieldnames = ['filename', 'cluster', 'likelihood', 'rank']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for filename in filenames_to_extract:
            if filename in first_csv_data:
                writer.writerow(first_csv_data[filename])
                matches_found += 1
    
    return matches_found

def create_new_filename_1(original_filename, suffix="_filenames"):
    """
    Insert a suffix before the file extension
    
    Parameters:
    original_filename (str): Original filename
    suffix (str): Suffix to insert before extension (default: "_filenames")
    
    Returns:
    str: Modified filename
    """
    # Split the filename into base and extension
    base, extension = original_filename.rsplit('.', 1)
    
    # Create the new filename with the suffix
    new_filename = f"{base}{suffix}.{extension}"
    
    return new_filename



# Example usage
if __name__ == "__main__":
    """ def extract_and_save_filenames """
    # Replace with your actual file paths
    input_csv = "results/ssaal_trained_model/2025-05-05_04-41-17/gmm_results/ranked_cluster_assignments_updated_20250505_044118.csv"
    output_csv = "results/ssaal_trained_model/2025-05-05_04-41-17/gmm_results/ranked_cluster_assignments_updated_20250505_044118_filename.csv"
    
    success = extract_and_save_filenames(input_csv, output_csv)
    
    if success:
        pass
        # print("Operation completed successfully.")
    else:
        print("Operation failed.")

    """ def extract_matching_records """
    matches = extract_matching_records(
        'results/ssaal_trained_model/2025-05-05_04-41-17/gmm_results/ranked_cluster_assignments_updated_20250505_044118.csv',  # Path to first CSV file
        'results/ssaal_trained_model/2025-05-05_04-41-17/gmm_results/ranked_cluster_assignments_updated_20250505_044118_extracted.csv',  # Path to second CSV file
        'results/ssaal_trained_model/2025-05-05_04-41-17/gmm_results/ranked_cluster_assignments_updated_20250505_044118_common.csv'  # Path to output CSV file
    )
    print(f"Found {matches} matching records.")


    """ create_new_filename 1"""
    original = os.path.join("results/gmm_results/2025-04-22_03-38-29", "ranked_cluster_assignments_updated.csv")
    new_filename = create_new_filename_1(original)
    print(new_filename)
