import csv
from collections import defaultdict

def replace_likelihood_with_entropy_and_preserve_order(first_csv_file, second_csv_file, output_csv_file, ascending_order=True):
    """
    Replace 'likelihood' in the first CSV with 'entropy_equation3' from the second CSV where filenames match,
    and assign new ranks within each cluster based on the updated likelihood values,
    while preserving the original filename order.
    
    Args:
        first_csv_file (str): Path to the first CSV file (ranked_cluster_assignments.csv)
        second_csv_file (str): Path to the second CSV file (entropy_results.csv)
        output_csv_file (str): Path to the output CSV file
        ascending_order (bool): If True (default), low entropy gets rank 1 (ascending).
                               If False, high entropy gets rank 1 (descending).
    """
    # Step 1: Load entropy values from second CSV
    entropy_dict = {}
    with open(second_csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entropy_dict[row['image_name']] = row['entropy_equation3']
    
    print(f"Loaded entropy values for {len(entropy_dict)} images")
    
    # Step 2: Load and update first CSV rows
    updated_rows = []
    matched_count = 0
    with open(first_csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            filename = row['filename']
            if filename in entropy_dict:
                row['likelihood'] = entropy_dict[filename]
                matched_count += 1
            updated_rows.append(row)
    
    print(f"Updated likelihood values for {matched_count} matching images")
    
    # Step 3: Group rows by cluster (for re-ranking)
    cluster_to_rows = defaultdict(list)
    for idx, row in enumerate(updated_rows):
        cluster_to_rows[row['cluster']].append((idx, row))  # Keep original index for mapping
    
    print(f"Processing {len(cluster_to_rows)} clusters")
    
    # Step 4: Compute rank in each cluster based on updated likelihood (entropy)
    index_to_new_rank = {}
    
    # Determine sorting order based on parameter
    if ascending_order:
        print("Using ascending order: LOW entropy → rank 1 (most confident/least uncertain)")
        reverse_sort = False
    else:
        print("Using descending order: HIGH entropy → rank 1 (most uncertain)")
        reverse_sort = True
    
    for cluster, row_list in cluster_to_rows.items():
        # Sort by float likelihood (entropy) - ascending or descending based on parameter
        sorted_by_entropy = sorted(row_list, key=lambda x: float(x[1]['likelihood']), reverse=reverse_sort)
        
        # Debug: Show ranking for first cluster
        if cluster == list(cluster_to_rows.keys())[0]:  # First cluster
            # print(f"\nExample ranking for cluster {cluster}:")
            for rank, (original_idx, row) in enumerate(sorted_by_entropy[:5], start=1):  # Show first 5
                entropy_val = float(row['likelihood'])
                # print(f"  Rank {rank}: {row['filename']} (entropy: {entropy_val:.4f})")
            if len(sorted_by_entropy) > 5:
                # print(f"  ... and {len(sorted_by_entropy) - 5} more images")
                pass
        
        # Assign ranks
        for rank, (original_idx, _) in enumerate(sorted_by_entropy, start=1):
            index_to_new_rank[original_idx] = str(rank)
    
    # Step 5: Update the rank field without changing order
    for idx, row in enumerate(updated_rows):
        if idx in index_to_new_rank:
            row['rank'] = index_to_new_rank[idx]
    
    # Step 6: Write the final output with preserved order
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    # Summary statistics
    # print(f"\nSummary:")
    # print(f"- Total images processed: {len(updated_rows)}")
    # print(f"- Images with updated entropy: {matched_count}")
    # print(f"- Number of clusters: {len(cluster_to_rows)}")
    # print(f"- Ranking order: {'Ascending (low entropy = rank 1)' if ascending_order else 'Descending (high entropy = rank 1)'}")
    print(f"- Updated CSV with preserved order and re-ranked clusters saved to {output_csv_file}")

def main():
    """
    Main function with examples for different ranking orders
    """
    first_csv = "results/ranked_cluster_assignments_updated.csv"
    second_csv = "results/entropy_results.csv"
    
    # Default: Low entropy ranked first (ascending order)
    print("=== DEFAULT: Low Entropy Ranked First (Ascending) ===")
    output_csv_asc = "results/updated_likelihood_with_entropy.csv"
    output_csv_asc = first_csv
    replace_likelihood_with_entropy_and_preserve_order(
        first_csv, second_csv, output_csv_asc, ascending_order=True
    )
    

if __name__ == "__main__":
    # Example usage with different ranking orders
    first_csv = "results/ranked_cluster_assignments.csv"
    second_csv = "results/entropy_results.csv"
    
    # Use main function (generates both ascending and descending)
    main()
    
   