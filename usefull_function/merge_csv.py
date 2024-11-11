import pandas as pd
import os
from glob import glob

def merge_csv_files(input_folder, output_folder='dataset', output_filename='merged_data.csv'):
    """
    Merge multiple CSV files from a folder into a single CSV file.
    
    Parameters:
    -----------
    input_folder : str
        Path to the folder containing CSV files
    output_folder : str
        Folder where the merged file will be saved
    output_filename : str
        Name of the merged output file
    
    Returns:
    --------
    str : Path to the merged file
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    # Get list of all CSV files in the input folder
    csv_files = glob(os.path.join(input_folder, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"- {os.path.basename(file)}")
    
    # Initialize an empty list to store dataframes
    dfs = []
    
    # Read each CSV file
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded {os.path.basename(file)}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {os.path.basename(file)}: {e}")
    
    if not dfs:
        print("No data frames were successfully loaded")
        return None
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged dataframe
    output_path = os.path.join(output_folder, output_filename)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\nMerge complete!")
    print(f"Total records: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Merged file saved as: {output_path}")
    
    return output_path

def check_duplicates(csv_path, output_folder='dataset', remove_duplicates=False):
    """
    Check for duplicate rows in a CSV file and optionally remove them.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file to check
    output_folder : str
        Folder where to save the deduplicated file (if remove_duplicates=True)
    remove_duplicates : bool
        Whether to remove duplicates and save a new file
    
    Returns:
    --------
    tuple : (total_rows, duplicate_count, output_path)
    """
    print(f"\nChecking duplicates in: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    # Find duplicates
    duplicates = df[df.duplicated()]
    duplicate_count = len(duplicates)
    
    # Print duplicate statistics
    print(f"\nDuplicate Analysis:")
    print(f"Total rows: {total_rows}")
    print(f"Duplicate rows: {duplicate_count}")
    print(f"Duplicate percentage: {(duplicate_count/total_rows*100):.2f}%")
    
    output_path = None
    
    # Remove duplicates if requested
    if remove_duplicates and duplicate_count > 0:
        # Create cleaned filename
        filename = os.path.basename(csv_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{base}_no_duplicates{ext}")
        
        # Remove duplicates and save
        df_clean = df.drop_duplicates()
        df_clean.to_csv(output_path, index=False)
        
        print(f"\nDuplicates removed:")
        print(f"Original rows: {total_rows}")
        print(f"Rows after deduplication: {len(df_clean)}")
        print(f"Deduplicated file saved as: {output_path}")
    
    return total_rows, duplicate_count, output_path

def main():
    """
    Main function to demonstrate usage of both functions.
    """
    # Step 1: Merge CSV files
    merged_file = merge_csv_files(
        input_folder='raw_data',
        output_folder='dataset',
        output_filename='DUMMY_DATA_hypertension_patient.csv'
    )
    
    if merged_file:
        # Step 2: Check for duplicates and remove them
        total, duplicates, cleaned_file = check_duplicates(
            csv_path=merged_file,
            output_folder='dataset',
            remove_duplicates=True
        )

if __name__ == "__main__":
    main()