import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(input_folder='dataset', input_filename='dummy_data.csv', random_state=42):
    """
    Split data into training (70%) and testing (30%) sets.
    
    Parameters:
    -----------
    input_folder : str
        Name of the folder containing the input CSV file
    input_filename : str
        Name of the input CSV file
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (train_data, test_data)
    """
    # Construct input file path
    input_path = os.path.join(input_folder, input_filename)
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_path)
        print(f"Data loaded successfully from: {input_path}")
        print(f"Total records: {len(df)}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Perform 70-30 split with stratification
    try:
        train_data, test_data = train_test_split(
            df,
            test_size=0.3,  # 30% for testing
            random_state=random_state,
            stratify=df['label_hypertension']  # Stratify based on target variable
        )
        
        # Define file paths for output
        train_path = os.path.join(input_folder, 'train_data.csv')
        test_path = os.path.join(input_folder, 'test_data.csv')
        
        # Save splits to CSV
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"\nData split summary:")
        print(f"Training set: {len(train_data)} records ({len(train_data)/len(df)*100:.1f}%)")
        print(f"Testing set:  {len(test_data)} records ({len(test_data)/len(df)*100:.1f}%)")
        
        # Calculate and print prevalence
        train_prev = float(train_data['label_hypertension'].mean() * 100)
        test_prev = float(test_data['label_hypertension'].mean() * 100)
        print(f"\nHypertension Prevalence:")
        print(f"Training set: {train_prev:.2f}%")
        print(f"Testing set:  {test_prev:.2f}%")
        
        print(f"\nFiles saved in '{input_folder}' folder:")
        print(f"- {train_path}")
        print(f"- {test_path}")
        
        return train_data, test_data
        
    except Exception as e:
        print(f"Error during data splitting: {str(e)}")
        raise  # This will show the full error traceback
        return None

def main():
    """
    Example usage of the split_data function.
    """
    try:
        # Split the data
        train_data, test_data = split_data(
            input_folder='dataset',
            input_filename='dummy_data.csv'
        )
        
        if train_data is not None and test_data is not None:
            print("Data split completed successfully!")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()