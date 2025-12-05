import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Limit joblib parallelism

import pandas as pd
import numpy as np
from customer_segmentation import CustomerSegmentation
import traceback

def verify_data_files():
    """Verify that all required data files exist and are readable"""
    required_files = [
        'customer_rfm_data.csv',
        'customer_behavior_data.csv',
        'ecommerce_behavior.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required data files: {missing_files}")
    
    return True

def load_and_combine_data():
    """Load and combine all available datasets"""
    try:
        # Verify data files
        verify_data_files()
        
        # Load individual datasets
        print("\nLoading RFM data...")
        rfm_data = pd.read_csv('customer_rfm_data.csv')
        print(f"RFM data shape: {rfm_data.shape}")
        
        print("\nLoading behavioral data...")
        behavior_data = pd.read_csv('customer_behavior_data.csv')
        print(f"Behavioral data shape: {behavior_data.shape}")
        
        print("\nLoading e-commerce data...")
        ecommerce_data = pd.read_csv('ecommerce_behavior.csv')
        print(f"E-commerce data shape: {ecommerce_data.shape}")
        
        # Convert date columns
        print("\nProcessing date columns...")
        behavior_data['InvoiceDate'] = pd.to_datetime(behavior_data['Last Purchase Date'])
        
        # Create necessary columns
        print("\nCreating derived columns...")
        rfm_data['TotalSpend'] = rfm_data['Monetary']
        rfm_data['Quantity'] = rfm_data['Frequency']
        
        # Combine datasets
        print("\nMerging datasets...")
        combined_data = pd.merge(rfm_data, behavior_data, on='CustomerID', how='inner')
        print(f"After first merge shape: {combined_data.shape}")
        
        combined_data = pd.merge(combined_data, ecommerce_data, on='CustomerID', how='inner')
        print(f"After second merge shape: {combined_data.shape}")
        
        # Calculate unit price
        combined_data['UnitPrice'] = combined_data['TotalSpend'] / combined_data['Quantity']
        
        # Save combined dataset
        print("\nSaving combined dataset...")
        combined_data.to_csv('combined_customer_data.csv', index=False)
        print(f"Combined data shape: {combined_data.shape}")
        print("Columns in combined data:", combined_data.columns.tolist())
        
        return combined_data
        
    except Exception as e:
        print(f"\nError in data loading and combination: {str(e)}")
        print("\nDetailed error traceback:")
        traceback.print_exc()
        raise

def main():
    try:
        print("=== Starting Customer Segmentation Analysis ===")
        
        # Initialize segmentation
        print("\nStep 1: Initializing customer segmentation...")
        segmentation = CustomerSegmentation()
        
        # Load data
        print("\nStep 2: Loading data...")
        segmentation.load_data()
        
        # Run complete analysis pipeline
        print("\nStep 3: Running analysis pipeline...")
        segmentation.explore_data()
        
        print("\nStep 4: Preprocessing data...")
        segmentation.preprocess_data()
        
        print("\nStep 5: Performing RFM analysis...")
        segmentation.analyze_rfm()
        
        print("\nStep 6: Creating RFM segments...")
        segmentation.create_rfm_segments()
        
        print("\nStep 7: Creating behavioral segments...")
        segmentation.create_behavioral_segments()
        
        print("\nStep 8: Visualizing results...")
        segmentation.visualize_results()
        
        print("\nStep 9: Saving results...")
        segmentation.save_results()

        print("\n=== Analysis Completed Successfully ===")
        print("\nResults have been saved to:")
        print("- rfm_analysis_results.csv")
        print("- behavioral_analysis_results.csv")
        print("- deep_learning_analysis_results.csv")
        
    except Exception as e:
        print("\n=== Analysis Failed ===")
        print(f"Error: {str(e)}")
        print("\nDetailed error traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 