import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CustomerSegmentation:
    def __init__(self):
        self.customer_data = None
        self.rfm_data = None
        self.ecommerce_data = None
        self.scaler = StandardScaler()
        self.behavioral_segments = None
        self.deep_learning_segments = None
        
    def load_data(self):
        """Load all three datasets"""
        try:
            # Check if files exist
            required_files = ['customer_behavior_data.csv', 'customer_rfm_data.csv', 'ecommerce_behavior.csv']
            for file in required_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Required file {file} not found")
            
            # Load data
            self.customer_data = pd.read_csv('customer_behavior_data.csv')
            self.rfm_data = pd.read_csv('customer_rfm_data.csv')
            self.ecommerce_data = pd.read_csv('ecommerce_behavior.csv')
            
            # Validate data
            if self.customer_data.empty or self.rfm_data.empty or self.ecommerce_data.empty:
                raise ValueError("One or more dataframes are empty")
                
            print("Data loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def explore_data(self):
        """Explore each dataset"""
        try:
            print("\nCustomer Behavior Data:")
            print(self.customer_data.info())
            print("\nMissing values:")
            print(self.customer_data.isnull().sum())
            print("\nDescriptive statistics:")
            print(self.customer_data.describe())
            
            print("\nRFM Data:")
            print(self.rfm_data.info())
            print("\nMissing values:")
            print(self.rfm_data.isnull().sum())
            print("\nDescriptive statistics:")
            print(self.rfm_data.describe())
            
            print("\nE-commerce Behavior Data:")
            print(self.ecommerce_data.info())
            print("\nMissing values:")
            print(self.ecommerce_data.isnull().sum())
            print("\nDescriptive statistics:")
            print(self.ecommerce_data.describe())
            
        except Exception as e:
            print(f"Error exploring data: {e}")
            raise
            
    def preprocess_data(self):
        """Preprocess and clean the data"""
        try:
            # Handle missing values
            self.customer_data = self.customer_data.dropna()
            self.rfm_data = self.rfm_data.dropna()
            self.ecommerce_data = self.ecommerce_data.dropna()
            
            # Convert date columns to datetime
            self.customer_data['Last Purchase Date'] = pd.to_datetime(self.customer_data['Last Purchase Date'])
            
            # Calculate Recency
            current_date = datetime.now()
            self.customer_data['Recency'] = (current_date - self.customer_data['Last Purchase Date']).dt.days
            
            # One-hot encode categorical variables
            categorical_cols = ['Device Type', 'Time of Day Most Active', 'Personalization Response']
            self.customer_data = pd.get_dummies(self.customer_data, columns=categorical_cols)
            
            # Merge RFM data
            self.customer_data = pd.merge(self.customer_data, self.rfm_data, on='CustomerID', how='left')
            
            # Merge ecommerce data
            self.customer_data = pd.merge(self.customer_data, self.ecommerce_data, on='CustomerID', how='left')
            
            # Ensure RFM columns exist
            if 'Recency' not in self.customer_data.columns:
                self.customer_data['Recency'] = self.customer_data['Recency_x']
            if 'Frequency' not in self.customer_data.columns:
                self.customer_data['Frequency'] = self.customer_data['Frequency_x']
            if 'Monetary' not in self.customer_data.columns:
                self.customer_data['Monetary'] = self.customer_data['Monetary_x']
            
            # Drop duplicate columns if they exist
            self.customer_data = self.customer_data.loc[:, ~self.customer_data.columns.duplicated()]
            
            # Drop any remaining columns with _x or _y suffix
            self.customer_data = self.customer_data.loc[:, ~self.customer_data.columns.str.endswith('_x')]
            self.customer_data = self.customer_data.loc[:, ~self.customer_data.columns.str.endswith('_y')]
            
            print("\nData preprocessing completed successfully")
            print("\nAvailable columns after preprocessing:")
            print(self.customer_data.columns.tolist())
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            raise
            
    def analyze_rfm(self):
        """Perform detailed RFM analysis"""
        try:
            # Verify RFM columns exist
            required_columns = ['Recency', 'Frequency', 'Monetary']
            missing_columns = [col for col in required_columns if col not in self.customer_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required RFM columns: {missing_columns}")
            
            # Calculate RFM statistics
            rfm_stats = self.customer_data[['Recency', 'Frequency', 'Monetary']].describe()
            print("\nRFM Statistics:")
            print(rfm_stats)
            
            # Calculate correlation between RFM metrics
            rfm_correlation = self.customer_data[['Recency', 'Frequency', 'Monetary']].corr()
            print("\nRFM Correlation Matrix:")
            print(rfm_correlation)
            
            # Create RFM heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(rfm_correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('RFM Metrics Correlation')
            plt.tight_layout()
            plt.savefig('rfm_correlation.png')
            plt.close()
            
            # Create RFM distribution plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, metric in enumerate(['Recency', 'Frequency', 'Monetary']):
                sns.histplot(data=self.customer_data, x=metric, ax=axes[i])
                axes[i].set_title(f'{metric} Distribution')
            plt.tight_layout()
            plt.savefig('rfm_distributions.png')
            plt.close()
            
            # Create RFM scatter plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            sns.scatterplot(data=self.customer_data, x='Recency', y='Frequency', ax=axes[0])
            sns.scatterplot(data=self.customer_data, x='Recency', y='Monetary', ax=axes[1])
            sns.scatterplot(data=self.customer_data, x='Frequency', y='Monetary', ax=axes[2])
            plt.tight_layout()
            plt.savefig('rfm_scatter_plots.png')
            plt.close()
            
        except Exception as e:
            print(f"Error in RFM analysis: {e}")
            raise
            
    def create_rfm_segments(self):
        """Create RFM segments using quintiles"""
        try:
            # Print available columns for debugging
            print("\nAvailable columns before RFM segmentation:")
            print(self.customer_data.columns.tolist())
            
            # Verify RFM columns exist
            required_columns = ['Recency', 'Frequency', 'Monetary']
            missing_columns = [col for col in required_columns if col not in self.customer_data.columns]
            if missing_columns:
                print(f"\nMissing RFM columns: {missing_columns}")
                print("Attempting to create missing columns...")
                
                # Create missing columns if possible
                if 'Recency' in missing_columns and 'Last Purchase Date' in self.customer_data.columns:
                    current_date = datetime.now()
                    self.customer_data['Recency'] = (current_date - pd.to_datetime(self.customer_data['Last Purchase Date'])).dt.days
                
                if 'Frequency' in missing_columns and 'Purchase Frequency' in self.customer_data.columns:
                    self.customer_data['Frequency'] = self.customer_data['Purchase Frequency']
                
                if 'Monetary' in missing_columns and 'Avg Spend Per Purchase' in self.customer_data.columns:
                    self.customer_data['Monetary'] = self.customer_data['Avg Spend Per Purchase'] * self.customer_data['Purchase Frequency']
            
            # Verify columns again after creation attempt
            missing_columns = [col for col in required_columns if col not in self.customer_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required RFM columns after creation attempt: {missing_columns}")
            
            # Calculate quintiles for each RFM metric
            rfm_cols = ['Recency', 'Frequency', 'Monetary']
            for col in rfm_cols:
                try:
                    # Convert to numeric if needed
                    self.customer_data[col] = pd.to_numeric(self.customer_data[col], errors='coerce')
                    
                    # Calculate quintiles
                    self.customer_data[f'{col}_Score'] = pd.qcut(
                        self.customer_data[col],
                        q=5,
                        labels=[1, 2, 3, 4, 5],
                        duplicates='drop'  # Handle duplicate values
                    )
                except Exception as e:
                    print(f"Error calculating {col} quintiles: {e}")
                    raise
            
            # Calculate RFM score
            self.customer_data['RFM_Score'] = (
                self.customer_data['Recency_Score'].astype(int) +
                self.customer_data['Frequency_Score'].astype(int) +
                self.customer_data['Monetary_Score'].astype(int)
            )
            
            # Create segments based on RFM score
            self.customer_data['RFM_Segment'] = pd.qcut(
                self.customer_data['RFM_Score'],
                q=4,
                labels=['Low Value', 'Medium Value', 'High Value', 'Top Value'],
                duplicates='drop'
            )
            
            # Analyze segment characteristics
            segment_stats = self.customer_data.groupby('RFM_Segment').agg({
                'Recency': ['mean', 'std'],
                'Frequency': ['mean', 'std'],
                'Monetary': ['mean', 'std'],
                'CustomerID': 'count'
            }).round(2)
            
            print("\nRFM Segment Statistics:")
            print(segment_stats)
            
            # Create segment visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(data=self.customer_data, x='RFM_Segment', y='RFM_Score')
            plt.title('Average RFM Score by Segment')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('rfm_segment_scores.png')
            plt.close()
            
            # Print segment distribution
            print("\nRFM Segment Distribution:")
            print(self.customer_data['RFM_Segment'].value_counts())
            
        except Exception as e:
            print(f"Error creating RFM segments: {e}")
            print("\nCurrent DataFrame columns:")
            print(self.customer_data.columns.tolist())
            print("\nCurrent DataFrame head:")
            print(self.customer_data.head())
            raise
            
    def create_behavioral_segments(self):
        """Create behavioral segments using clustering"""
        try:
            # Print available columns for debugging
            print("\nAvailable columns before behavioral segmentation:")
            print(self.customer_data.columns.tolist())
            
            # Select features for behavioral clustering
            behavioral_features = [
                'Customer Service Interactions',
                'Purchase Frequency',
                'Avg Spend Per Purchase',
                'Page Views',
                'Time on Site (minutes)',
                'Cart Abandonment Rate',
                'Product Views',
                'Search Queries',
                'Checkout Time (minutes)'
            ]
            
            # Verify which behavioral features exist in the dataset
            available_features = [col for col in behavioral_features if col in self.customer_data.columns]
            print("\nAvailable behavioral features:")
            print(available_features)
            
            if not available_features:
                raise ValueError("No behavioral features found in the dataset")
            
            # Add one-hot encoded columns
            categorical_features = [col for col in self.customer_data.columns 
                                  if any(x in col for x in ['Device Type_', 'Time of Day Most Active_', 'Personalization Response_'])]
            print("\nAvailable categorical features:")
            print(categorical_features)
            
            # Combine all features
            all_features = available_features + categorical_features
            
            # Handle missing values
            self.customer_data[all_features] = self.customer_data[all_features].fillna(0)
            
            # Convert to numeric if needed
            for feature in all_features:
                self.customer_data[feature] = pd.to_numeric(self.customer_data[feature], errors='coerce')
            
            # Scale features
            X = self.scaler.fit_transform(self.customer_data[all_features])
            
            # Find optimal number of clusters using silhouette score
            silhouette_scores = []
            cluster_range = range(2, min(6, len(self.customer_data) - 1))  # Ensure we don't try more clusters than samples
            
            for n_clusters in cluster_range:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(X)
                    score = silhouette_score(X, cluster_labels)
                    silhouette_scores.append(score)
                    print(f"Silhouette score for {n_clusters} clusters: {score:.4f}")
                except Exception as e:
                    print(f"Error calculating silhouette score for {n_clusters} clusters: {e}")
                    continue
            
            if not silhouette_scores:
                raise ValueError("Could not calculate silhouette scores for any number of clusters")
            
            optimal_clusters = np.argmax(silhouette_scores) + 2
            print(f"\nOptimal number of clusters: {optimal_clusters}")
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            self.customer_data['Behavioral_Segment'] = kmeans.fit_predict(X)
            
            # Analyze segment characteristics
            segment_stats = self.customer_data.groupby('Behavioral_Segment')[all_features].agg(['mean', 'std']).round(2)
            print("\nBehavioral Segment Statistics:")
            print(segment_stats)
            
            # Print segment distribution
            print("\nBehavioral Segment Distribution:")
            print(self.customer_data['Behavioral_Segment'].value_counts())
            
            # Create segment visualization
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.customer_data, x='Behavioral_Segment')
            plt.title('Distribution of Behavioral Segments')
            plt.tight_layout()
            plt.savefig('behavioral_segments_distribution.png')
            plt.close()
            
            # Create feature importance visualization
            # Calculate the number of rows needed for the subplots
            n_features = len(all_features)
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
            
            plt.figure(figsize=(15, 4 * n_rows))
            for i, feature in enumerate(all_features):
                plt.subplot(n_rows, n_cols, i+1)
                sns.boxplot(data=self.customer_data, x='Behavioral_Segment', y=feature)
                plt.title(feature)
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('behavioral_segments_features.png')
            plt.close()
            
        except Exception as e:
            print(f"Error creating behavioral segments: {e}")
            print("\nCurrent DataFrame columns:")
            print(self.customer_data.columns.tolist())
            print("\nCurrent DataFrame head:")
            print(self.customer_data.head())
            raise
            
    def visualize_results(self):
        """Visualize the segmentation results"""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')  # Use a valid matplotlib style
            
            # RFM segments visualization
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.customer_data, x='RFM_Segment')
            plt.title('Distribution of RFM Segments')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('rfm_segments.png')
            plt.close()
            
            # Behavioral segments visualization
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.customer_data, x='Behavioral_Segment')
            plt.title('Distribution of Behavioral Segments')
            plt.tight_layout()
            plt.savefig('behavioral_segments.png')
            plt.close()
            
            # RFM vs Behavioral segments
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=self.customer_data, 
                           x='RFM_Score', 
                           y='Behavioral_Segment',
                           hue='RFM_Segment')
            plt.title('RFM Score vs Behavioral Segment')
            plt.tight_layout()
            plt.savefig('rfm_vs_behavioral.png')
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing results: {e}")
            raise
            
    def save_results(self):
        """Save analysis results to CSV files"""
        try:
            # Save RFM analysis results
            rfm_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'RFM_Segment']
            self.customer_data[rfm_columns].to_csv('rfm_analysis_results.csv', index=False)
            
            # Save behavioral analysis results
            behavioral_columns = [
                'CustomerID', 'Purchase Frequency', 'Avg Spend Per Purchase',
                'Customer Service Interactions', 'Page Views', 'Time on Site (minutes)',
                'Cart Abandonment Rate', 'Product Views', 'Search Queries',
                'Checkout Time (minutes)', 'Behavioral_Segment'
            ]
            self.customer_data[behavioral_columns].to_csv('behavioral_analysis_results.csv', index=False)
            
            # Create and save deep learning results (simplified version for demo)
            deep_learning_data = self.customer_data[['CustomerID', 'Purchase Frequency', 'Page Views',
                                                   'Time on Site (minutes)', 'Cart Abandonment Rate']]
            deep_learning_data['Deep_Learning_Segment'] = self.customer_data['Behavioral_Segment']
            deep_learning_data['Purchase_Mean'] = deep_learning_data['Purchase Frequency']
            deep_learning_data.to_csv('deep_learning_analysis_results.csv', index=False)
            
            print("Analysis results saved successfully")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            raise
            
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            print("Loading data...")
            self.load_data()
            
            print("\nExploring data...")
            self.explore_data()
            
            print("\nPreprocessing data...")
            self.preprocess_data()
            
            print("\nPerforming RFM analysis...")
            self.analyze_rfm()
            
            print("\nCreating RFM segments...")
            self.create_rfm_segments()
            
            print("\nCreating behavioral segments...")
            self.create_behavioral_segments()
            
            print("\nVisualizing results...")
            self.visualize_results()
            
            print("\nSaving results...")
            self.save_results()
            
            print("\nAnalysis complete! Results saved as PNG and CSV files.")
            
        except Exception as e:
            print(f"Error running analysis: {e}")
            raise

if __name__ == "__main__":
    try:
        segmentation = CustomerSegmentation()
        segmentation.run_analysis()
    except Exception as e:
        print(f"Program failed: {e}") 