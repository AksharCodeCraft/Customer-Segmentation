import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import traceback
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self, data_path):
        self.data_path = data_path
        self.customer_data = None
        self.rfm_scores = None
        self.behavioral_features = None
        self.model_weights = {'rfm': 0.4, 'behavioral': 0.3, 'deep_learning': 0.3}
        self.ensemble_segments = None

    def load_data(self):
        """Load and preprocess the customer data"""
        try:
            # Load the data
            print(f"Loading data from: {self.data_path}")
            self.customer_data = pd.read_csv(self.data_path)
            
            # Ensure required columns exist
            required_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
                              'Page Views', 'Time on Site (minutes)', 'Cart Abandonment Rate']
            missing_columns = [col for col in required_columns if col not in self.customer_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert date columns if they exist
            date_columns = ['InvoiceDate', 'Last Purchase Date']
            for col in date_columns:
                if col in self.customer_data.columns:
                    self.customer_data[col] = pd.to_datetime(self.customer_data[col])
            
            print(f"\nData loaded successfully. Shape: {self.customer_data.shape}")
            print("\nAvailable columns:", self.customer_data.columns.tolist())
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("\nDetailed error traceback:")
            traceback.print_exc()
            raise

    def explore_data(self):
        """Perform initial data exploration"""
        try:
            print("\n=== DATA EXPLORATION ===")
            print("\nBasic Information:")
            print(self.customer_data.info())
            
            print("\nSummary Statistics:")
            print(self.customer_data.describe())
            
            print("\nMissing Values:")
            print(self.customer_data.isnull().sum())
            
        except Exception as e:
            print(f"Error in data exploration: {e}")
            raise

    def preprocess_data(self):
        """Preprocess the data for analysis"""
        try:
            # Remove any negative values in numeric columns
            numeric_cols = self.customer_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col in ['Quantity', 'UnitPrice', 'TotalSpend', 'Monetary']:
                    self.customer_data = self.customer_data[self.customer_data[col] > 0]
            
            # Ensure all required numeric columns are present
            required_numeric = ['Recency', 'Frequency', 'Monetary']
            for col in required_numeric:
                if col not in self.customer_data.columns:
                    raise ValueError(f"Missing required numeric column: {col}")
            
            # Create additional features if date columns exist
            if 'InvoiceDate' in self.customer_data.columns:
                self.customer_data['Year'] = self.customer_data['InvoiceDate'].dt.year
                self.customer_data['Month'] = self.customer_data['InvoiceDate'].dt.month
                self.customer_data['Day'] = self.customer_data['InvoiceDate'].dt.day
                self.customer_data['DayOfWeek'] = self.customer_data['InvoiceDate'].dt.dayofweek
            
            print("Data preprocessing completed successfully.")
            print(f"Final dataset shape: {self.customer_data.shape}")
            
        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            print("\nDetailed error traceback:")
            traceback.print_exc()
            raise

    def analyze_and_summarize_results(self):
        """Analyze and summarize the segmentation results in a clear, tabulated format"""
        try:
            print("\n=== CUSTOMER SEGMENTATION ANALYSIS SUMMARY ===\n")
            
            # 1. RFM Analysis Summary
            print("1. RFM SEGMENTATION ANALYSIS")
            print("-" * 50)
            rfm_summary = self.customer_data.groupby('RFM_Segment').agg({
                'Recency': ['mean', 'std'],
                'Frequency': ['mean', 'std'],
                'Monetary': ['mean', 'std'],
                'CustomerID': 'count'
            }).round(2)
            
            # Calculate percentage of customers in each segment
            total_customers = len(self.customer_data)
            rfm_summary['Percentage'] = (rfm_summary[('CustomerID', 'count')] / total_customers * 100).round(2)
            
            print("\nRFM Segment Statistics:")
            print(rfm_summary)
            
            # 2. Behavioral Analysis Summary
            print("\n2. BEHAVIORAL SEGMENTATION ANALYSIS")
            print("-" * 50)
            
            # Get all behavioral features
            behavioral_features = [col for col in self.customer_data.columns 
                                 if col not in ['CustomerID', 'RFM_Segment', 'Behavioral_Segment', 
                                              'Recency', 'Frequency', 'Monetary', 'RFM_Score']]
            
            behavioral_summary = self.customer_data.groupby('Behavioral_Segment').agg({
                **{feature: ['mean', 'std'] for feature in behavioral_features},
                'CustomerID': 'count'
            }).round(2)
            
            # Calculate percentage of customers in each segment
            behavioral_summary['Percentage'] = (behavioral_summary[('CustomerID', 'count')] / total_customers * 100).round(2)
            
            print("\nBehavioral Segment Statistics:")
            print(behavioral_summary)
            
            # 3. Cross-Segmentation Analysis
            print("\n3. CROSS-SEGMENTATION ANALYSIS")
            print("-" * 50)
            cross_segmentation = pd.crosstab(
                self.customer_data['RFM_Segment'],
                self.customer_data['Behavioral_Segment'],
                normalize='index'
            ).round(2) * 100
            
            print("\nRFM vs Behavioral Segment Distribution (%):")
            print(cross_segmentation)
            
            # 4. Key Insights and Recommendations
            print("\n4. KEY INSIGHTS AND RECOMMENDATIONS")
            print("-" * 50)
            
            # RFM Insights
            print("\nRFM Insights:")
            top_rfm_segment = rfm_summary['Percentage'].idxmax()
            print(f"- The largest customer segment is '{top_rfm_segment}' with {rfm_summary.loc[top_rfm_segment, 'Percentage']}% of customers")
            
            # Behavioral Insights
            print("\nBehavioral Insights:")
            top_behavioral_segment = behavioral_summary['Percentage'].idxmax()
            print(f"- The most common behavioral pattern is Segment {top_behavioral_segment} with {behavioral_summary.loc[top_behavioral_segment, 'Percentage']}% of customers")
            
            # Cross-Segmentation Insights
            print("\nCross-Segmentation Insights:")
            max_overlap = cross_segmentation.max().max()
            max_overlap_indices = np.where(cross_segmentation == max_overlap)
            rfm_segment = cross_segmentation.index[max_overlap_indices[0][0]]
            behavioral_segment = cross_segmentation.columns[max_overlap_indices[1][0]]
            print(f"- The strongest overlap is between RFM segment '{rfm_segment}' and Behavioral segment {behavioral_segment} ({max_overlap}%)")
            
            # Recommendations
            print("\nRecommendations:")
            print("1. Focus marketing efforts on the largest RFM segment to maximize ROI")
            print("2. Develop targeted campaigns for the most common behavioral patterns")
            print("3. Create personalized experiences for customers in the overlapping segments")
            print("4. Monitor segment migration over time to identify trends")
            print("5. Consider segment-specific pricing and promotion strategies")
            
            # Save results to CSV
            rfm_summary.to_csv('rfm_analysis_summary.csv')
            behavioral_summary.to_csv('behavioral_analysis_summary.csv')
            cross_segmentation.to_csv('cross_segmentation_analysis.csv')
            
            print("\nDetailed analysis results have been saved to CSV files.")
            
        except Exception as e:
            print(f"Error in analyzing and summarizing results: {e}")
            raise

    def traditional_rfm_analysis(self):
        """Traditional RFM analysis using quintiles"""
        try:
            # Calculate RFM metrics
            rfm = self.customer_data.groupby('CustomerID').agg({
                'Recency': 'first',
                'Frequency': 'first',
                'Monetary': 'first'
            })
            
            # Calculate quintiles
            rfm['R_Quintile'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
            rfm['F_Quintile'] = pd.qcut(rfm['Frequency'], q=5, labels=[1, 2, 3, 4, 5])
            rfm['M_Quintile'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
            
            # Calculate RFM score
            rfm['RFM_Score'] = rfm['R_Quintile'].astype(int) + rfm['F_Quintile'].astype(int) + rfm['M_Quintile'].astype(int)
            
            # Create segments based on RFM score
            rfm['RFM_Segment'] = pd.qcut(rfm['RFM_Score'], q=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
            
            # Print RFM analysis results
            print("\n=== RFM ANALYSIS RESULTS ===")
            print("\nRFM Score Distribution:")
            print(rfm['RFM_Score'].value_counts().sort_index())
            
            print("\nRFM Segment Distribution:")
            print(rfm['RFM_Segment'].value_counts())
            
            # Save RFM results
            rfm.to_csv('rfm_analysis_results.csv')
            
            return rfm
            
        except Exception as e:
            print(f"Error in traditional RFM analysis: {e}")
            raise

    def behavioral_psychology_analysis(self):
        """Behavioral psychology analysis incorporating personality traits and cognitive biases"""
        try:
            # Calculate behavioral features
            behavioral_data = self.customer_data.groupby('CustomerID').agg({
                'Page Views': 'mean',
                'Time on Site (minutes)': 'mean',
                'Cart Abandonment Rate': 'mean',
                'Product Views': 'mean',
                'Search Queries': 'mean',
                'Checkout Time (minutes)': 'mean',
                'Customer Service Interactions': 'mean',
                'Purchase Frequency': 'mean',
                'Avg Spend Per Purchase': 'mean'
            })
            
            # Calculate personality trait proxies
            # Openness: Variety in product views and search queries
            behavioral_data['Openness'] = (behavioral_data['Product Views'] + behavioral_data['Search Queries']) / 2
            
            # Conscientiousness: Regularity of purchases and low cart abandonment
            behavioral_data['Conscientiousness'] = (1 - behavioral_data['Cart Abandonment Rate']) * behavioral_data['Purchase Frequency']
            
            # Extraversion: High page views and time on site
            behavioral_data['Extraversion'] = (behavioral_data['Page Views'] + behavioral_data['Time on Site (minutes)']) / 2
            
            # Agreeableness: Customer service interactions
            behavioral_data['Agreeableness'] = behavioral_data['Customer Service Interactions']
            
            # Neuroticism: Checkout time and cart abandonment
            behavioral_data['Neuroticism'] = (behavioral_data['Checkout Time (minutes)'] + behavioral_data['Cart Abandonment Rate']) / 2
            
            # Normalize scores
            behavioral_data = (behavioral_data - behavioral_data.mean()) / behavioral_data.std()
            
            # Perform clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            behavioral_data['Behavioral_Segment'] = kmeans.fit_predict(behavioral_data)
            
            # Print behavioral analysis results
            print("\n=== BEHAVIORAL ANALYSIS RESULTS ===")
            print("\nBehavioral Segment Distribution:")
            print(behavioral_data['Behavioral_Segment'].value_counts().sort_index())
            
            # Save behavioral results
            behavioral_data.to_csv('behavioral_analysis_results.csv')
            
            return behavioral_data
            
        except Exception as e:
            print(f"Error in behavioral psychology analysis: {e}")
            raise

    def deep_learning_behavioral_analysis(self):
        """Deep learning based behavioral analysis using time-series patterns"""
        try:
            # Create time-series features
            time_series_data = self.customer_data.groupby(['CustomerID', 'InvoiceDate']).agg({
                'TotalSpend': 'sum',
                'Quantity': 'sum',
                'UnitPrice': 'mean'
            }).reset_index()
            
            # Create sequence features
            sequence_features = []
            for customer_id in time_series_data['CustomerID'].unique():
                customer_data = time_series_data[time_series_data['CustomerID'] == customer_id]
                customer_data = customer_data.sort_values('InvoiceDate')
                
                # Calculate time-based features
                purchase_sequence = customer_data['TotalSpend'].values
                
                # Calculate sequence statistics
                sequence_features.append({
                    'CustomerID': customer_id,
                    'Purchase_Mean': np.mean(purchase_sequence),
                    'Purchase_Std': np.std(purchase_sequence) if len(purchase_sequence) > 1 else 0,
                    'Purchase_Min': np.min(purchase_sequence),
                    'Purchase_Max': np.max(purchase_sequence),
                    'Purchase_Range': np.ptp(purchase_sequence)
                })
            
            sequence_features = pd.DataFrame(sequence_features)
            
            # Calculate engagement metrics
            engagement_metrics = self.customer_data.groupby('CustomerID').agg({
                'Page Views': 'mean',
                'Time on Site (minutes)': 'mean',
                'Cart Abandonment Rate': 'mean',
                'Product Views': 'mean',
                'Search Queries': 'mean'
            })
            
            # Combine all features
            deep_learning_features = pd.merge(sequence_features, engagement_metrics, on='CustomerID')
            
            # Normalize features
            scaler = StandardScaler()
            feature_columns = deep_learning_features.columns.drop('CustomerID')
            deep_learning_features[feature_columns] = scaler.fit_transform(deep_learning_features[feature_columns])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            deep_learning_features['Deep_Learning_Segment'] = kmeans.fit_predict(deep_learning_features[feature_columns])
            
            # Print analysis results
            print("\n=== DEEP LEARNING BEHAVIORAL ANALYSIS RESULTS ===")
            print("\nSegment Distribution:")
            segment_dist = deep_learning_features['Deep_Learning_Segment'].value_counts().sort_index()
            print(segment_dist)
            
            # Calculate segment characteristics
            segment_profiles = deep_learning_features.groupby('Deep_Learning_Segment')[feature_columns].mean()
            
            print("\nSegment Profiles (normalized values):")
            print(segment_profiles.round(2))
            
            # Save results
            deep_learning_features.to_csv('deep_learning_analysis_results.csv', index=False)
            
            return deep_learning_features
            
        except Exception as e:
            print(f"Error in deep learning behavioral analysis: {e}")
            raise

    def create_ensemble_segments(self):
        """Create ensemble segments by combining all models"""
        try:
            # Get predictions from all models
            rfm_segments = self.traditional_rfm_analysis()
            behavioral_segments = self.behavioral_psychology_analysis()
            deep_learning_segments = self.deep_learning_behavioral_analysis()
            
            # Reset index for all dataframes to ensure proper alignment
            rfm_segments = rfm_segments.reset_index()
            behavioral_segments = behavioral_segments.reset_index()
            deep_learning_segments = deep_learning_segments.reset_index()
            
            # Combine segments with weights
            ensemble_data = pd.DataFrame({
                'CustomerID': rfm_segments['CustomerID'],
                'RFM_Segment': rfm_segments['RFM_Segment'],
                'Behavioral_Segment': behavioral_segments['Behavioral_Segment'],
                'Deep_Learning_Segment': deep_learning_segments['Deep_Learning_Segment']
            })
            
            # Create meta-features
            meta_features = pd.get_dummies(ensemble_data[['RFM_Segment', 'Behavioral_Segment', 'Deep_Learning_Segment']])
            
            # Normalize meta-features
            scaler = StandardScaler()
            meta_features_scaled = scaler.fit_transform(meta_features)
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            ensemble_data['Final_Segment'] = kmeans.fit_predict(meta_features_scaled)
            
            # Map segments to meaningful labels
            segment_mapping = {
                0: 'Low-Value Traditional',
                1: 'High-Value Traditional',
                2: 'Behavioral-Focused',
                3: 'Deep-Learning Identified',
                4: 'Mixed Profile'
            }
            
            ensemble_data['Final_Segment_Label'] = ensemble_data['Final_Segment'].map(segment_mapping)
            
            # Print ensemble analysis results
            print("\n=== ENSEMBLE SEGMENTATION RESULTS ===")
            print("\nSegment Distribution:")
            print(ensemble_data['Final_Segment_Label'].value_counts())
            
            # Calculate segment profiles
            segment_profiles = ensemble_data.groupby('Final_Segment_Label').agg({
                'RFM_Segment': lambda x: x.mode()[0],
                'Behavioral_Segment': lambda x: x.mode()[0],
                'Deep_Learning_Segment': lambda x: x.mode()[0]
            })
            
            print("\nSegment Profiles:")
            print(segment_profiles)
            
            self.ensemble_segments = ensemble_data
            return ensemble_data
            
        except Exception as e:
            print(f"Error in creating ensemble segments: {e}")
            raise

    def analyze_ensemble_results(self):
        """Analyze and summarize the ensemble segmentation results"""
        try:
            if self.ensemble_segments is None:
                self.create_ensemble_segments()
            
            print("\n=== ENSEMBLE SEGMENTATION ANALYSIS ===\n")
            
            # Segment distribution
            segment_dist = self.ensemble_segments['Final_Segment_Label'].value_counts(normalize=True) * 100
            print("\nSegment Distribution (%):")
            print(segment_dist.round(2))
            
            # Model agreement analysis
            agreement_matrix = pd.crosstab(
                self.ensemble_segments['RFM_Segment'],
                self.ensemble_segments['Behavioral_Segment']
            )
            
            print("\nModel Agreement Matrix:")
            print(agreement_matrix)
            
            # Calculate segment characteristics
            segment_chars = self.ensemble_segments.groupby('Final_Segment_Label').agg({
                'RFM_Segment': lambda x: x.mode()[0],
                'Behavioral_Segment': 'mean',
                'Deep_Learning_Segment': 'mean'
            }).round(2)
            
            print("\nSegment Characteristics:")
            print(segment_chars)
            
            # Save results
            self.ensemble_segments.to_csv('ensemble_segments.csv', index=False)
            segment_dist.to_csv('segment_distribution.csv')
            agreement_matrix.to_csv('model_agreement.csv')
            
            print("\nEnsemble analysis results saved to CSV files.")
            
            # Generate recommendations
            print("\nRecommendations for Each Segment:")
            for segment in self.ensemble_segments['Final_Segment_Label'].unique():
                segment_data = self.ensemble_segments[self.ensemble_segments['Final_Segment_Label'] == segment]
                print(f"\n{segment}:")
                print(f"- Size: {len(segment_data)} customers ({(len(segment_data)/len(self.ensemble_segments)*100):.1f}%)")
                print(f"- Typical RFM Segment: {segment_data['RFM_Segment'].mode()[0]}")
                print(f"- Average Behavioral Segment: {segment_data['Behavioral_Segment'].mean():.1f}")
                print(f"- Average Deep Learning Segment: {segment_data['Deep_Learning_Segment'].mean():.1f}")
            
        except Exception as e:
            print(f"Error in analyzing ensemble results: {e}")
            raise

    def run_analysis(self):
        """Run the complete analysis pipeline including ensemble methods"""
        try:
            print("Loading data...")
            self.load_data()
            
            print("\nExploring data...")
            self.explore_data()
            
            print("\nPreprocessing data...")
            self.preprocess_data()
            
            print("\nPerforming traditional RFM analysis...")
            self.traditional_rfm_analysis()
            
            print("\nPerforming behavioral psychology analysis...")
            self.behavioral_psychology_analysis()
            
            print("\nPerforming deep learning behavioral analysis...")
            self.deep_learning_behavioral_analysis()
            
            print("\nCreating ensemble segments...")
            self.create_ensemble_segments()
            
            print("\nAnalyzing ensemble results...")
            self.analyze_ensemble_results()
            
            print("\nAnalysis complete! Results saved as CSV files.")
            
        except Exception as e:
            print(f"Error running analysis: {e}")
            raise 