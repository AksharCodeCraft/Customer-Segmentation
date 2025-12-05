# Advanced Customer Segmentation Project

This project implements an advanced customer segmentation system using traditional machine learning approaches, combining RFM (Recency, Frequency, Monetary) analysis with behavioral segmentation.

## Project Structure

- `customer_segmentation.py`: Main script containing the segmentation implementation
- `requirements.txt`: Project dependencies
- Data files:
  - `customer_behavior_data.csv`: Customer behavior data
  - `customer_rfm_data.csv`: RFM metrics data
  - `ecommerce_behavior.csv`: E-commerce behavior data

## Features

1. **Data Processing**
   - Data loading and exploration
   - Missing value handling
   - Feature engineering
   - Data normalization

2. **RFM Analysis**
   - Recency calculation
   - Frequency analysis
   - Monetary value computation
   - Customer segmentation based on RFM scores

3. **Behavioral Segmentation**
   - Feature selection for behavioral analysis
   - K-means clustering
   - Optimal cluster number determination
   - Behavioral segment creation

4. **Visualization**
   - RFM segment distribution
   - Behavioral segment distribution
   - RFM vs Behavioral segment comparison

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python customer_segmentation.py
```

## Output

The script will:
1. Load and preprocess the data
2. Create RFM segments
3. Generate behavioral segments
4. Display visualizations of the results

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- python-dateutil 