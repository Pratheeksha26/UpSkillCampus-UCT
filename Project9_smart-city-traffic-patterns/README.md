# Smart City Traffic Patterns

This project analyzes and forecasts smart city traffic patterns.

## Overview
This repository contains code for analyzing smart city traffic data.

## Introduction
Smart cities generate vast amounts of data, especially from transportation systems. Analyzing and predicting traffic flow is crucial for optimizing urban mobility, reducing congestion, and improving the quality of life for residents.

This project focuses on traffic pattern analysis and forecasting using machine learning techniques. The goal is to build a predictive model that can estimate traffic volume based on various temporal features and junction characteristics.

## Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd smart-city-traffic-patterns
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

The project consists of the following main components:

```
smart-city-traffic-patterns/
├── datasets/
│   ├── train_aWnotuB.csv      # Training data with traffic counts
│   ├── datasets_8494_11879_test_BdBKkAj.csv  # Test data for prediction
│   └── submission.csv          # Submission file template
├── plots/                      # Generated visualizations
│   ├── traffic_over_time.png   # Traffic volume over time per junction
│   └── feature_importance.png  # Feature importance analysis
├── traffic_forecasting.py      # Main analysis and prediction script
├── README.md                   # Project documentation
└── .gitignore                  # Specifies files to ignore in version control
```

## Usage

### Running the Analysis
To run the traffic analysis and generate predictions, execute the main script:
```bash
python traffic_forecasting.py
```

The script performs the following steps:
1. **Load Data**: Reads the training and test datasets.
2. **Feature Engineering**: Extracts temporal features (year, month, day, hour, day of week, weekend/holiday flags).
3. **Model Training**: Trains a RandomForestRegressor model.
4. **Prediction**: Generates traffic volume predictions for the test set.
5. **Submission**: Creates a submission file in the required format.

### Generated Outputs
- **Submission File**: `datasets/submission.csv` - Contains predicted traffic volumes.
- **Plots**: Visualizations saved in the `plots/` directory:
  - `traffic_over_time.png`: Shows traffic trends across different junctions.
  - `feature_importance.png`: Displays feature importance from the Random Forest model.

## Technologies Used

The project utilizes the following libraries:
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Matplotlib & Seaborn**: Data visualization and plotting.
- **Scikit-learn**: Machine learning modeling.

## License
This project is for educational purposes as part of the University of Cape Town internship program.
