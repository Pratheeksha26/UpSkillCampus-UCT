# Project: Agriculture Crop Production Prediction

This repository contains the code and data for predicting crop production and yield in India using machine‑learning models.

## Overview
- **Streamlit app** (`app.py`) – interactive UI for real‑time predictions.
- **Trained models** (`models/*.pkl`) – production and yield models with associated scalers and label encoders.
- **Data** – processed CSV files with engineered features.
- **Visualizations** – EDA and model evaluation plots.

## Setup
```bash
# clone the repo
git clone https://github.com/Pratheeksha26/UpSkillCampus-UCT.git
cd UpSkillCampus-UCT/Project4_Ag_Prediction of Agriculture Crop Production In India

# create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Run the Streamlit app
```bash
streamlit run app.py
```
Open the displayed URL (typically http://localhost:8501) in your browser.

## Deploy to the cloud (Heroku / AWS)
The repository includes `requirements.txt` and a `Procfile` (if you choose Heroku). Follow the standard deployment guides for your platform.

## License
This project is for educational purposes. Feel free to adapt and extend it.
