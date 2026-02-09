# RecipeNet App v2 - Injection Molding Color Recipe Prediction

Enhanced version of the injection molding color recipe prediction system with an interactive recipe simulator.

## Features

### 1️⃣ Recipe Prediction
- Predict pigment recipes from target color names
- Enhanced model with optimized weights
- Top-K pigment selection for practical recipes
- Excel export functionality

### 2️⃣ Recipe Simulator
- Interactive recipe adjustment with top-N pigment selection
- Real-time color prediction from custom recipes
- ΔE00 calculation to compare with target colors
- Visual color comparison

## Installation

### Prerequisites
- Python 3.8 or higher
- Conda (recommended) or virtualenv

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Saymooon/RecipeNet-App-v2.git
cd RecipeNet-App-v2
```

2. Create and activate a conda environment:
```bash
conda create -n ColorMatching python=3.8
conda activate ColorMatching
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app_v2.py
```

The app will open in your default web browser at `http://localhost:8501`

## Required Files

Make sure all the following files are in the same directory:

1. `app_v2.py` - Main application file
2. `recipe_model_optimized_weight_0.1.pth` - Optimized RecipeNet model
3. `name_encoder.pkl` - Text encoder for color names
4. `xgb_surrogate_2.pkl` - XGBoost model for recipe prediction
5. `xgb_surrogate_3.pkl` - XGBoost model for recipe simulator
6. `swatch_recipe_merged_1120.csv` - Reference dataset
7. `requirements.txt` - Python dependencies

## How to Use

### Recipe Prediction
1. Select a target color from the dropdown (56 colors available)
2. Click "레시피 예측 실행" (Run Recipe Prediction)
3. View the predicted recipe and download as Excel if needed

### Recipe Simulator
1. After predicting a recipe, scroll down to the simulator section
2. Click "🔄 예측 레시피 불러오기" to load the predicted recipe
3. Adjust pigment amounts using the sliders
4. Click "🎨 색상 예측 실행" to see the predicted color
5. Compare the result with the target color using ΔE00 metric

## Technical Details

- **Model Architecture**: RecipeNet with 3-head attention mechanism
- **Optimization**: Model weight = 0.7, Similar recipe weight = 0.3
- **Top-K Selection**: Default 5 pigments for practical manufacturing
- **Color Space**: CIE Lab color space
- **Color Difference**: CIEDE2000 (ΔE00) metric

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please open an issue on GitHub.
