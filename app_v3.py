"""
Inference Script for Recipe Prediction Model
"""

import numpy as np
import pandas as pd
import torch
import joblib
import warnings
from typing import Dict, Optional

from config import CONFIG
from models import RecipeNet3Head, SimpleNameEncoder
from utils import seed_everything, find_most_similar_recipe, deltaE_00, FeatureBuilder

warnings.filterwarnings('ignore')


class RecipePredictor:
    """
    Recipe prediction inference class
    """
    def __init__(self, model_path=None, encoder_path=None, surrogate_path=None, data_path=None):
        """
        Initialize predictor

        Args:
            model_path: path to saved model weights
            encoder_path: path to saved name encoder
            surrogate_path: path to saved surrogate model
            data_path: path to reference data for similar recipe initialization
        """
        self.device = torch.device(CONFIG['device'])

        # Load model
        model_path = model_path or CONFIG['model_save_path']
        encoder_path = encoder_path or CONFIG['encoder_save_path']
        surrogate_path = surrogate_path or CONFIG['surrogate_path']
        data_path = data_path or CONFIG['data_path']

        print(f"Loading model from {model_path}...")
        in_dim = len(CONFIG['spectrum_cols']) + len(CONFIG['lab_cols']) + CONFIG['embed_dim']
        num_pigments = len(CONFIG['recipe_cols'])

        self.model = RecipeNet3Head(
            in_dim=in_dim,
            num_pigments=num_pigments,
            d_model=CONFIG['d_model'],
            nhead=CONFIG['nhead'],
            nlayers=CONFIG['nlayers']
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded")

        # Load name encoder
        print(f"Loading name encoder from {encoder_path}...")
        self.name_encoder = joblib.load(encoder_path)
        print("Name encoder loaded")

        # Load surrogate model
        print(f"Loading surrogate model from {surrogate_path}...")
        self.surrogate_model = joblib.load(surrogate_path)
        print("Surrogate model loaded")

        # Load reference data
        print(f"Loading reference data from {data_path}...")
        self.ref_df = pd.read_csv(data_path, index_col=0)
        self.ref_df = self.ref_df.dropna()
        print(f"Loaded {len(self.ref_df)} reference samples")

        # Find TiO2 base index
        self.base_idx = CONFIG['recipe_cols'].index(CONFIG['tio2_name'])

        # Prepare feature builder
        self.feature_builder = FeatureBuilder(CONFIG['spectrum_cols'], CONFIG['lab_cols'])
        self.feature_builder.fit_transform(self.ref_df)

    def predict(self, spectrum, lab, color_name, use_similar_init=True, similar_weight=0.3):
        """
        Predict recipe for target color

        Args:
            spectrum: array of shape (31,) - spectral reflectance 400-700nm
            lab: array of shape (3,) - L*, a*, b* values
            color_name: str - color name
            use_similar_init: whether to blend with similar recipe
            similar_weight: weight for similar recipe (0-1)
        Returns:
            Dictionary with prediction results
        """
        seed_everything(CONFIG['random_state'])

        # Encode color name
        text_embedding = self.name_encoder.encode(color_name)

        # Create feature DataFrame for scaling
        feature_dict = {}
        for i, col in enumerate(CONFIG['spectrum_cols']):
            feature_dict[col] = [spectrum[i]]
        for i, col in enumerate(CONFIG['lab_cols']):
            feature_dict[col] = [lab[i]]

        temp_df = pd.DataFrame(feature_dict)
        scaled_features = self.feature_builder.transform(temp_df)

        # Concatenate features
        features = np.concatenate([scaled_features[0], text_embedding])
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict with model
        with torch.no_grad():
            pred_base, pred_chroma, pred_total = self.model(features_tensor)

            # Reconstruct recipe with correct base_idx position
            others = (1 - pred_base) * pred_chroma
            chunks = []
            k = 0
            for j in range(self.model.num_pigments):
                if j == self.base_idx:
                    chunks.append(pred_base)
                else:
                    chunks.append(others[:, k:k+1])
                    k += 1
            full_recipe = torch.cat(chunks, dim=1)

        recipe_pred = full_recipe.cpu().numpy()[0]
        total_pred = pred_total.cpu().numpy()[0, 0]

        # Similar recipe initialization
        if use_similar_init:
            similar_recipe = find_most_similar_recipe(
                lab, self.ref_df, CONFIG['lab_cols'], CONFIG['recipe_cols']
            )
            # Normalize similar recipe
            similar_recipe_norm = similar_recipe / (similar_recipe.sum() + 1e-8)

            # Blend
            recipe_pred = (1 - similar_weight) * recipe_pred + similar_weight * similar_recipe_norm

        # Predict Lab using surrogate
        lab_pred = self.surrogate_model.predict(recipe_pred.reshape(1, -1))[0]

        # Calculate DeltaE
        deltaE = deltaE_00(lab.reshape(1, -1), lab_pred.reshape(1, -1))[0]

        # Scale recipe by total load
        recipe_grams = recipe_pred * total_pred

        results = {
            'recipe_proportions': recipe_pred,
            'recipe_grams': recipe_grams,
            'total_load': total_pred,
            'predicted_lab': lab_pred,
            'target_lab': lab,
            'deltaE_00': deltaE,
            'color_name': color_name
        }

        return results

    def predict_from_dataframe(self, df):
        """
        Batch prediction from DataFrame

        Args:
            df: DataFrame with spectrum, lab, and color name columns
        Returns:
            List of prediction results
        """
        results = []
        for idx, row in df.iterrows():
            spectrum = row[CONFIG['spectrum_cols']].values
            lab = row[CONFIG['lab_cols']].values
            color_name = row[CONFIG['color_col']]

            result = self.predict(spectrum, lab, color_name)
            result['index'] = idx
            results.append(result)

        return results

    def print_recipe(self, results, top_n=10):
        """
        Print recipe in readable format

        Args:
            results: prediction results dictionary
            top_n: number of top pigments to display
        """
        print("\n" + "=" * 80)
        print(f"Recipe Prediction for: {results['color_name']}")
        print("=" * 80)

        print(f"\nTarget Lab: L*={results['target_lab'][0]:.2f}, "
              f"a*={results['target_lab'][1]:.2f}, b*={results['target_lab'][2]:.2f}")
        print(f"Predicted Lab: L*={results['predicted_lab'][0]:.2f}, "
              f"a*={results['predicted_lab'][1]:.2f}, b*={results['predicted_lab'][2]:.2f}")
        print(f"ΔE00: {results['deltaE_00']:.2f}")

        print(f"\nTotal Load: {results['total_load']:.2f} grams")

        # Get top pigments
        recipe_grams = results['recipe_grams']
        pigment_names = CONFIG['recipe_cols']

        pigment_amounts = [(name, amount) for name, amount in zip(pigment_names, recipe_grams)]
        pigment_amounts.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} Pigments:")
        print("-" * 80)
        for i, (name, amount) in enumerate(pigment_amounts[:top_n], 1):
            proportion = amount / results['total_load'] * 100 if results['total_load'] > 0 else 0
            print(f"{i:2d}. {name:30s}: {amount:8.3f}g ({proportion:5.2f}%)")

        print("=" * 80)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Recipe Prediction Inference')
    parser.add_argument('--test_file', '-t', type=str, default=None,
                        help='Path to test file (csv or xlsx). If not provided, uses random samples from training data.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save output results (csv)')
    parser.add_argument('--num_samples', '-n', type=int, default=None,
                        help='Number of samples to test. If not specified, uses all validation samples.')
    args = parser.parse_args()

    print("=" * 80)
    print("Recipe Prediction Inference")
    print("=" * 80)

    # Initialize predictor
    predictor = RecipePredictor()

    # Load test data
    if args.test_file:
        print(f"\nLoading test data from {args.test_file}...")
        if args.test_file.endswith('.xlsx') or args.test_file.endswith('.xls'):
            df = pd.read_excel(args.test_file)
        else:
            df = pd.read_csv(args.test_file, index_col=0)
        test_indices = list(range(len(df)))
    else:
        print(f"\nLoading test data from {CONFIG['data_path']}...")
        df = pd.read_csv(CONFIG['data_path'], index_col=0)
        df = df.dropna()

        # Load validation indices saved during training
        val_indices_path = CONFIG.get('val_indices_path', 'models/val_indices.pkl')
        try:
            val_indices = joblib.load(val_indices_path)
            print(f"Loaded {len(val_indices)} validation indices from {val_indices_path}")

            # Use all validation indices or sample if num_samples is specified
            if args.num_samples and args.num_samples < len(val_indices):
                np.random.seed(42)
                test_indices = np.random.choice(val_indices, size=args.num_samples, replace=False)
            else:
                test_indices = val_indices
        except FileNotFoundError:
            print(f"Warning: Validation indices not found at {val_indices_path}")
            print("Using random samples from all data (may include training samples)")
            np.random.seed(42)
            num_samples = args.num_samples if args.num_samples else 5
            test_indices = np.random.choice(len(df), size=num_samples, replace=False)

    print(f"\nTesting on {len(test_indices)} samples...\n")

    results_list = []
    for idx in test_indices:
        row = df.iloc[idx]
        spectrum = row[CONFIG['spectrum_cols']].values
        lab = row[CONFIG['lab_cols']].values
        color_name = row[CONFIG['color_col']]

        # Predict
        result = predictor.predict(
            spectrum, lab, color_name,
            use_similar_init=CONFIG['use_similar_init'],
            similar_weight=CONFIG['similar_weight']
        )

        # Print
        predictor.print_recipe(result, top_n=10)

        results_list.append({
            'color_name': color_name,
            'target_L': lab[0],
            'target_a': lab[1],
            'target_b': lab[2],
            'pred_L': result['predicted_lab'][0],
            'pred_a': result['predicted_lab'][1],
            'pred_b': result['predicted_lab'][2],
            'deltaE_00': result['deltaE_00'],
            'total_load': result['total_load']
        })

    # Save results
    results_df = pd.DataFrame(results_list)
    output_path = args.output if args.output else CONFIG['results_path']
    if output_path.endswith('.xlsx') or output_path.endswith('.xls'):
        results_df.to_excel(output_path, index=False)
    else:
        results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Mean ΔE00: {results_df['deltaE_00'].mean():.2f}")
    print(f"Median ΔE00: {results_df['deltaE_00'].median():.2f}")
    print(f"95th percentile ΔE00: {results_df['deltaE_00'].quantile(0.95):.2f}")
    print(f"Max ΔE00: {results_df['deltaE_00'].max():.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
