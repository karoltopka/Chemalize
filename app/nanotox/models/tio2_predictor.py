"""
TiO2 Predictor wrapper class for the EBM model
"""
import pandas as pd
import numpy as np


class TiO2Predictor:
    """
    Wrapper class for TiO2 nanoparticle toxicity prediction using EBM model.

    The model predicts cell viability (%) based on nanoparticle characteristics.
    """

    # Expected feature order for the model (matches training data)
    FEATURE_ORDER = [
        'Shape',
        'anatase',
        'rutile',
        'Diameter (nm)',
        'Cell_Type',
        'Time (hr)',
        'Test',
        'Concentration (ug/ml)_log'
    ]

    # Mapping from user-friendly names to model feature names
    FEATURE_MAPPING = {
        'Anatase': 'anatase',
        'Rutile': 'rutile',
        'Concentration (ug/ml)': 'Concentration (ug/ml)_log'
    }

    def __init__(self, model_package):
        """
        Initialize the predictor with a loaded model package.

        Args:
            model_package: Dictionary containing the EBM model and metadata
        """
        # Handle both dict with 'model' key and direct model object
        if isinstance(model_package, dict):
            self.model = model_package.get('model', model_package)
            self.metadata = model_package.get('metadata', {})
        else:
            self.model = model_package
            self.metadata = {}

    def _transform_input(self, data):
        """
        Transform user input to model-expected format.

        - Renames Anatase -> anatase, Rutile -> rutile
        - Converts Concentration to log10
        """
        transformed = {}
        for key, value in data.items():
            if key == 'Concentration (ug/ml)':
                # Apply log10 transformation
                conc = float(value)
                transformed['Concentration (ug/ml)_log'] = np.log10(conc) if conc > 0 else 0
            elif key in self.FEATURE_MAPPING:
                transformed[self.FEATURE_MAPPING[key]] = value
            else:
                transformed[key] = value
        return transformed

    def predict(self, data):
        """
        Make a single prediction.

        Args:
            data: Dictionary with feature names as keys

        Returns:
            float: Predicted cell viability (%)
        """
        transformed = self._transform_input(data)
        df = self._prepare_dataframe(transformed)
        prediction = self.model.predict(df)
        return float(prediction[0])

    def predict_batch(self, data_list):
        """
        Make predictions for multiple samples.

        Args:
            data_list: List of dictionaries with feature names as keys

        Returns:
            list: List of predicted cell viabilities (%)
        """
        if not data_list:
            return []

        # Transform each row
        transformed_list = [self._transform_input(data) for data in data_list]
        df = pd.DataFrame(transformed_list)
        df = self._reorder_columns(df)
        predictions = self.model.predict(df)
        return predictions.tolist()

    def predict_concentration_curve(self, base_data, concentrations):
        """
        Generate predictions for a range of concentrations.

        Args:
            base_data: Dictionary with all features except concentration
            concentrations: List of concentration values to predict

        Returns:
            list: List of (concentration, viability) tuples
        """
        results = []
        for conc in concentrations:
            data = base_data.copy()
            data['Concentration (ug/ml)'] = conc
            viability = self.predict(data)
            results.append((conc, viability))
        return results

    def get_feature_contributions(self, data):
        """
        Get feature contributions for a prediction (EBM explainability).

        Args:
            data: Dictionary with feature names as keys

        Returns:
            dict: Feature names mapped to their contributions (with user-friendly names)
        """
        transformed = self._transform_input(data)
        df = self._prepare_dataframe(transformed)

        # Reverse mapping for user-friendly display
        reverse_mapping = {
            'anatase': 'Anatase',
            'rutile': 'Rutile',
            'Concentration (ug/ml)_log': 'Concentration (ug/ml)'
        }

        try:
            # EBM models have explain_local method
            explanation = self.model.explain_local(df)

            # Extract contributions from the explanation
            contributions = {}
            if hasattr(explanation, 'data'):
                local_data = explanation.data(0)
                if local_data and 'names' in local_data and 'scores' in local_data:
                    for name, score in zip(local_data['names'], local_data['scores']):
                        # Use user-friendly name if available
                        display_name = reverse_mapping.get(name, name)
                        contributions[display_name] = float(score)

            return contributions
        except Exception as e:
            # If explain_local is not available, return empty dict
            print(f"Could not get feature contributions: {e}")
            return {}

    def get_global_feature_importance(self):
        """
        Get global feature importance from the model.

        Returns:
            dict: Feature names mapped to their importance scores (with user-friendly names)
        """
        # Reverse mapping for user-friendly display
        reverse_mapping = {
            'anatase': 'Anatase',
            'rutile': 'Rutile',
            'Concentration (ug/ml)_log': 'Concentration (ug/ml)'
        }

        try:
            # EBM models have explain_global method
            explanation = self.model.explain_global()

            importance = {}
            if hasattr(explanation, 'data'):
                global_data = explanation.data()
                if global_data and 'names' in global_data and 'scores' in global_data:
                    for name, score in zip(global_data['names'], global_data['scores']):
                        # Use user-friendly name if available
                        display_name = reverse_mapping.get(name, name)
                        importance[display_name] = float(score)

            return importance
        except Exception as e:
            print(f"Could not get global feature importance: {e}")
            return {}

    def _prepare_dataframe(self, data):
        """
        Prepare a single sample as a DataFrame with correct column order.

        Args:
            data: Dictionary with feature names as keys

        Returns:
            pd.DataFrame: DataFrame with correct column order
        """
        df = pd.DataFrame([data])
        return self._reorder_columns(df)

    def _reorder_columns(self, df):
        """
        Reorder DataFrame columns to match expected feature order.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with reordered columns
        """
        # Only include columns that exist in both the data and expected order
        cols = [col for col in self.FEATURE_ORDER if col in df.columns]
        return df[cols]

    def get_feature_names(self):
        """Get the list of expected feature names."""
        return self.FEATURE_ORDER.copy()
