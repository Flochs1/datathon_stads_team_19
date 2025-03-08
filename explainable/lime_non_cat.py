import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from preprocessing import get_recreational_error, process_data

data_loaded = pd.read_csv('data/datathon_data.csv')
data = data_loaded.copy()
train_data = process_data(data, return_label=False)

def get_explainer(sample):
    sample = sample.astype(np.float64)
    """
    Generates a LIME explanation for the given sample using the get_recreational_error scoring function.

    Parameters:
        sample (pd.Series, np.ndarray, or similar):
            A single instance that you want to explain.
        training_data (pd.DataFrame):
            The DataFrame used as training data for normalization and to supply the column names.

    Returns:
        A LIME explanation object. You can visualize the explanation (for example, using .show_in_notebook()).
    """
    #sample = sample[categorical_attr_names + ['DMBTR', 'WRBTR']]
    # Initialize the LIME Tabular Explainer
    explainer = LimeTabularExplainer(
        training_data=np.array(train_data),  # Convert training data for internal normalization
        feature_names=train_data.columns,
        categorical_features=[i for i, col in enumerate(train_data.columns) if col not in ['DMBTR', 'WRBTR']],
        categorical_names={i: [0,1] for i, col in enumerate(train_data.columns) if col not in ['DMBTR', 'WRBTR']},
        mode='regression',
        random_state=42
    )

    # Define the scoring function that LIME will use. This function takes a batch of samples (2D array)
    # and returns the corresponding error scores using your get_recreational_error function.
    def score_fn(sample):
        sample = pd.DataFrame(sample, columns=train_data.columns)
        return get_recreational_error(sample)

    # Generate and return the explanation for the given sample
    exp = explainer.explain_instance(np.array(sample)[0], score_fn, num_features=10)
    return exp