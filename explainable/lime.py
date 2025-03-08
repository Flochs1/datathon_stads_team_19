import numpy as np
import pandas as pd
import sklearn
from lime.lime_tabular import LimeTabularExplainer

from preprocessing import get_recreational_error, process_data
data_loaded = pd.read_csv('data/datathon_data.csv')
data = data_loaded.copy()

# List of categorical attribute names (as used in your training DataFrame)
categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT']

data = data[categorical_attr_names + ['DMBTR', 'WRBTR']]
cat_feature_indices = [data.columns.get_loc(col) for col in categorical_attr_names]
categorical_names = {}
categorical_encoders = {}
for feature in cat_feature_indices:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_
    categorical_encoders[feature] = le
def get_explainer(sample):
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

    sample = sample[categorical_attr_names + ['DMBTR', 'WRBTR']]
    for i in cat_feature_indices:
        sample.iloc[i] = categorical_encoders[i].transform([sample.iloc[i]])[0]
    for col in ['DMBTR', 'WRBTR']:
        sample[col] = sample[col].astype(np.float64)
    #sample = sample[categorical_attr_names + ['DMBTR', 'WRBTR']]
    # Initialize the LIME Tabular Explainer
    explainer = LimeTabularExplainer(
        training_data=np.array(data),  # Convert training data for internal normalization
        feature_names=data.columns,
        categorical_features=cat_feature_indices ,
        categorical_names=categorical_names,
        mode='regression',
        random_state=42
    )

    # Define the scoring function that LIME will use. This function takes a batch of samples (2D array)
    # and returns the corresponding error scores using your get_recreational_error function.
    def score_fn(sample):
        decoded_sample = sample.copy()
        # Loop over each categorical feature index
        df = pd.DataFrame(decoded_sample, columns=data.columns)
        for col in categorical_attr_names:
            idx = data.columns.get_loc(col)
            # Convert values to int
            col_vals = decoded_sample[:, idx].astype(int)
            # Clip values to valid range (if necessary)
            col_vals = np.clip(col_vals, 0, len(categorical_encoders[data.columns.get_loc(col)].classes_) - 1)
            # Apply inverse transform to the entire column
            inverse = categorical_encoders[data.columns.get_loc(col)].inverse_transform(col_vals)
            df[col] = pd.Series(inverse)
        return get_recreational_error(process_data(df, return_label=False))

    # Generate and return the explanation for the given sample
    exp = explainer.explain_instance(sample, score_fn, num_features=5)
    return exp

