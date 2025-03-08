import pandas as pd
import shap


from preprocessing import get_recreational_error, process_data
data_loaded = pd.read_csv('data/datathon_data.csv')
data_processed = process_data(data_loaded, return_label=False)

# Set the number of background samples for summarization
K = 20  # adjust as needed

# Option 1: Summarize with k-means
background_data = shap.kmeans(data_processed, K)
anomal = data_loaded[data_loaded['label'] == 'anomal']
normal = data_loaded[data_loaded['label'] == 'normal']
anomal_sample = anomal.sample(frac=0.1, random_state=0)
normal_sample = normal.sample(frac = 0.01, random_state=0)
background_data_non_cat = pd.concat([anomal_sample, normal_sample])


def get_shap_explainer(sample):
    explainer = shap.KernelExplainer(lambda x: get_recreational_error(pd.DataFrame(x, columns=data_processed.columns)), background_data)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample)
    return shap_values


def error(sample):
    return get_recreational_error(process_data(pd.DataFrame(sample, columns=data_loaded.columns), return_label=False))
def get_shap_explainer_non_cat(sample):
    explainer = shap.KernelExplainer(error, background_data_non_cat)
    shap_values = explainer.shap_values(sample)
    return shap_values