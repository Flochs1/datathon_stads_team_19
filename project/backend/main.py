# backend/main.py

from fastapi import FastAPI, HTTPException
import pandas as pd

# For decision tree explanation, we use scikit-learn
from sklearn.tree import DecisionTreeClassifier, export_text

from explainable.integrated_gradients import integrated_gradients, autoencoder_model, get_top_attributions
from explainable.lime import get_explainer
from explainable.shappy import get_shap_explainer_non_cat
from project.backend.preprocessing.preprocessing import process_data, get_labels

data_loaded = pd.read_csv('data/datathon_data.csv')
data_transformed = process_data(data_loaded, return_label=False)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}




def get_df_from_dict(data):
    ser = pd.Series(data)
    return ser


# Endpoint for Integrated Gradients
@app.post("/xai/integrated_gradients")
def integrated_gradients_endpoint(sample):
    try:
        sample_df = get_df_from_dict(sample['data'])
        attributions = integrated_gradients(autoencoder_model, process_data(sample_df, return_label=False), steps=50)
        top_features = pd.DataFrame(get_top_attributions(attributions, 10))
        top_features.index = data_transformed.columns[top_features.index]
        top_features.columns = ['attribution']
        return top_features['attribution'].to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Endpoint for LIME Explanation
@app.post("/xai/lime")
def lime_explanation_endpoint(sample):
    #try:
    sample_df = get_df_from_dict(sample['data'])
    # Get the LIME explainer (this uses your get_explainer function from lime.py)
    explainer = get_explainer(sample_df)
    # Convert explanation to a list of (feature, weight) tuples
    explanation = explainer.as_list()
    explanation = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)
    explanation = pd.DataFrame(explanation, columns=['index', 'weight'])
    explanation.set_index('index', inplace=True)
    return explanation['weight'].to_dict()
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))

# Endpoint for SHAP Explanation
@app.post("/xai/shap")
def shap_explanation_endpoint(sample):
    #try:
        sample_df = get_df_from_dict(sample['data'])
        print(sample_df)
        print(data_loaded.iloc[102552])


        shap = get_shap_explainer_non_cat(sample_df)
        #shap = get_shap_explainer_non_cat(data_loaded.iloc[int(sample_df['BELNR'])])
        ser = pd.Series(shap, index=sample_df.index).sort_values(key = lambda x: x.abs(), ascending=False)
        df = pd.DataFrame(ser, columns = ['shap_values'])
        return df['shap_values'].to_dict()
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Decision Tree Explanation (demo implementation)
"""
@app.post("/xai/decision_tree")
def decision_tree_explanation_endpoint(sample: SampleData):
    try:
        df = pd.DataFrame([sample.data])
        # For demonstration, we assume that df has numeric features.
        X = df.copy()
        # Create a dummy target (e.g. binary based on threshold of the first feature)
        col = X.columns[0]
        y = (X[col].astype(float) > X[col].astype(float).mean()).astype(int)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        tree_rules = export_text(clf, feature_names=list(X.columns))
        return {"decision_tree_explanation": tree_rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
if __name__ == "__main__":
    data1 = {"BELNR": "170320", "WAERS": "C1", "BUKRS": "C11", "KTOSL": "C1", "PRCTR": "C30", "BSCHL": "A1",
             "HKONT": "B1", "DMBTR": "910645.965114", "WRBTR": "54452.8097799", 'label': 'anormal'}
    data2 = {"BELNR": "370090", "WAERS": "E00", "BUKRS": "W67", "KTOSL": "W79", "PRCTR": "T97", "BSCHL": "N82",
             "HKONT": "J10", "DMBTR": "92445516.8528", "WRBTR": "59585037.0177", 'label': 'anormal'}
    int_gradient = integrated_gradients_endpoint({"data": data2})
    print(int_gradient)
    lime = lime_explanation_endpoint({'data': data2})
    print(lime)
    shap = shap_explanation_endpoint({'data': data2})
    print(shap)
    #import uvicorn
    #uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


