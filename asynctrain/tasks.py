import os

import joblib
import pandas
import requests
import shap
from io import BytesIO
import matplotlib.pyplot as plt
import base64
from django_rq import job
import json


from ml_model.models.linear_regression import LumbaLinearRegression
from ml_model.models.decision_tree_classification import LumbaDecisionTreeClassifier
from ml_model.models.decision_tree_regression import LumbaDecisionTreeRegressor
from ml_model.models.xg_boost_regression import LumbaXGBoostRegressor
from ml_model.models.xg_boost_classification import LumbaXGBoostClassifier
from ml_model.models.random_forest_regression import LumbaRandomForestRegressor
from ml_model.models.random_forest_classification import LumbaRandomForestClassifier
from ml_model.models.neural_network_regression import LumbaNeuralNetworkRegression
from ml_model.models.neural_network_classification import LumbaNeuralNetworkClassification
from ml_model.models.kmeans import LumbaKMeans
from ml_model.models.db_scan import LumbaDBScan
from modeling.settings import BACKEND_API_URL



def calculate_shap_values(best_model, X, model_type, X_train=None, X_test=None):
    if model_type == "classification" or model_type == "rf":
        explainer = shap.Explainer(best_model)
        shap_values = explainer.shap_values(X_test)
    elif model_type == "regression" or model_type == "xgboost":
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
    elif model_type == "neural_network":
        # Summarize the background data using shap.sample
        background = shap.sample(X_train, 50)

        # Create a SHAP explainer using the summarized background
        explainer = shap.KernelExplainer(best_model.predict, background)  # Use the first 50 instances as the background

        # Compute SHAP values for the test set
        
        shap_values = explainer.shap_values(X_test.iloc[:50, :], nsamples=500)
    else:
        raise ValueError("Unsupported model type")

    # Generate SHAP summary plot
    plt.figure()
    if model_type != "neural_network":
        shap.summary_plot(shap_values, X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test.iloc[:50, :], show=False)
        
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_str

@job('default', timeout=86400)
def asynctrain(model_metadata):
    url = BACKEND_API_URL + '/modeling/'

    print(model_metadata)

    df = pandas.read_csv(model_metadata['dataset_link'])
    print(df.head())
    df = df.drop(columns = ['Unnamed: 0'])

    requests.put(url,
                 params={
                     'modelname': model_metadata['modelname'],
                     'datasetname': model_metadata['datasetname'],
                     'workspace': model_metadata['workspace'],
                     'username': model_metadata['username']
                 },
                 data={'status': 'in progress'}
                 )
    # print("training with record id " + current_task.id + " in progress")

    # train model
    print("inii",model_metadata)
    model_type = ""
    if model_metadata['method'] == 'REGRESSION':
        if model_metadata['algorithm'] == 'LINEAR':
            LR = LumbaLinearRegression(df)
            response = LR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = {
                "r2_score": response["r2_score"],
                "mae": response["mean_absolute_error"],
                "mse": response["mean_squared_error"]
            }
            model_metadata["model"] = response["model"]
            model_type = "regression"
        if model_metadata['algorithm'] == 'DECISION_TREE':
            DTR = LumbaDecisionTreeRegressor(df)
            print("masukk")
            response = DTR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = {
                "r2_score": response["r2_score"],
                "mae": response["mean_absolute_error"],
                "mse": response["mean_squared_error"]
            }
            model_metadata["model"] = response["model"]
            model_type = "regression"
        if model_metadata['algorithm'] == 'RANDOM_FOREST':
            RFR = LumbaRandomForestRegressor(df)
            response = RFR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = {
                "r2_score": response["r2_score"],
                "mae": response["mean_absolute_error"],
                "mse": response["mean_squared_error"]
            }
            model_metadata["model"] = response["model"]
            model_type = "rf"
        if model_metadata['algorithm'] == 'NEURAL_NETWORK':
            NNR = LumbaNeuralNetworkRegression(df)
            response = NNR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = {
                "r2_score": response["r2_score"],
                "mae": response["mean_absolute_error"],
                "mse": response["mean_squared_error"]
            }
            model_metadata["model"] = response["model"]
            model_type = "neural_network"
        if model_metadata['algorithm'] == 'XG_BOOST':
            XBR = LumbaXGBoostRegressor(df)
            response = XBR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = {
                "r2_score": response["r2_score"],
                "mae": response["mean_absolute_error"],
                "mse": response["mean_squared_error"]
            }
            model_metadata["model"] = response["model"]
            model_type == "xgboost"

    if model_metadata['method'] == 'CLASSIFICATION':
        if model_metadata['algorithm'] == 'DECISION_TREE':
            DT = LumbaDecisionTreeClassifier(df)
            response = DT.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = {
                "accuracy_score": response["accuracy_score"],
                "recall_score": response["recall_score"],
                "precision_score": response["precision_score"],
                "f1_score": response["f1_score"]
            }
            model_metadata["model"] = response["model"]
            model_type = "classification"
        if model_metadata['algorithm'] == 'NEURAL_NETWORK':
            NNC = LumbaNeuralNetworkClassification(df)
            response = NNC.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = {
                "accuracy_score": response["accuracy_score"],
                "recall_score": response["recall_score"],
                "precision_score": response["precision_score"],
                "f1_score": response["f1_score"]
            }
            model_metadata["model"] = response["model"]
            model_type = "neural_network"
        if model_metadata['algorithm'] == 'RANDOM_FOREST':
            RFC = LumbaRandomForestClassifier(df)
            response = RFC.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = {
                "accuracy_score": response["accuracy_score"],
                "recall_score": response["recall_score"],
                "precision_score": response["precision_score"],
                "f1_score": response["f1_score"]
            }
            model_metadata["model"] = response["model"]
            model_type = "classification"
        if model_metadata['algorithm'] == 'XG_BOOST':
            XGC = LumbaXGBoostClassifier(df)
            response = XGC.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = {
                "accuracy_score": response["accuracy_score"],
                "recall_score": response["recall_score"],
                "precision_score": response["precision_score"],
                "f1_score": response["f1_score"]
            }
            model_type == "xgboost"

    if model_metadata['method'] == 'CLUSTERING':
        model_type = "classification"
        if model_metadata['algorithm'] == 'KMEANS':
            KM = LumbaKMeans(df)
            response = KM.train_model()
            model_metadata["metrics"] = "silhouette_score"
            model_metadata["score"] = response["silhouette_score"]
            model_metadata["model"] = response["model"]
            model_metadata["shap_model"] = response["shap_model"]
        if model_metadata['algorithm'] == 'DBSCAN':
            DB = LumbaDBScan(df)
            response = DB.train_model()
            model_metadata["metrics"] = "silhouette_score"
            model_metadata["score"] = response["silhouette_score"]
            model_metadata["model"] = response["model"]
            model_metadata["shap_model"] = response["shap_model"]


    model_saved_name = f"{model_metadata['modelname']}.pkl"
    joblib.dump(response['model'], model_saved_name)
    model_metadata["score"] = json.dumps(model_metadata["score"])
    print(model_metadata["score"])

    if model_metadata['method'] == 'CLASSIFICATION' or model_metadata['method'] == 'REGRESSION' :
        shap_values = calculate_shap_values(model_metadata["model"], df.drop(columns=[model_metadata['target']]), model_type, X_train=response["X_train"], X_test=response["X_test"])
        model_metadata['shap_values'] = shap_values
    else:
        shap_values = calculate_shap_values(model_metadata["shap_model"], df, X_test=df)
    # save model to pkl format
    print(model_saved_name)
    requests.put(url,
                 params={
                     'modelname': model_metadata['modelname'],
                     'datasetname': model_metadata['datasetname'],
                     'workspace': model_metadata['workspace'],
                     'username': model_metadata['username'],
                 },
                 data={
                     **model_metadata,
                     'status': 'completed',
                 },
                 files={
                     'model_file': open(model_saved_name, 'rb')
                 }
                 )
    # os.remove(model_saved_name)
    # print("training with record id " + current_task.id + " completed")
    print(model_metadata)
    model_metadata.pop('model', None)
    return model_metadata
