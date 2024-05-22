import joblib
import pandas
import requests
import shap
from celery import shared_task
from minio import Minio

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

def calculate_shap_values(best_model, X_test):
		explainer = shap.TreeExplainer(best_model)
		shap_values = explainer.shap_values(X_test)
		
		# Convert SHAP values to a format that can be returned as JSON
		shap_values_json = [shap_values[i].tolist() for i in range(len(shap_values))]
		
		return shap_values_json
@shared_task
def asynctrain(model_metadata):
    url = 'http://127.0.0.1:8000/modeling/'

    print(model_metadata)

    df = pandas.read_csv(model_metadata['dataset_link'])

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
    if model_metadata['method'] == 'REGRESSION':
        if model_metadata['algorithm'] == 'LINEAR':
            LR = LumbaLinearRegression(df)
            response = LR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = response["r2_score"]
            model_metadata["model"] = response["model"]
        if model_metadata['algorithm'] == 'DECISION_TREE':
            DTR = LumbaDecisionTreeRegressor(df)
            print("masukk")
            response = DTR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = response["r2_score"]
            model_metadata["model"] = response["model"]
        if model_metadata['algorithm'] == 'RANDOM_FOREST':
            RFR = LumbaRandomForestRegressor(df)
            response = RFR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = response["r2_score"]
            model_metadata["model"] = response["model"]
        if model_metadata['algorithm'] == 'NEURAL_NETWORK':
            NNR = LumbaNeuralNetworkRegression(df)
            response = NNR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = response["r2_score"]	
            model_metadata["model"] = response["model"]	
        if model_metadata['algorithm'] == 'XG_BOOST':
            XBR = LumbaXGBoostRegressor(df)
            response = XBR.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "r2_score"
            model_metadata["score"] = response["r2_score"]	
            model_metadata["model"] = response["model"]	
            
    if model_metadata['method'] == 'CLASSIFICATION':
        if model_metadata['algorithm'] == 'DECISION_TREE':
            DT = LumbaDecisionTreeClassifier(df)
            response = DT.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = response["accuracy_score"]
            model_metadata["model"] = response["model"]
        if model_metadata['algorithm'] == 'NEURAL_NETWORK':
            NNC = LumbaNeuralNetworkClassification(df)
            response = NNC.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = response["accuracy_score"]
            model_metadata["model"] = response["model"]
        if model_metadata['algorithm'] == 'RANDOM_FOREST':
            RFC = LumbaRandomForestClassifier(df)
            response = RFC.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = response["accuracy_score"]
            model_metadata["model"] = response["model"]
        if model_metadata['algorithm'] == 'XG_BOOST':
            XGC = LumbaXGBoostClassifier(df)
            response = XGC.train_model(target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = response["accuracy_score"]
            model_metadata["model"] = response["model"]

    if model_metadata['method'] == 'CLUSTER':
        if model_metadata['algorithm'] == 'KMEANS':
            KM = LumbaKMeans(df)
            response = KM.train_model(train_column_names=model_metadata['feature'].split(','))
            model_metadata["metrics"] = "silhouette_score"
            model_metadata["score"] = response["silhouette_score"]
            model_metadata["labels"] = response["labels_predicted"]
        if model_metadata['algorithm'] == 'DB_SCAN':
            DB = LumbaDBScan(df)
            response = DB.train_model(train_column_names=model_metadata['feature'].split(','))
            model_metadata["metrics"] = "silhouette_score"
            model_metadata["score"] = response["silhouette_score"]
            model_metadata["labels"] = response["labels_predicted"]

    shap_values = calculate_shap_values(model_metadata["model"], df.drop(columns=[model_metadata['target']]))
    model_metadata['shap_values'] = shap_values
    # save model to pkl format
    model_saved_name = f"{model_metadata['modelname']}.pkl"
    joblib.dump(response['model'], model_saved_name)
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
    return model_metadata
