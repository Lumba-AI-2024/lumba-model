import asyncio
import json
import os
from django.http import JsonResponse
import pandas as pd
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
import requests
import joblib

async def asynctrain(df, training_record, model_metadata):

	# update training record to 'in progress'
	# TODO: commented out for dev
	# url = 'http://127.0.0.1:8000/modeling/updaterecord/'
	# json = {'id': training_record['id'], 'status':'in progress'}
	# record = requests.post(url, json=json)
	print("training with record id "+ str(training_record['id']) + " in progress")

	print(model_metadata)

	# train model
	if model_metadata['method'] == 'REGRESSION':
		if model_metadata['algorithm'] == 'LINEAR':
			LRR = LumbaLinearRegression(df)
			response = LRR.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "r2_score"
			model_metadata["score"] = response["r2_score"]
		if model_metadata['algorithm'] == 'DECISION_TREE':
			DTR = LumbaDecisionTreeRegressor(df)
			response = DTR.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "r2_score"
			model_metadata["score"] = response["r2_score"]
		if model_metadata['algorithm'] == 'RANDOM_FOREST':
			RFR = LumbaRandomForestRegressor(df)
			response = RFR.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "r2_score"
			model_metadata["score"] = response["r2_score"]
		if model_metadata['algorithm'] == 'NEURAL_NETWORK':
			NNR = LumbaNeuralNetworkRegression(df)
			response = NNR.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "r2_score"
			model_metadata["score"] = response["r2_score"]		
		if model_metadata['algorithm'] == 'XG_BOOST':
			XBR = LumbaXGBoostRegressor(df)
			response = XBR.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "r2_score"
			model_metadata["score"] = response["r2_score"]		
	
	if model_metadata['method'] == 'CLASSIFICATION':
		if model_metadata['algorithm'] == 'DECISION_TREE':
			DTC = LumbaDecisionTreeClassifier(df)
			response = DTC.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "accuracy_score"
			model_metadata["score"] = response["accuracy_score"]
		if model_metadata['algorithm'] == 'NEURAL_NETWORK':
			NNC = LumbaNeuralNetworkClassification(df)
			response = NNC.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "accuracy_score"
			model_metadata["score"] = response["accuracy_score"]
		if model_metadata['algorithm'] == 'RANDOM_FOREST':
			RFC = LumbaRandomForestClassifier(df)
			response = RFC.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "accuracy_score"
			model_metadata["score"] = response["accuracy_score"]
		if model_metadata['algorithm'] == 'XG_BOOST':
			XGC = LumbaXGBoostClassifier(df)
			response = XGC.train_model(target_column_name=model_metadata['target'])
			model_metadata["metrics"] = "accuracy_score"
			model_metadata["score"] = response["accuracy_score"]

	if model_metadata['method'] == 'CLUSTER':
		if model_metadata['algorithm'] == 'KMEANS':
			KM = LumbaKMeans(df)
			response = KM.train_model()
			model_metadata["metrics"] = "silhouette_score"
			model_metadata["score"] = response["silhouette_score"]
			# model_metadata["labels"] = response["labels_predicted"]
		if model_metadata['algorithm'] == 'DB_SCAN':
			DB = LumbaDBScan(df)
			response = DB.train_model()
			model_metadata["metrics"] = "silhouette_score"
			model_metadata["score"] = response["silhouette_score"]
			# model_metadata["labels"] = response["labels_predicted"]

	# save model to pkl format
	model_saved_name = f"{model_metadata['model_name']}.pkl"
	joblib.dump(response['model'], model_saved_name)

	# save the pkl in the model_metadata
	model_metadata['model'] = model_saved_name
	

	# save model
	# TODO: commented out for dev
	# url = 'http://127.0.0.1:8000/modeling/save/'
	# requests.post(url, data=model_metadata, files={'file': open(model_saved_name, 'rb')})

	# update training record to 'completed'
	# TODO: commented out for dev
	# url = 'http://127.0.0.1:8000/modeling/updaterecord/'
	# json = {'id': training_record['id'], 'status':'completed'}
	# record = requests.post(url, json=json)
	os.remove(model_saved_name)
	print("training with record id "+ str(training_record['id']) + " completed")
	return model_metadata

# this function will return record in json 
# {'id': 5, 'status': 'accepted'}
async def async_train_endpoint(request):
	try:
		model_metadata = request.POST.dict()
		file = request.FILES['file']
	except:
		return JsonResponse({'message': "input error"})
	df = pd.read_csv(file)

	# create training record in main service db
	# TODO: commented out for dev
	# url = 'http://127.0.0.1:8000/modeling/createrecord/'
	# json = {'status':'accepted'}
	# record = requests.post(url, json=json)

	training_record = {
		# 'id' : record.json()['id'],
		# 'status' : record.json()['status'],

		# TODO: commented out for dev
		'id': '0001',
		'status': 'this is a dummy',
	}
	result = await asyncio.gather(asynctrain(df, training_record, model_metadata))
	return JsonResponse(result, safe=False)
