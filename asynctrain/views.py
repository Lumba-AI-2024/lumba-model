import asyncio
import json
import os

import pandas
from django.http import JsonResponse

from ml_model.models.linear_regression import LumbaLinearRegression
from ml_model.models.decision_tree import LumbaDecisionTreeClassifier
import requests
import joblib


async def asynctrain(df, model_metadata):
    # update training record to 'in progress'
    # TODO: commented out for dev
    url = 'http://127.0.0.1:8000/modeling/'
    print(model_metadata)
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
    if model_metadata['method'] == 'REGRESSION':
        if model_metadata['algorithm'] == 'LINEAR':
            LR = LumbaLinearRegression(df)
            response = LR.train_model(train_column_name=model_metadata['feature'].split(','),
                                      target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "mean_absolute_error"
            model_metadata["score"] = response["mean_absolute_error"]
    if model_metadata['method'] == 'CLASSIFICATION':
        if model_metadata['algorithm'] == 'DECISION_TREE':
            DT = LumbaDecisionTreeClassifier(df)
            response = DT.train_model(train_column_names=model_metadata['feature'].split(','),
                                      target_column_name=model_metadata['target'])
            model_metadata["metrics"] = "accuracy_score"
            model_metadata["score"] = response["accuracy_score"]

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
                     'status': 'completed',
                 },
                 files={
                     'model_file': open(model_saved_name, 'rb')
                 }
                 )
    # os.remove(model_saved_name)
    # print("training with record id " + current_task.id + " completed")
    return model_metadata


# this function will return record in json
# {'id': 5, 'status': 'accepted'}
async def async_train_endpoint(request):
    """
        Input the entire
    """
    print(request.POST.dict())
    try:
        model_metadata = request.POST.dict()
        # TODO: get the file from request, or get them from minio
        # _file = request.FILES['file']
    except:
        return JsonResponse({'message': "input error"}, status=400)

    print(model_metadata)

    df = pandas.read_csv(request.FILES['file'])

    await asyncio.gather(asynctrain(df, model_metadata))
    return JsonResponse(model_metadata, status=200)
