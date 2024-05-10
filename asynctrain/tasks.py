import joblib
import pandas
import requests
from celery import shared_task
from minio import Minio

from ml_model.models.decision_tree import LumbaDecisionTreeClassifier
from ml_model.models.linear_regression import LumbaLinearRegression


@shared_task
def asynctrain(model_metadata):
    # update training record to 'in progress'
    # TODO: commented out for dev
    url = 'http://127.0.0.1:8000/modeling/'

    print(model_metadata)

    minio_client = Minio("34.101.59.56:9000",
                         access_key="zl6ggTd5WUAaV2NMaGJj",
                         secret_key="mtUHWqwV2GlpW8eALQ0quZEWCHkZqQlbBAXKuXus",
                         secure=False)

    obj = minio_client.get_object('lumba-directory',
                                  model_metadata['dataset_link'])

    df = pandas.read_csv(obj)


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
                     **model_metadata,
                     'status': 'completed',
                 },
                 files={
                     'model_file': open(model_saved_name, 'rb')
                 }
                 )
    # os.remove(model_saved_name)
    # print("training with record id " + current_task.id + " completed")
    return model_metadata
