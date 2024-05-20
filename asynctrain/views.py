import asyncio

import pandas
from celery import shared_task
from django.http import JsonResponse

from asynctrain.tasks import asynctrain



def async_train_endpoint(request):
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

    # print(model_metadata)

    async_result  = asynctrain.delay(model_metadata)
    # print(async_result)
    response_data = {
        'task_id': async_result.id,
        'status': 'Processing'
    }
     
    return JsonResponse(response_data, status=200)
