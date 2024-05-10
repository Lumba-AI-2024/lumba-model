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

    print(model_metadata)

    asynctrain.delay(model_metadata)
    return JsonResponse(model_metadata, status=200)
