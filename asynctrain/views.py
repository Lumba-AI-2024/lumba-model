import django_rq
from django.http import JsonResponse

from asynctrain.tasks import asynctrain



def async_train_endpoint(request):
    """
        Input the entire
    """
    print(request.POST.dict())
    try:
        model_metadata = request.POST.dict()
    except:
        return JsonResponse({'message': "input error"}, status=400)

    if (model_metadata['algorithm'] == 'XG_BOOST') :
        queue = django_rq.get_queue('xgboost')
        queue.enqueue(asynctrain, model_metadata)
    else:
        async_result = asynctrain.delay(model_metadata)

    # print(async_result)
    response_data = {
        'task_id': async_result.id,
        'status': 'Processing'
    }
     
    return JsonResponse(response_data, status=200)
