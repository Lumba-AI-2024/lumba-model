from django.urls import path

from asynctrain.views import async_train_endpoint

urlpatterns = [
    path('', async_train_endpoint),
]