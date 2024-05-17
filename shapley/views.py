import pandas as pd
import shap
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

@csrf_exempt
async def receive_files(request):
    if request.method == 'POST':
        try:
            model_file = request.FILES['model_file']
            data_file = request.FILES['data_file']
        except KeyError:
            return JsonResponse({'error': 'Files are missing from the request'}, status=400)

        # Save files temporarily
        model_path = default_storage.save(model_file.name, model_file)
        data_path = default_storage.save(data_file.name, data_file)

        # Call the asynchronous function to process SHAP values
        response = await process_shap_values(model_path, data_path)

        # Clean up: remove the files after processing
        default_storage.delete(model_path)
        default_storage.delete(data_path)

        return JsonResponse(response)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)

async def process_shap_values(model_path, data_path):
    # Load the model
    model = joblib.load(model_path)

    # Load the dataset
    data = pd.read_csv(data_path)
    if 'target' in data.columns:
        X = data.drop('target', axis=1)
        y = data['target']
    else:
        return {'error': 'No target column found in dataset'}

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Get summary plot as an image or data (depending on your requirement)
    shap_sum = shap.summary_plot(shap_values, X, show=False)
    
    # Save the plot to a file and send the file name back to the client
    plot_path = 'shap_summary_plot.png'
    shap.save_display_data(shap_sum, plot_path)

    return {
        'message': 'SHAP values processed successfully',
        'plot_path': plot_path
    }
