from django.shortcuts import render
from .forms import PredictForm
import numpy as np

def predict(request):
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            # use your trained NLP model to make a prediction
            prediction = np.random.randint(0, 2) # replace this with your model's prediction
            return render(request, 'myapp/result.html', {'prediction': prediction})
    else:
        form = PredictForm()
    return render(request, 'myapp/predict.html', {'form': form})