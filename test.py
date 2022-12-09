import pandas as pd
from keras.models import load_model
import os
from pathlib import Path
def test(output_folder, X_test, y_test, epoch_num=None):
    Y_test = pd.get_dummies(y_test)

    model = load_model(output_folder)
    result=model.evaluate(X_test,Y_test)
    try:
        return {
        'message': 'Model: '+output_folder+' Accuracy: '+str(round(result[1]*100,1))+' Loss: '+str(round(result[0],2))
    }
    except:
        pass