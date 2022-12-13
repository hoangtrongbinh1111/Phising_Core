import pandas as pd
from keras.models import load_model
import os
from pathlib import Path
def test(labId, X_test, y_test, epoch_num=0):
    Y_test = pd.get_dummies(y_test)
    output_folder = f'./modelDir/{labId}/log_train/checkpoint-{epoch_num}.hdf5'
    model = load_model(output_folder)
    result=model.evaluate(X_test,Y_test)

    try:
        return {
        'message': 'Checkpoint epoch '+epoch_num+'====> Accuracy: '+str(round(result[1]*100,1))+' Loss: '+str(round(result[0],2))
    }
    except:
        pass