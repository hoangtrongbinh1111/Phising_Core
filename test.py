import pandas as pd
from keras.models import load_model
import os
from pathlib import Path
def test(output_folder, X_test, y_test, epoch_num=None):
    Y_test = pd.get_dummies(y_test)
    for x in os.listdir(output_folder):
        if Path(x).suffix == '.hdf5':
            if epoch_num==int(x.split('.')[-2].split('-')[-1]):
                model_path= str(x)
                break
            else:
                continue

    model = load_model(output_folder+'/'+model_path)
    result=model.evaluate(X_test,Y_test)
   

    try:
        return {
        'message': 'Model: '+model_path+' Accuracy: '+str(round(result[1]*100,1))+' Loss: '+str(round(result[0],2))
    }
    except:
        pass