import pandas as pd
from keras.models import load_model
import os
from pathlib import Path
from preprocess import inference_preprocess
import time
def inference(output_folder, data_path, epoch_num=None):
    for x in os.listdir(output_folder):
        if Path(x).suffix == '.hdf5':
            if epoch_num==int(x.split('.')[-2].split('-')[-1]):
                model_path= str(x)
                break
            else:
                continue

    model = load_model(os.path.join(output_folder,model_path))
    output_data=inference_preprocess(data_path)
    start_time=time.time()
    result=model.predict(output_data.get('dataset'))
    end_time=time.time()
    return {
        'message': output_data.get('message')+'\nModel: '+model_path+'\nInference time: %2.1fs' %(end_time-start_time),
        'inference_result':result
    }