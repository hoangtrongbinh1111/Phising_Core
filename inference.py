import pandas as pd
from keras.models import load_model
import os
from pathlib import Path
from preprocess import inference_preprocess
import time
def inference(labId, data_path, epoch_num=1):
    output_folder = f'./modelDir/{labId}/log_train/checkpoint-{epoch_num}.hdf5'
    model = load_model(output_folder)
    output_data=inference_preprocess(data_path)
    start_time=time.time()
    result=model.predict(output_data.get('dataset'))
    end_time=time.time()
    try:
        return {
        'message': output_data.get('message')+'\Checkpoint epoch {epoch_num}: \nInference time: %2.1fs' %(end_time-start_time),
        'inference_result':result
    }
    except: 
        pass