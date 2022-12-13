import pandas as pd
from preprocess import train_test_preprocess
from preprocess import inference_preprocess
from train import train
from test import test
from inference import inference
import os
import sys
import joblib
BEST_EPOCH_NUM = 2
OUTPUT_FOLDER = './modelDir/0cb3c269-5ffc-4e49-bb36-a1d5cfcce7b7/log_train/model-sequential-epoch-{BEST_EPOCH_NUM}.hdf5'

async def demoTrain(data):

   
    datasetId = data['datasetId']
    preprocess_output = joblib.load(f'../datasetDir/{datasetId}/uploads/preprocess_output.pkl')

    X_train = preprocess_output.get('X_train')
    y_train = preprocess_output.get('y_train')
    model_type = preprocess_output.get('model_type')

    print('===============================TRAINING...===================================')
    train_output = train(X_train, y_train, epoch_num=data['train_num_epoch'],
                         batch_size=data['train_batch_size'], model_type=model_type, labId=data['labId'])
    try:
        for res_per_epoch in train_output:
            if res_per_epoch:
                yield res_per_epoch

    except:
        print("error")

    print('===============================TRAINED===================================')

async def demoTest(data):
   
    datasetId = data['datasetId']
    preprocess_output = joblib.load(f'../datasetDir/{datasetId}/uploads/preprocess_output.pkl')

    print('===============================TESTING...===================================')
    X_test = preprocess_output.get('X_test')
    y_test = preprocess_output.get('y_test')
    test_output=test(labId=data['labId'],X_test=X_test,y_test=y_test,epoch_num=data['epoch_selected'])

    try:
        return test_output.get('message')
    except: 
        print("error")
    print('===============================TESTED===================================')

async def demoInfer(data):
    print('===============================PREPROCESSING...===================================')
    if 'preprocess_output.pkl' not in os.listdir():
        preprocess_output = inference_preprocess(
            data_path='./data.csv', feature_set = data['feature_set'])
        joblib.dump(preprocess_output, filename='preprocess_output.pkl')
    else:
        preprocess_output = joblib.load('preprocess_output.pkl')
        print('Load from previous preprocessed_output.pkl')
        print(preprocess_output.get('message'))


    print('===============================PREPROCESSED===================================')


    print('===============================INFERENCING...===================================')
    inference_output=inference(labId=data['labId'], data_path='./data.csv', epoch_num=data['epoch_selected'])

    try:
        return {
           "message": inference_output.get('message'),
           "result" : inference_output.get('inference_result')
        }
    except:
        print('error')
    print('===============================INFERENCED===================================')


async def preProcess(data):
    print('===============================PREPROCESSING...===================================')
    preprocess_output = train_test_preprocess(data_path=data['data_path'])        
    datasetId = data['datasetId']
    dirname = os.path.dirname(__file__)
    output_folder = f'./datasetDir/{datasetId}/uploads'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.join(dirname, output_folder)
    joblib.dump(preprocess_output, filename )
    return_value = output_folder +'/preprocess_output.pkl'
    print(return_value)
    print('===============================PREPROCESSED===================================')
    return {
        "datasetId":data['datasetId'],
        "savePath":return_value
    }