import pandas as pd
from preprocess import train_preprocess
from preprocess import inference_preprocess
from train import train
from test import test
from inference import inference
import os
import joblib
BEST_EPOCH_NUM = 2
OUTPUT_FOLDER = './modelDir/0cb3c269-5ffc-4e49-bb36-a1d5cfcce7b7/log_train/model-sequential-epoch-{BEST_EPOCH_NUM}.hdf5'

async def demoTrain(data):
    # print('===============================PREPROCESSING...===================================')
    # preprocess_output=train_preprocess(data_path=DATA_PATH,model_type="sequential",test_size=0.3)
    # print(preprocess_output.get('message'))
    # print('===============================PREPROCESSED===================================')
    print('===============================PREPROCESSING...===================================')
    if 'preprocess_output.pkl' not in os.listdir():
        preprocess_output = train_preprocess(
            data_path=data['data_path'], model_type=data['model_type'], test_size=data['test_size'], number_records=data['number_records'])
        joblib.dump(preprocess_output, filename='preprocess_output.pkl')
    else:
        preprocess_output = joblib.load('preprocess_output.pkl')
        print('Load from previous preprocessed_output.pkl')
    print(preprocess_output.get('message'))
    print('===============================PREPROCESSED===================================')
    X_train = preprocess_output.get('X_train')
    y_train = preprocess_output.get('y_train')
    model_type = preprocess_output.get('model_type')
    # yield {
    #     "x_test": X_test,
    #     "y_test": y_test
    # }

    print('===============================TRAINING...===================================')
    train_output = train(X_train, y_train, epoch_num=data['train_num_epoch'],
                         batch_size=data['train_batch_size'], model_type=model_type, labId=data['labId'])
    try:
        for res_per_epoch in train_output:
            if res_per_epoch:
                # print('1')
                # print(res_per_epoch)
                yield res_per_epoch

    except:
        print("error")

    # print(train_output.get('message'))
    print('===============================TRAINED===================================')

async def demoTest(data):
    print('===============================PREPROCESSING...===================================')
    if 'preprocess_output.pkl' not in os.listdir():
        preprocess_output = train_preprocess(
            data_path=data['data_path'], model_type=data['model_type'], test_size=data['test_size'], number_records=data['number_records'])
        joblib.dump(preprocess_output, filename='preprocess_output.pkl')
    else:
        preprocess_output = joblib.load('preprocess_output.pkl')
        print('Load from previous preprocessed_output.pkl')
        print(preprocess_output.get('message'))


    print('===============================PREPROCESSED===================================')

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