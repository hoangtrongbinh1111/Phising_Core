import pandas as pd
from preprocess import train_preprocess
from preprocess import inference_preprocess
from train import train
from test import test
from inference import inference
import os
import joblib
BEST_EPOCH_NUM = 10
OUTPUT_FOLDER = './output/28 Nov 2022 16h50'

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
    test_output=test(output_folder=OUTPUT_FOLDER,X_test=X_test,y_test=y_test,epoch_num=BEST_EPOCH_NUM)
    # print(test_output.get('message'))

    try:
        return test_output.get('message')
    except: 
        print("error")
    print('===============================TESTED===================================')


async def demoInfer(data):
    print('===============================PREPROCESSING...===================================')
    if 'preprocess_output.pkl' not in os.listdir():
        preprocess_output = inference_preprocess(
            data_path='./data/data.csv', feature_set = data['feature_set'])
        joblib.dump(preprocess_output, filename='preprocess_output.pkl')
    else:
        preprocess_output = joblib.load('preprocess_output.pkl')
        print('Load from previous preprocessed_output.pkl')
        print(preprocess_output.get('message'))


    print('===============================PREPROCESSED===================================')


    print('===============================INFERENCING...===================================')
    inference_output=inference(output_folder=OUTPUT_FOLDER, data_path='./data/data.csv', epoch_num=BEST_EPOCH_NUM)

    try:
        return {
           "message": inference_output.get('message'),
           "result" : inference_output.get('inference_result')
        }
    except:
        print('error')
    print('===============================INFERENCED===================================')