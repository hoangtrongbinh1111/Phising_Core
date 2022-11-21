import pandas as pd
from preprocess import train_preprocess
from train import train
from test import test
from inference import inference
import os
import joblib
DEFAUTL_SAMPLE_NUM=100
EPOCH_NUM=10
BATCH_SIZE=32
BEST_EPOCH_NUM=10
DATA_PATH='./data/data.csv'
async def demo(): 
    # print('===============================PREPROCESSING...===================================')
    # preprocess_output=train_preprocess(data_path=DATA_PATH,model_type="sequential",test_size=0.3)
    # print(preprocess_output.get('message'))
    # print('===============================PREPROCESSED===================================')
    print('===============================PREPROCESSING...===================================')
    if 'preprocess_output.pkl' not in os.listdir():
        preprocess_output=train_preprocess(data_path=DATA_PATH,model_type="sequential",test_size=0.3, number_records=DEFAUTL_SAMPLE_NUM)
        joblib.dump(preprocess_output, filename='preprocess_output.pkl')
    else:
        preprocess_output=joblib.load('preprocess_output.pkl')
        print('Load from previous preprocessed_output.pkl')
    print(preprocess_output.get('message'))
    print('===============================PREPROCESSED===================================')
    X_train=preprocess_output.get('X_train')
    y_train=preprocess_output.get('y_train')
    X_test=preprocess_output.get('X_test')
    y_test=preprocess_output.get('y_test')
    model_type=preprocess_output.get('model_type')

    print('===============================TRAINING...===================================')
    train_output = train(X_train,y_train,epoch_num=EPOCH_NUM,batch_size=BATCH_SIZE,model_type=model_type)
    try:
        for res_per_epoch in train_output:
            if res_per_epoch:
                yield res_per_epoch
    except:
        print("error")
    
    # print(train_output.get('message'))
    print('===============================TRAINED===================================')

    # print('===============================TESTING...===================================')    
    # test_output=test(output_folder=train_output.get('output_folder'),X_test=X_test,y_test=y_test,epoch_num=BEST_EPOCH_NUM)
    # print(test_output.get('message'))
    # print('===============================TESTED===================================')   

    # print('===============================INFERENCING...===================================')    
    # inference_output=inference(output_folder=train_output.get('output_folder'), data_path='./data/data.csv', epoch_num=BEST_EPOCH_NUM)
    # print(inference_output.get('message'))
    # print(inference_output.get('inference_result'))
    # print('===============================INFERENCED===================================')  