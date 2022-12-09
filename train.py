"""# START REGION: MODEL SELECTION & TRAINING"""

from time import time
import tensorflow as tf
from keras import optimizers
from keras import losses, metrics
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
from datetime import datetime
import os
import asyncio
# INPUT: X y train, model type, batch size, number of epoch
# OUTPUT: output folder path, message


def train(X_train, y_train, epoch_num, batch_size, model_type, labId, model_config=None):
    if model_type == 'sequential':
        Y_train = pd.get_dummies(y_train)

        model = Sequential()
        model.add(Dense(1024, activation='relu',
                  input_shape=(X_train.shape[1],)))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='sigmoid'))

        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                      loss=losses.binary_crossentropy,
                      metrics=[metrics.binary_accuracy])
        output_folder = f'./modelDir/{labId}/log_train'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filepath = output_folder+"/model-"+model_type+"-epoch-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(
            filepath, verbose=0,
            save_best_only=False, save_weights_only=False,
            save_freq="epoch")
        try:
            for epoch in range(epoch_num):
                history = model.fit(X_train, Y_train, epochs=epoch+1, initial_epoch=epoch,
                                    batch_size=batch_size, verbose=1, callbacks=[checkpoint_callback])
                temp = history.history

                yield {
                    "binary_accuracy": temp['binary_accuracy'],
                    "loss": temp['loss']
                }
        except:
            pass

        # return {
        #     'output_folder': output_folder,
        # }

    # import matplotlib.pyplot as plt

    # acc = history.history['binary_accuracy']
    # loss = history.history['loss']
    # epochs = range(len(acc))
    # plt.plot(epochs, acc, 'b-', label='Testing accuracy')
    # plt.plot(epochs, loss, 'g-', label='Training Loss')
    # plt.title('Training accuracy')
    # plt.xlabel('Epoch',fontsize=15)
    # plt.ylabel('Percentage',fontsize=15)
    # plt.legend()
    # plt.savefig(output_folder+'/train_process.png')

    # return {
    #         'output_folder':output_folder,
    #         'message':'Checkpoints and image are saved to ' + output_folder
    #         }
"""# END REGION: MODEL SELECTION & TRAINING"""
