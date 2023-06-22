import datetime
import os

from collections import OrderedDict

import flwr as fl

import numpy as np
from flask import Flask, request
import tensorflow as tf
from keras.layers import Flatten, Dense
from tensorflow.keras import Model

app = Flask(__name__)
SERVER_ADDRESS = "127.0.0.1:8080"
CLIENT_ID = 0
data_received = False

image_size = [224, 224]
batch_size = 28



# def eval_net(opt):
#     save_dir = increment_path(Path("runs/eval/") / opt.name)
#     os.mkdir(save_dir)
#
#     results, _, _ = val.run(
#         data=opt.data,
#         model=creation_of_the_model(opt),
#         device=opt.device,
#         save_dir=save_dir,
#         save_model=True,
#     )
#
#     return results

def load_data():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "raw_data",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, val_ds



# Use one of the following URLs per Client
# "FL_Data/hh_01"
# "FL_Data/hh_07"
# "FL_Data/hh_14"
@app.route('/startClient', methods=['GET'])
def start_client():
    # Make TensorFlow log less verbose
    # metrics.append(tf.keras.metrics.Precision())
    # metrics.append(tf.keras.metrics.Recall())

    base_model = tf.keras.applications.resnet.ResNet101(classes=2, include_top=False, pooling='avg')
    head_model = Flatten()(base_model.output)
    head_model = Dense(1, activation='sigmoid')(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    train_ds, val_ds = load_data()


    # Define Flower client
    class Client(fl.client.NumPyClient):

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(train_ds, epochs=1, batch_size=batch_size, steps_per_epoch=3)
            len = train_ds.cardinality().numpy()
            weights = model.get_weights()
            return weights, int(len), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(val_ds)
            return loss, int(val_ds.cardinality().numpy()), {"accuracy": float(accuracy)}


    # Start Flower client
    client = Client()
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)

    return {
        'statusCode': 200,
        'body': 'Client finished'
    }

if __name__ == '__main__':
   # app.run()
   start_client()



