import flwr as fl

import tensorflow as tf
from keras.layers import Flatten, Dense
from tensorflow.keras import Model


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

def load_data(client_id):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"raw_data/Client{client_id}",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"raw_data/Client{client_id}",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, val_ds



def start_client(client_id = 1, server_address = "radio-server.testing:8080"):
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

    train_ds, val_ds = load_data(client_id)


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
    fl.client.start_numpy_client(server_address=server_address, client=client)


