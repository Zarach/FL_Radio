import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D

SERVER_ADDRESS = "0.0.0.0:8080"

def create_model():
    base_model = tf.keras.applications.resnet.ResNet101(classes=2, include_top=False, pooling='avg')
    head_model = Flatten()(base_model.output)
    head_model = Dense(1, activation='sigmoid')(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)
    return model

# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     print(metrics)
#     accuracies = [num_examples * m["metrics/mAP50(B)"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#
#     # Aggregate and return custom metric (weighted average)
#     return {"metrics/mAP50(B)": sum(accuracies) / sum(examples)}

net = create_model()

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1#,
    #initial_parameters=fl.common.ndarrays_to_parameters(params)
    # evaluate_metrics_aggregation_fn=weighted_average,
    # fit_metrics_aggregation_fn=weighted_average
)

# Start Flower server
def start_server():
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

start_server()