from clearml import Task, Dataset

client_number = 0

task = Task.init(project_name='FL_MRT', task_name=f'Client {client_number}')
task.execute_remotely(queue_name='default', clone=False, exit_process=True)


import client
from ultralytics import YOLO
from cv2.version import headless

# get local copy of DataBases
dataset_databases = Dataset.get(dataset_project='FL_MRT', dataset_name='MRT_Raw_Data')
dataset_path_databases = dataset_databases.get_mutable_local_copy("raw_data/", True)

client.start_client("127.0.0.1")