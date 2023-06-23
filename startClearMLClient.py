from clearml import Task, Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server-address", default="radio-server.testing:8080")
parser.add_argument("--client-id", default=1)

args = parser.parse_args()

task = Task.init(project_name='Radio_MRT', task_name=f'Client {args.client_id}')
task.execute_remotely(queue_name='default', clone=False, exit_process=True)

import client

# get local copy of DataBases
dataset_databases = Dataset.get(dataset_project='FL_Radio', dataset_name='Radio_Raw_Data')
dataset_path_databases = dataset_databases.get_mutable_local_copy("raw_data/", True)

client.start_client(args.client_id, args.server_address)