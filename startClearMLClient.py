from clearml import Task, Dataset

client_id = 1

task = Task.init(project_name='Radio_MRT', task_name=f'Client {client_number}')
task.execute_remotely(queue_name='default', clone=False, exit_process=True)


import client

# get local copy of DataBases
dataset_databases = Dataset.get(dataset_project='FL_Radio', dataset_name='Radio_Raw_Data')
dataset_path_databases = dataset_databases.get_mutable_local_copy("raw_data/", True)

client.start_client(client_id)