from clearml import StorageManager, Dataset

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="FL_Radio", dataset_name="Radio_Raw_Data"
)

# add the example csv
dataset.add_files(path='raw_data/')

# Upload dataset to ClearML server (customizable)
dataset.upload()

# commit dataset changes
dataset.finalize()