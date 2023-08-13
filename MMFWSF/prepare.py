import os
import jsonlines
from speechbrain.dataio.dataio import read_audio, merge_csvs
from speechbrain.utils.data_utils import download_file
import shutil
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import random

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)


def prepare_Egolife(
    save_folder, dataset_csv, caption_type, skip_prep=False, seed=1234
):
    """
    This function prepares the EgoLife dataset.

    data_folder : path to EgoLife dataset.
    save_folder: path where to save the csv manifest files.
    caption_type : one of the following:

      "uniact":{input=audio, output=semantics}

    split the dataset to train (8), valid (1), test (1)
    """
    if skip_prep:
        return

    # May need to change
    with open(dataset_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = list(csv_reader)

    random.seed(seed)
    values = np.linspace(0, len(csv_reader) -1 , len(csv_reader), dtype=int)
    random.shuffle(values)
    training_dataset, testing_dataset = train_test_split(values, train_size= int(len(csv_reader) * 0.8), test_size=len(csv_reader) - int(len(csv_reader) * 0.8))
    validing_dataset = testing_dataset[0:int(len(testing_dataset)/2)]
    testing_dataset = testing_dataset[int(len(testing_dataset)/2):]
    
    splits = [
        "train",
        "test",
        "valid"
    ]

    for split in splits:
        new_filename = (
            os.path.join(save_folder, split) + "-type=%s.csv" % caption_type
        )
        if os.path.exists(new_filename):
            continue
        print("Preparing %s..." % new_filename)

        if (split == "train"):
            list_now = training_dataset
            print(len(list_now))
        elif (split == "test"):
            list_now = testing_dataset
        elif (split == "valid"):
            list_now = validing_dataset

        IDs = []
        duration = []
        wav = []
        semantics = []
        transcript = []

        for data_id in list_now:
            data_sample = csv_reader[data_id]
            IDs.append(data_sample[0])
            duration.append(data_sample[1])
            wav.append(data_sample[3])
            semantics.append(data_sample[4])
            transcript.append(data_sample[4])
        
        df = pd.DataFrame(
                {
                    "ID": IDs,
                    "duration": duration,
                    "wav": wav,
                    "semantics": semantics,
                    "transcript": transcript,
                }
            )
        df.to_csv(new_filename, index=False)

