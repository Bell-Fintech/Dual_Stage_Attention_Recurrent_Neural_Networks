from torch.utils.data import Dataset

import numpy as np, pandas as pd

import time
#path_to_training_dataset = "data/timeseries_training_testing_final.csv"
#path_to_testing_dataset = "../Datasets/timeseries_testing_final.csv"


class Testing_Dataset_Maker(Dataset):

    counter = 0


    def __init__(self, validation_data):


        #self.custom_validation_dataset_from_csv = np.array(pd.read_csv(path_to_training_dataset, header=None))
        self.custom_validation_dataset = validation_data
        print("Shape of the validation data set sent to the data loader is: ",self.custom_validation_dataset.shape)
        self.time_sequence_length = 100


        print("Total samples in the validation data set is: ", len(self.custom_validation_dataset)) #1068288


    def __len__(self):


        return len(self.custom_validation_dataset)



    def __getitem__(self, idx):

        #print(self.custom_training_dataset_from_csv[idx].shape) 101

        features = self.custom_validation_dataset[idx][:self.time_sequence_length]


        label = self.custom_validation_dataset[idx][self.time_sequence_length]
        #print(features.shape)
        #print(label.shape)

        return(features, label) #return a tuple




'''

The DataLoader takes a Dataset object (and, therefore, any subclass extending it) and several other optional parameters (listed on the PyTorch DataLoader docs). 
Among the parameters, we have the option of shuffling the data, determining the batch size and the number of workers to load data in parallel. 

'''

#create training and testing data sets
# training_set = mean_cpu_usage[:(9 * len(mean_cpu_usage)//10)]#use 90% data as training data
# testing_set = mean_cpu_usage[(9 * len(mean_cpu_usage)//10):]
