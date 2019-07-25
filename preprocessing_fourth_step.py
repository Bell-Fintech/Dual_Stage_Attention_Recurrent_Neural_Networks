


import numpy as np, pandas as pd, time
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)

def build_csv(time_sequence_length):

    #path_to_original_dataset = "../Datasets/part-00499-of-00500.csv"
    path_to_combined_dataset_with_no_head_or_tail = "data/final_cpu_usage.csv"

    mean_cpu_usage = pd.read_csv(path_to_combined_dataset_with_no_head_or_tail,header=None, index_col= None)  # load the data set -make sure u read with header = None....IMP here
    print("Final csv samples count: ",mean_cpu_usage.shape)

    #mean_cpu_usage = mean_cpu_usage[:10000]#take only 10k for now

    mean_cpu_usage = np.array(mean_cpu_usage)


    # mean_cpu_usage_limited_training = mean_cpu_usage[:9000]
    # mean_cpu_usage_limited_testing = mean_cpu_usage[9000:10000:1]

    #In pandas, we must indices are supposed to be immutable

    timeseries_dataset_sequence= []
    timeseries_dataset_labels= []


#-----------------------------------------------------------training dataset
    for i in range(len(mean_cpu_usage)):

        timeseries_dataset_temp = np.zeros((time_sequence_length))

        if i < len(mean_cpu_usage) - time_sequence_length:
            for j in range(time_sequence_length):
                timeseries_dataset_temp[j] = (mean_cpu_usage[i + j])

            timeseries_dataset_sequence.append(timeseries_dataset_temp)

            timeseries_dataset_labels.append(mean_cpu_usage[i + time_sequence_length])



    timeseries_dataset_sequence = np.array(timeseries_dataset_sequence)



    timeseries_dataset_labels = np.array(timeseries_dataset_labels)



    final_training_and_testing_dataset = np.column_stack((timeseries_dataset_sequence, timeseries_dataset_labels))
    #854353 (got 61 columns instead of 101)
    #print(final_training_and_testing_dataset.shape) (2499299, 101)


    np.random.shuffle(final_training_and_testing_dataset)

    #print(len(final_dataset)) 2499299 --> 100 less than the no of time steps in the final_cpu_usage i.e., 2499399--> correct

    # final_training_dataset = final_dataset[: int ( 0.9 * len (final_dataset) )]
    #
    # final_testing_dataset = final_dataset[int ( 0.9 * len (final_dataset) ):]

    # print("Total number of samples in the training dataset: ",len (final_training_dataset))
    # print("Total number of samples in the testing dataset: ",len(final_testing_dataset))

    print("final_data_after_preprocessing", final_training_and_testing_dataset.shape) #(2499299, 101), 100 less than 2499399
    np.savetxt("data/final_data_after_preprocessing.csv", final_training_and_testing_dataset, delimiter=",")
    #np.savetxt("data/timeseries_testing_final.csv", final_testing_dataset, delimiter=",")




    print("Dataset creation completed successfully.\n")

#----------------------------VVI the op is shhuffled so the order is not preserved but everything is still correct--------------

if __name__ == "__main__":
    build_csv(100)
