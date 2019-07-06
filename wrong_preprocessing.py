import pandas as pd
import numpy as np
import time








if __name__ == "__main__":

    path_to_original_dataset = "data/part-00499-of-00500.csv" #Note this version does not contain sampled CPU usage i.e., column no 19

    raw_csv = pd.read_csv(path_to_original_dataset,header=None, index_col= None)  # load the data set -make sure u read with header = None....IMP here
    print("original samples count ",raw_csv.shape)

    print(raw_csv.iloc[:,0:6])
    time.sleep(2222)

    mean_cpu_usage = raw_csv.iloc[:, 5]  # get the CPU usage column only for now
    mean_cpu_usage = mean_cpu_usage[:10000]#take only 10k for now
    mean_cpu_usage = np.array(mean_cpu_usage)


    mean_cpu_usage_limited_training = mean_cpu_usage[:9000]
    mean_cpu_usage_limited_testing = mean_cpu_usage[9000:10000:1]

    #In pandas, we must indices are supposed to be immutable
