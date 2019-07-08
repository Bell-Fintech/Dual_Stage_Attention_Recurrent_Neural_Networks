import pandas as pd
import numpy as np
import os, csv, time
import multiprocessing as mp

#checks if u can actually multiprocess, turns out some routine call do not allow multiprocessing
os.system('taskset -p %s' %os.getpid()) #If the CPU affinity is returned as f, of ff, you can access multiple cores. In my case it would start like that,
                                        # but upon importing numpy or scipy.any_module, it would switch to 1



def generate_timestep_csv(sublist_of_csv_files):


    # create a list of floats to put the mean CPU usage values for each timestep
    mean_cpu_usage_list = [0] * 3000000  # max end time for last file is 2.506200e+12 convert to seconds from microseconds first

    # read them one by one
    for csv_file in sublist_of_csv_files:

        # see which csv file we are working currently
        print("CSV Table is: ", str(csv_file), "\n")

        with open("../clusterdata-2011-1/task_usage/" + str(csv_file), "r") as table:

            # read using datareader
            datareader = csv.reader(table)

            # manipulate each row of data from the csv file
            for row in datareader:

                start_timestep = int(row[0])
                end_timestep = int(row[1])

                timesteps_range =  end_timestep - start_timestep # lets ignore the last time step since the document says only 300s for the measurement period

                for timestep in range(int(timesteps_range / 1000000.0) ): # instead of + 1):

                    actual_timestep_to_use = (int(row[0]) / 1000000) + timestep

                    #print("Timestep is ",actual_timestep_to_use)  # row[0] + timestep gives the starting time step for this particular task
                    #
               

     # print("Initial CPU usage was: ", mean_cpu_usage_list[actual_timestep_to_use])

                    mean_cpu_usage_list[int(actual_timestep_to_use)] += float(row[5])

                    # print("New cpu usage is ", mean_cpu_usage_list[actual_timestep_to_use])

            # print("Mean CPU Usage List: ", mean_cpu_usage_list)
    return mean_cpu_usage_list
    #df = pd.DataFrame(mean_cpu_usage_list)

    #df.to_csv('mean_cpu_usage_list'+str(sublist_of_csv_files[0])+'.csv', index=False)
    #return df

if __name__=="__main__":

        
    start_time = time.time()

    pool_size = mp.cpu_count()

    pool = mp.Pool(pool_size)

    os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))

    print("Total CPUs available for multiprocessing: ",mp.cpu_count())

    #sort the list of csv files for sequential processing
    list_of_csv_files = os.listdir("../clusterdata-2011-1/task_usage/")

    list_of_csv_files.sort()

    print("The last csv file in this batch is: ",list_of_csv_files[-1])
    #for my laptop
    #list_of_csv_files = [list_of_csv_files[i:i+ 3] for i in range(0, len(list_of_csv_files), 3)]#generating a list with 12 elements for 12 CPUs
    #for star server
    list_of_csv_files = [list_of_csv_files[i:i+ 16] for i in range(0, len(list_of_csv_files), 16)]#generating a list with 12 elements for 12 CPUs

    print("The no of processes created for multi-processing is: ",len(list_of_csv_files))#--> 12


    result = pool.map(generate_timestep_csv, list_of_csv_files)

    print("Total number of elements in result: ",len(result))

    print("Completed calculating mean cpu usage. Now writing to csv.")

    with open("mean_cpu_usage.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        numpy_array_total = np.array(result)
        writer.writerows(numpy_array_total)

    #check the dimensionality of this csv after you are done.



    print("Completed! Total time taken: ",time.time()-start_time)


