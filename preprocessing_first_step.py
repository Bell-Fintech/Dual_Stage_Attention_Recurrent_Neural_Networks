import pandas as pd
import numpy as np
import os, csv, time
import multiprocessing as mp

#checks if u can actually multiprocess, turns out some routine call do not allow multiprocessing
os.system('taskset -p %s' %os.getpid()) #If the CPU affinity is returned as f, of ff, you can access multiple cores. In my case it would start like that,
                                        # but upon importing numpy or scipy.any_module, it would switch to 1

def calculate_max_timestep(sublist_of_csv_files):

    max = 0

    for csv_file in sublist_of_csv_files:
        print("Currently working on calculating max end time step of table: ",csv_file)
        table = pd.read_csv("../clusterdata-2011-1/task_usage/" + csv_file, header=None, index_col=None)

        maximum_of_all_columns = table.max()
        max_timestep = maximum_of_all_columns[1]

        if max_timestep > max:
            print("Maximum timestep changed from ", max, " to ", max_timestep)
            max = max_timestep

    return max



def generate_timestep_csv(sublist_of_csv_files):

    print("sublist of csv files is: ",str(sublist_of_csv_files))
    print(sublist_of_csv_files[0])
    #time.sleep(1111)
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
    list_of_csv_files = os.listdir("data/csv_files/")

    list_of_csv_files.sort()
    # print(list_of_csv_files)

    print("The last csv file is: ",list_of_csv_files[-1])

    #for my laptop
    #list_of_csv_files = [list_of_csv_files[i:i+ 3] for i in range(0, len(list_of_csv_files), 3)]#generating a list with 12 elements for 12 CPUs
    #for star server
    list_of_sublist_csv_files = [list_of_csv_files[i:i+ 16] for i in range(0, len(list_of_csv_files), 16)]#generating a list with 12 elements for 12 CPUs


    pool_for_max_timestep_calculation = mp.Pool(pool_size)
    max_result = pool_for_max_timestep_calculation.map(calculate_max_timestep, list_of_sublist_csv_files)

    print("The maximum end timestep should be: ",max(max_result))

    print("The no of processes created for multi-processing is: ",len(list_of_sublist_csv_files))#--> 32


    result = pool.map(generate_timestep_csv, list_of_sublist_csv_files)

    print("Total number of elements in result: ",len(result))#------------should be equal to the number of processors used for multiprocessing

    print("Completed calculating mean cpu usage for differnet CPUs. Now writing to csv.")

    for process in range(len(result)):

        with open("mean_cpu_usage_process_"+str(process)+".csv", "w") as f:

            writer = csv.writer(f, delimiter=",")
            writer.writerow(result[process])

    #check the dimensionality of this csv after you are done.



    print("Completed! Total time taken: ",time.time()-start_time)


