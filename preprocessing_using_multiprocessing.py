import pandas as pd
import os, csv, time
import multiprocessing as mp




def generate_timestep_csv(sublist_of_csv_files):


    # create a list of floats to put the mean CPU usage values for each timestep
    mean_cpu_usage_list = [0] * 200000  # max end time for last file is 2.506200e+12 convert to seconds from microseconds first

    # read them one by one
    for csv_file in sublist_of_csv_files:

        # see which csv file we are working currently
        print("CSV Table is: ", str(csv_file), "\n")

        with open("/media/being-aerys/SP PHD U3/clusterdata-2011-1/task_usage/" + str(csv_file), "r") as table:

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
    df = pd.DataFrame(mean_cpu_usage_list)

    df.to_csv('mean_cpu_usage_list'+str(sublist_of_csv_files[0])+'.csv', index=False)


if __name__=="__main__":

    pool = mp.Pool(mp.cpu_count())
    print("Total CPUs available for multiprocessing: ",mp.cpu_count())

    #sort the list of csv files for sequential processing
    list_of_csv_files = os.listdir("/media/being-aerys/SP PHD U3/clusterdata-2011-1/task_usage/")

    list_of_csv_files.sort()

    list_of_csv_files = [list_of_csv_files[i:i+ 42] for i in range(0, len(list_of_csv_files), 42)]#generating a list with 12 elements for 12 CPUs

    print(len(list_of_csv_files))#--> 12


    pool.map(generate_timestep_csv, list_of_csv_files)

    print("Completed!")









