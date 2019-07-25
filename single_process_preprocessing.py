import pandas as pd
import os, csv, time

if __name__=="__main__":

    #sort the list of csv files for sequential processing
    list_of_csv_files = os.listdir("/media/being-aerys/SP PHD U3/clusterdata-2011-1/task_usage/")
    list_of_csv_files.sort()

    #create a list of floats to put the mean CPU usage values for each timestep
    mean_cpu_usage_list = [0] * 3000000 #max end time for last file is 2.506200e+12 convert to seconds from microseconds first

    #read then one by one
    for csv_file in list_of_csv_files:

        #see which csv file we are working currently
        print("CSV Table is: ", str(csv_file), "\n")

        with open("/media/being-aerys/SP PHD U3/clusterdata-2011-1/task_usage/" + str(csv_file), "r") as table:


            #read using datareader
            datareader = csv.reader(table)

            #manipulate each row of data from the csv file
            for row in datareader:

                start_timestep = int(row[0])
                end_timestep = int(row[1])


                for timestep in range(int((end_timestep - start_timestep) / 1000000)):

                    actual_timestep_to_use = ((int(row[0])+timestep)//1000000) + timestep



                    # print("Timestep is ",actual_timestep_to_use)  # row[0] + timestep gives the starting time step for this particular task
                    #
                    # print("Initial CPU usage was: ", mean_cpu_usage_list[actual_timestep_to_use])

                    mean_cpu_usage_list[actual_timestep_to_use] += float(row[5])




                    # print("New cpu usage is ", mean_cpu_usage_list[actual_timestep_to_use])

            print("Mean CPU Usage List: ",mean_cpu_usage_list)


    df = pd.DataFrame(mean_cpu_usage_list)

    df.to_csv('mean_cpu_usage_list.csv', index=False)  # store the list of rows into a csv
    print("Completed!")









