import pandas as pd
import os, csv, time

if __name__=="__main__":

    #sort the list of csv files for sequential processing
    list_of_csv_files = os.listdir("data/csv_files/")
    list_of_csv_files.sort()

    #create a list of floats to put the mean CPU usage values for each timestep
    mean_cpu_usage_list = [0] * 100000

    #read then one by one
    for csv_file in list_of_csv_files:

        #see which csv file we are working currently
        print("CSV Table is: ", str(csv_file), "\n")

        with open("data/csv_files/" + str(csv_file), "r") as table:

            #read using datareader
            datareader = csv.reader(table)
            #manipulate each row of data from the csv file
            for row in datareader:

                for timestep in range(int((int(row[1]) - int(row[0])) / 1000000)):

                    # print("----------------------------------")
                    # print(row[0])
                    # print(timestep)
                    # print(int(row[0])+timestep)
                    # print((int(row[0])+timestep)//1000000)#floor division
                    #
                    # print("------------------------------------")

                    actual_timestep_to_use = ((int(row[0])+timestep)//1000000) + timestep

                    #print("Timestep is ", actual_timestep_to_use) # row[0] + timestep gives the starting time step for this particular task

                    #print("Initial CPU usage was: ", mean_cpu_usage_list[actual_timestep_to_use])

                    if actual_timestep_to_use == 601:
                        print("Timestep is ",
                              actual_timestep_to_use)  # row[0] + timestep gives the starting time step for this particular task

                        print("Initial CPU usage was: ", mean_cpu_usage_list[actual_timestep_to_use])

                        # mean_cpu_usage_list[actual_timestep_to_use] += float(row[5])

                        #print("New cpu usage is ", mean_cpu_usage_list[actual_timestep_to_use])

                    mean_cpu_usage_list[actual_timestep_to_use] += float(row[5])

                    #print("New cpu usage is ", mean_cpu_usage_list[actual_timestep_to_use])

                    if actual_timestep_to_use == 601:
                        # print("Timestep is ",
                        #       actual_timestep_to_use)  # row[0] + timestep gives the starting time step for this particular task
                        #
                        # print("Initial CPU usage was: ", mean_cpu_usage_list[actual_timestep_to_use])

                        #mean_cpu_usage_list[actual_timestep_to_use] += float(row[5])

                        print("New cpu usage is ", mean_cpu_usage_list[actual_timestep_to_use])






                #     print(timestep)
                #
                # time.sleep(333)
                # #calculate for which time steps we need to add the cpu usage from this particlar record
                # for timestep in range(int((int(row[1]) - int(row[0])) / 1000000)):  # division
                #
                #     #add the cpu usage value to that time step
                #     print("Timestep is ",timestep)
                #     print("Initial CPU usage was: ",row[5])
                #     mean_cpu_usage_list[timestep] += float(row[5])
                #     print("New cpu usage is ", mean_cpu_usage_list[timestep])
                #     time.sleep(5)








