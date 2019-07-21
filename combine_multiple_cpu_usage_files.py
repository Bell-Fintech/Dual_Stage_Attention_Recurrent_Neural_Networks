'''This file combines 32 CPU usage files obtained after multiprocessing into one. '''




import time
import csv, os, numpy as np

if  __name__=="__main__":


    # a = np.array([1,2,3])
    # b = np.array([4,5,6])
    # print(a + b)

    preprocessed_files_path = "cpu_usage_files/"
    list_of_csv_files = os.listdir(preprocessed_files_path)
    #list_of_csv_files.sort() This puts 11 after 1 instead of 2

    #sort
    list_sorted_in_ascending_order = sorted(list_of_csv_files, key=lambda x: int(os.path.splitext(x)[0]))    #list_of_csv_files.sort()

    #Check that sorting is correct
    print("List of files to read from in ascending order: ",list_sorted_in_ascending_order)

    #calculate the length of one of the cpu_usage_files
    print("We will get the no of timesteps from ", list_sorted_in_ascending_order[0])

    input_file = np.genfromtxt("cpu_usage_files/"+ list_sorted_in_ascending_order[0], delimiter=',')
    combined_file_length = input_file.shape[0] #-------> 3000000


    #create a numpy array that will store the cumulative cpu usage
    combined_cpu_usage = np.zeros((1,combined_file_length))# cpu usage csv files are also this long

    for csv_file in list_sorted_in_ascending_order:

        # see which csv file we are working currently on
        print("CSV Table is: ", str(csv_file), "\n")

        input_file = np.genfromtxt("cpu_usage_files/"+ csv_file, delimiter=',')
        input_file = input_file.reshape((1,combined_cpu_usage.shape[1])) #both ccombined_file and input_file are (1, 3000000) now

        combined_cpu_usage = combined_cpu_usage + input_file
        #print("Combined_Cpu_usage_600 and 601 timestep: ", combined_cpu_usage[:, 100000], " ",combined_cpu_usage[:, 600000])

    print("All files merged into a single array. Now, saving into a csv file after reshaving from (1,3000000) to (3000000,1).")

    combined_cpu_usage = combined_cpu_usage.reshape((3000000,1))
    np.savetxt("combined_all_cpu_usage.csv", combined_cpu_usage, delimiter=",")

    '''I checked the dimension of the combined file. It conforms to 3000000.'''


