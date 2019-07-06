import time, os
import pandas as pd, numpy as np
import csv

# def read_data_from_a_table(table, final_table):
#
#     print("Input Table is: ",table)
#
#     with open("data/csv_files/" + table, "rb") as csvfile:
#         datareader = csv.reader(csvfile)
#         yield next(datareader)
#         count = 0
#         input_table_length = len(table)
#
#         print("hello table")
#




#
#
# def read_from_table(table,final_data_csv):
#
#     print("Called read_from_table")
#

    # with open("data/csv_files/" + table, "rb") as csvfile:
    #     datareader = csv.reader(csvfile)
    #     yield next(datareader)
    #     count = 0
    #     input_table_length = len(table)
    #
    #     with open(final_data_csv,"wb") as csvfile:
    #         filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #         filewriter.writerow(['Timestep', 'CPU_usage'])
    #
    #
    #     for row in datareader:
    #         print(row[1],row[0])
    #         time.sleep(111)
    #         for timestep_in_new_table in range(int(row[1] - row[0])):
    #             with open(final_data_csv,"wb") as csvfile:
    #
    #                 filewriter = csv.writer(csvfile, delimiter = ',')
    #                 for timestep_in_new_table in range(int(row[1] - row[0])):
    #                     filewriter.writerow([1,2,3])


def hello(arg):
    print(arg)


if __name__ == "__main__":

    lst = os.listdir("data/csv_files/")
    lst.sort()
    chunksize = 10000
    final_list_size = 250

    final_data_csv = "final_data_csv.csv"

    for table in lst:

        print("CSV Table is: ", str(table), "\n")

        for chunk in pd.read_csv(final_data_csv, chunksize=chunksize):#-----------------new file
            #process(chunk)

            with open("data/csv_files/" + table, "r") as csvfile:
                datareader = csv.reader(csvfile)

                #yield next(datareader)
                count = 0
                input_table_length = len(table)

                for row in datareader:



                    with open(final_data_csv, "wb") as csvfile:

                        filewriter = csv.writer(csvfile, delimiter=',')

                        for timestep_in_new_table in range(int((int(row[1]) - int(row[0]))/1000000)): #division

                            #new table ko yo timestep_in_new_table row ma rakhne


















   #
   #  input_raw_data = pd.read_csv("data/part-00499-of-00500.csv",header=None, index_col= None)
   #
   #  no_of_samples_in_file = len(input_raw_data)
   #
   #  # print(no_of_samples_in_file)
   #  # time.sleep(333)
   #
   #  maximum_of_all_columns = input_raw_data.max()
   #
   #  maximum_end_point_of_measurement = maximum_of_all_columns[1]
   #
   #  sample_part_of_file = input_raw_data.iloc[:10,:] #numpy does require this iloc
   #
   #  #cumulative_cpu_usage_at_each_timestep = np.zeros(int(maximum_end_point_of_measurement)) #-----------this gives memory error
   #  cumulative_cpu_usage_at_each_timestep = np.zeros(int(10)) #-----------this gives memory error
   #
   #
   #  for task in range(len(sample_part_of_file)):#----------------------------task is 0 to 9 #replace input_raw_data with sample_part_of_file for testing
   #
   #      for timestep in range(int(sample_part_of_file.iloc[task,:][1] - sample_part_of_file.iloc[task,:][0])):
   #
   #          cumulative_cpu_usage_at_each_timestep[ sample_part_of_file.iloc[task,:][0] + timestep] = cumulative_cpu_usage_at_each_timestep[int(sample_part_of_file.iloc[task,:][0]) + timestep] + sample_part_of_file.iloc[task,:][5]
   #
   #          print(cumulative_cpu_usage_at_each_timestep)
   # # print(cumulative_cpu_usage_at_each_timestep)
   #
   #


















# with open(filename) as f:
#     path_points = [tuple(line) for line in csv.reader(f)]
# path_points = [(float(point[0]), float(point[1]), float(point[2])) for point in path_points]

    # def writeToCSV(self, x, y):
    #     with open(self.file_path, mode='a') as waypoint_file:
    #         waypoint_writer = csv.writer(waypoint_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         waypoint_writer.writerow([x, y, 0])




