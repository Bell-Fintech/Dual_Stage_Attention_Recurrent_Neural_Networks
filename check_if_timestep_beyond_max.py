import pandas as pd
import os, csv, time

if __name__=="__main__":

    #sort the list of csv files for sequential processing
    list_of_csv_files = os.listdir("/media/being-aerys/SP PHD U3/clusterdata-2011-1/task_usage")
    list_of_csv_files.sort()
    #list_of_csv_files.reverse() #check if timestep beyond the value given


    list_to_maintain_total_rows_in_csvs = [0] * 500
    csv_counter = 0


    #read then one by one
    for csv_file in list_of_csv_files:


        print("CSV to check: ", str(csv_file), "\n")
        # print(pd.read_csv("/media/being-aerys/SP PHD U3/clusterdata-2011-1/task_usage/" + str(csv_file)))
        # time.sleep(333)
        with open("/media/being-aerys/SP PHD U3/clusterdata-2011-1/task_usage/" + str(csv_file), "r") as table:

            row_counter_for_csv = 0
            #read using datareader
            datareader = csv.reader(table)


            #manipulate each row of data from the csv file
            for row in datareader:

                print(row)
                start_timestep = int(row[0])
                end_timestep = int(row[1])

                if end_timestep > 2.506200e+12: #this value is the  end time step of the last task of the last csv file
                    raise ValueError("End timestep is beyond 2.506200e+12 for this row: ",row_counter_for_csv)
                row_counter_for_csv += 1

            print("Total rows: ",row_counter_for_csv)
            list_to_maintain_total_rows_in_csvs[csv_counter] = row_counter_for_csv #maintain the no of rows on a list

            csv_counter += 1


    df = pd.DataFrame(list_to_maintain_total_rows_in_csvs)

    df.to_csv('list_to_maintain_total_rows_in_csvs.csv', index=False) #store the list of rows into a csv
















