import numpy as np

if __name__ == "__main__":

    combined_cpu_usage = np.genfromtxt("data/combined_all_cpu_usage.csv", delimiter=",")

    combined_cpu_usage = combined_cpu_usage[601:2500000]  # combined_cpu_usage[-1] is not 0

    print("After removing the zero paddings, total length: ",combined_cpu_usage.shape) #2499399

    np.savetxt("data/final_cpu_usage.csv", combined_cpu_usage, delimiter=",")
