import pandas as pd
f=pd.read_csv("test.csv")
keep_col = ['day','month','lat','long']
new_f = f[keep_col]
new_f.to_csv("newFile.csv", index=False)