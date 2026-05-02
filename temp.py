import pandas as pd






rows_list = []
for i in range(10):
    dict1 = {}
    # get input row in dictionary format
    # key = col_name
    dict1["value"] = i
    if i%2 == 0: dict1["even"] = True

    rows_list.append(dict1)

df = pd.DataFrame(rows_list)
print(df[df["even"]==True])