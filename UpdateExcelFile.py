# importing pandas
import pandas as pd
import glob
import shutil
import numpy as np
import os 

CURRENT_DIRECTORY = os.getcwd()

ALG_RES_PATH = os.path.join(CURRENT_DIRECTORY, "Summer_2015-Ames_ULA_bitwise.csv")

ALG_OUTLIER_PATH = os.path.join(CURRENT_DIRECTORY, "Summer_2015-Ames_ULA_bitwise_outliers.csv") 

ALG_RES_OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY, "")
  
# read text file into pandas DataFrame
df = pd.read_csv(ALG_RES_PATH, sep=",")

df2 = pd.read_csv(ALG_OUTLIER_PATH, sep=",")

# print(list(df2.columns))

# outliers = df2['OUTLIER FILES'].dropna().unique().tolist()

# for f in outliers:
#     shutil.copy2(SRC_PATH + f, DEST_PATH)


#print(df2['Outlier'])
filenames = df2.loc[df2['Outlier'] == False, 'Filename'].values.tolist()
medianangles = df2.loc[df2['Outlier'] == False, 'Median Angle'].values.tolist()
meanangles = df2.loc[df2['Outlier'] == False, 'Mean Angle'].values.tolist()

for i in range(len(filenames)):
    listIndex = np.where(df['Filename'] == filenames[i])[0][0]
    df.at[listIndex, 'Median Angle'] = medianangles[i]
    df.at[listIndex, 'Mean Angle'] = meanangles[i]           
    
df.to_csv(ALG_RES_OUTPUT_PATH + ALG_RES_PATH.split("\\")[-1])