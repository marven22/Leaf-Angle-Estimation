import pandas as pd
from scipy import spatial
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import os

CURRENT_DIRECTORY = os.getcwd()

CSV_FILE_PATH = os.path.join(CURRENT_DIRECTORY, "")

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product/ (norm_a * norm_b)

df = pd.read_csv(
CSV_FILE_PATH + "Summer_2015-Ames_ULA_bitwise_cropped_outliers-new.csv"
) 

algorithmResults = df['Mean Angle'].tolist()

student1 = df['ula_1'].tolist()

student2 = df['ula_2'].tolist()

x = []

for i in range(len(algorithmResults)):
    x.append(i)
        
print(np.isnan(algorithmResults).any())
    
val_out = 1 - cdist([algorithmResults], [student1], 'cosine')

alg_SP_result = 1 - spatial.distance.cosine(algorithmResults, student1)

alg_Can_result = 1 - spatial.distance.cosine(algorithmResults, student2)

SP_Can_result = 1 - spatial.distance.cosine(student1, student2)

print(alg_SP_result)

print(SP_Can_result)

print(alg_Can_result)

plt.xlabel("Image Number")
plt.ylabel("Leaf Angle")
plt.title("Results from Proposed Algorithm and Student 1 for Summer_2015-Ames_ULA")

plt.plot(x, algorithmResults, marker = 'o', color = 'crimson', label = "Proposed Algorithm")

plt.plot(x, student1, marker = 's', color = 'black', label = 'student 1')

plt.xticks(x[::50])

#plt.margins(0.2)

#for i in range(len(algorithmResults)):
#    plt.plot(x[i], algorithmResults[i], marker = 'o', color = 'crimson', label = "Proposed Algorithm" if i == 0 else "")
#    plt.plot(x[i], SamplePointResults[i], marker = 's', color = 'black', label = "mla_1" if i == 0 else "")    
    #plt.plot(x[i], CanopeoResults[i], marker = 'D', color = 'blue', label = "mla_2" if i == 0 else "")        

plt.legend()
plt.show()
