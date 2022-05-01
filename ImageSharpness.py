from PIL import Image
import numpy as np
import os

CURRENT_DIRECTORY = os.getcwd()

# Note: Add filename in the double quotes
filename = os.path.join(CURRENT_DIRECTORY, "") 
im = Image.open(filename).convert('L') # to grayscale
array = np.asarray(im, dtype=np.int32)

gy, gx = np.gradient(array)
gnorm = np.sqrt(gx**2 + gy**2)
sharpness = np.average(gnorm)

print(sharpness)