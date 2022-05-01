# load the required packages
import cv2
import numpy as np
import glob

PATH = "C:\\Users\\marven\\Documents\\Fall-2021\\LeafAngle\\ear_angle_masks_2\\OutlierFiles\\"

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

for img in glob.glob(PATH + "*.jpg"):
    # load the image into system memory    
    image = cv2.imread(PATH + img.split("\\")[-1], flags=cv2.IMREAD_COLOR)



    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    image_sharp = cv2.filter2D(src=image_sharp, ddepth=-1, kernel=kernel)
    cv2.imwrite(PATH + img.split("\\")[-1], image_sharp)
