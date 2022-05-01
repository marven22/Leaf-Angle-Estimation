import cv2
import os
from os import listdir
import numpy as np
import random as rng
import math
from itertools import combinations
from collections import OrderedDict, Counter
from operator import itemgetter
import glob
import csv
import pandas as pd
import sys

CURRENT_DIRECTORY = os.getcwd()

IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "ULAInputImages", "")

OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY, "ear_angle_masks_2", "Outlier_OriginalLineImages")

RESULT_PATH = os.path.join(CURRENT_DIRECTORY, "")

MERGED_HOUGH_LINE_IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "ear_angle_masks_2", "MergedHoughLines")

#RECT_PATH = "C:\\Users\\marven\\Documents\\Fall-2021\\LeafAngle\\ear_angle_masks_2\\rectangles\\"

kernel = np.ones((3, 3), np.uint8)

# Finds the slope of two lines using the two-point method
def findSlope(a, b):
        return ((b[1] - a[1])/(b[0] - a[0]))

# Returns the first line from a list of lines
def get_lines(lines_in):
    return [l[0] for l in lines_in]

# https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
# Merges multiple small line segments into a single line segment
def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 100
    min_angle_to_merge = 5

    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        #lines[idx] = False
            # append new group
            super_lines.append(new_group)


    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final

def merge_lines_segments1(lines, use_log=False):
    if(len(lines) == 1):
        return lines[0]

    line_i = lines[0]

    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):

        #sort by y
        points = sorted(points, key=lambda point: point[1])

        if use_log:
            print("use y")
    else:

        #sort by x
        points = sorted(points, key=lambda point: point[0])

        if use_log:
            print("use x")

    return [points[0], points[len(points)-1]]

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw
def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

    if (min(dist1,dist2,dist3,dist4) < 100):
        return True
    else:
        return False

def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

def SelectLinesAroundProbHoughMedian(median, lines, threshold):
    validLines = []
    lowAngle = median - threshold
    highAngle = median + threshold
    for i in range(len(lines)):
        line = lines[i]
        x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
        lineAngle = np.round(math.degrees(abs(math.atan2((y2 - y1), (x2 - x1)))))
        if lowAngle <= lineAngle <= highAngle:
            validLines.append(lines[i])
    return validLines

#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
# http://paulbourke.net/geometry/pointlineplane/
def DistancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine


def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])


    return min(dist1,dist2,dist3,dist4)


def FindLargestContour(contours):
    maxArea = 0
    maxContour = -1
    for i in range(len(contours)):
        if(cv2.contourArea(contours[i]) > maxArea):
            maxArea = cv2.contourArea(contours[i])
            maxContour = i

    return contours[maxContour]
    
def IsImageFlipped(contours, imgShape):
    moments = cv2.moments(contours)    
    if moments["m00"] == 0: 
        moments["m00", "m01"] = 1
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"]) 
    if(x < imgShape[1]/2):
        return True
    else:
        return False
        
def CropImageContour(contour, imgShape, imgName):
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    flipped = IsImageFlipped(contour, imgShape)
    
    img = np.zeros((imgShape[0], imgShape[1]), np.uint8)
    

    if flipped:        
        cv2.rectangle(img, (int(x + w/2), y), (x + w, y + h), (255, 255, 255), -1)        
        #print(f.split("\\")[-1])
        #print(str(x))
        #print(str(y))
        #print(str(int((x + w)/2)))
        #print(str(int(y +h)))
    else:
        cv2.rectangle(img, (x, y), (int(x + w/2), y + h), (255, 255, 255), -1)

    return img

def WriteImage(img, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    cv2.imwrite(os.path.join(CURRENT_DIRECTORY, dir, filename.split("\\")[-1]), img)

def SelectLinesAroundProbHoughMedian(median, lines, threshold):
    validLines = []
    lowAngle = median - threshold
    highAngle = median + threshold
    for i in range(len(lines)):
        line = lines[i]
        x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
        lineAngle = np.round(math.degrees(abs(math.atan2((y2 - y1), (x2 - x1)))))
        if lowAngle <= lineAngle <= highAngle:
            validLines.append(lines[i])
    return validLines

def CountValidLines(lines):
    count = 0
    validLines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #print(abs(math.atan2((y2 - y1), (x2 - x1))))
        if x1 <= 100 or x2 <= 100 or x1 >= houghLines.shape[1] - 100 or x2 >= houghLines.shape[1] - 100 or y1 >= houghLines.shape[0] - 50 or y2 >= houghLines.shape[0] - 50 or y1 == y2 or x1 == x2 or (abs(math.atan2((y2 - y1), (x2 - x1))) > 1.39 and abs(math.atan2((y2 - y1), (x2 - x1))) < 1.60) or abs(math.atan2((y2 - y1), (x2 - x1))) < 0.3:                   
            continue    
        
        count = count + 1
        validLines.append(line)
    #if count > 6:
        #return validLines
    return validLines

def findDictMode(input_dict):
    track={}

    for key,value in input_dict.items():
        if value not in track:
            track[value]=0
        else:
            track[value]+=1        

    return max(track,key=track.get)

def RetrieveParallelLines(lines):
    lineDict = {}
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        slope = round(abs(findSlope((x1, y1), (x2, y2))), 2)
        lineDict[i] = slope
    mode = findDictMode(lineDict)
    
    parallelLines = []
    for k, v in lineDict.items():
        if v == mode:
            parallelLines.append(lines[k])
           
    return parallelLines
        
if os.path.isfile(RESULT_PATH + "Summer_2015-Ames_ULA_bitwise_outliers.csv"):
        os.remove(RESULT_PATH + "Summer_2015-Ames_ULA_bitwise_outliers.csv")
csvFile = open(RESULT_PATH + "Summer_2015-Ames_ULA_bitwise_outliers.csv", "w")

resultsFile = csv.writer(csvFile)
header = ("Filename", "Median Angle", "Mean Angle", "Merged Angle", "mla_1", "mla_2")
#resultsFile.write("Filename            " + " " + "Prob Hough Median" + "     " + "Prob Hough Mean" + "     " + "Merged Angle\n")
resultsFile.writerow(header)        
   
for f in glob.glob(IMAGE_PATH + "*.jpg"):   
    try:
        img2 = cv2.imread(IMAGE_PATH + f.split("\\")[-1])

        img2Copy = img2.copy()

        # Convert the mask to gray scale
        img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Binary threshold the image and plot the pixels of the stem in white and background in black
        ret, thresh = cv2.threshold(img2Gray, 1, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contours) > 1):
            contours = FindLargestContour(contours)

        #cropped = CropImageContour(contours, (img2Gray.shape[0], img2Gray.shape[1]), f.split("\\")[-1])

        #cropped_bitwise = cv2.bitwise_and(img2Copy, img2Copy, mask = cropped)

        #cv2.imwrite(RECT_PATH + f.split("\\")[-1], cropped_bitwise)

        thresh = np.zeros((img2Gray.shape[0], img2Gray.shape[1]), np.uint8)

        thresh = cv2.drawContours(thresh, [contours], -1, 255, -1) 

        flipped = IsImageFlipped(contours, (img2Copy.shape[0], img2Copy.shape[1]))

        bitwise = cv2.bitwise_and(img2Copy, img2Copy, mask = thresh)     

        img2Gray = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img2Gray, 1, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        thresh = cv2.drawContours(thresh, contours, -1, 255, -1) 

        cv2.imwrite("thresh_img2.jpg", thresh)   

        #cv2.imwrite("thresh_img2.jpg", thresh)

        # Dilate the image to join any discontinuities
        edges = cv2.dilate(kernel, thresh, iterations = 2)

        # Perform canny edge detection on the image
        edges = cv2.Canny(image=img2Gray, threshold1=100, threshold2=200) # Canny Edge Detection

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imwrite("edge.jpg", edges)

        contImg = np.zeros((edges.shape[0], edges.shape[1], 3), np.uint8)

        houghLines = np.zeros((edges.shape[0], edges.shape[1], 3), np.uint8)

        mergedLines = np.zeros((edges.shape[0], edges.shape[1], 3), np.uint8)

        # For the majority of it, parallel lines is turned on with minLineLength set to 300
        # MinLineLength is decreased to account for images where the leaf and stem are only a tiny portion.
        # MinLineLengths of 50 and lower are used.
        found = False
        votes = 180 #used 120 for 78 outliers
        angles = []
        validLines = []
        while(not found):
            # Compute Probabilistic Hough Lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, votes, lines = 1, minLineLength = 10, maxLineGap = 500)
            print(votes)
            if(lines is None):
                votes = votes - 10
                continue

            if(votes == 0):
                break   

    #         for line in lines:
    #             x1, y1, x2, y2 = line[0]

    #             if x1 <= 100 or x2 <= 100 or x1 >= houghLines.shape[1] - 100 or x2 >= houghLines.shape[1] - 100 or y1 >= houghLines.shape[0] - 50 or y2 >= houghLines.shape[0] - 50 or y1 == y2 or x1 == x2 or (abs(math.atan2((y2 - y1), (x2 - x1))) > 1.39 
    #                         and abs(math.atan2((y2 - y1), (x2 - x1))) < 1.60) or abs(math.atan2((y2 - y1), (x2 - x1))) < 0.3:        

    #                 continue
            lines = CountValidLines(lines)   

            if(len(lines) == 0):
                votes = votes - 10
                continue

            lines = RetrieveParallelLines(lines)  

            #print(len(lines))

            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = math.atan2((y2 - y1), (x2 - x1)) 
                if(flipped):
                    if(slope > 0):                
                        continue
                else:
                    if(slope < 0):
                        continue

                found = True
                print(flipped)
                angles.append(abs(math.degrees(slope)))
                validLines.append(line)
                cv2.line(bitwise, (x1, y1), (x2, y2), (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), 2)
            votes = votes - 10
        median = np.median(angles)
        mean = np.mean(angles)

        if(median < 9):
            median = 90 - median
            mean = 90 - mean

    #     validLines = SelectLinesAroundProbHoughMedian(median, validLines, 5)

    #     # prepare
    #     _lines = []
    #     for _line in get_lines(validLines):
    #         _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
    #     # sort
    #     _lines_x = []
    #     _lines_y = []
    #     for line_i in _lines:
    #         orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    #         if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
    #             _lines_y.append(line_i)
    #         else:
    #             _lines_x.append(line_i)

    #     _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    #     _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    #     merged_lines_x = merge_lines_pipeline_2(_lines_x)
    #     merged_lines_y = merge_lines_pipeline_2(_lines_y)

    #     merged_lines_all = []
    #     merged_lines_all.extend(merged_lines_x)
    #     merged_lines_all.extend(merged_lines_y)

    #     for line in merged_lines_all:
    #         cv2.line(mergedLines, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 1)

    #     cv2.imwrite(MERGED_HOUGH_LINE_IMAGE_PATH + f.split("\\")[-1], mergedLines)

    #     mergedLineImg = mergedLines#cv2.imread(, 0)

    #     mergedLineImg = cv2.cvtColor(mergedLineImg, cv2.COLOR_BGR2GRAY)

    #     # Perform canny edge detection 
    #     edges = cv2.Canny(image=mergedLineImg, threshold1=100, threshold2=200) # Canny Edge Detection

    #     # Compute the hough lines using houghLinesP method
    #     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength = 10, maxLineGap = 500)

    #     if lines is None:
    #         print(f)
    #         continue    

    #     # Create a new image to plot the hough lines
    #     houghLines = np.zeros((img2.shape[0], img2.shape[1], 3), np.uint8) #cv2.imread(PATH + "skeleton-resized.jpg")

    #     for line in lines:
    #         x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2],line[0][3]

    #     # If x1 == x2, then the line is vertical whose slope is undefined. So, we pass
    #         if x1 <= 100 or x2 <= 100 or x1 >= mergedLineImg.shape[1] - 100 or x2 >= mergedLineImg.shape[1] - 100 or y1 >= mergedLineImg.shape[0] - 50 or y2 >= mergedLineImg.shape[0] - 50 or y1 == y2 or x1 == x2 or (abs(math.atan2((y2 - y1), (x2 - x1))) > 1.44 and abs(math.atan2((y2 - y1), (x2 - x1))) < 1.60) or abs(math.atan2((y2 - y1), (x2 - x1))) < 0.3:
    #             continue
    #         # Compute the slope of the line.
    #         slope = abs(findSlope((x2, y2), (x1, y1)))
    #         # Emperically determine the valid range of slopes that are to be considered.
    #         #if slope is not None:
    #         if slope > 0.1:
    #             flipped = False
    #             angle = np.round(math.degrees(abs(math.atan2((y2 - y1), (x2 - x1))))) 
    #             if angle > 90:
    #                 flipped = True
    #                 angle = 180 - angle


    #             theta = math.degrees(math.tan(math.radians(angle)))
    #             slope = findSlope((x2, y2), (x1, y1))
    #             r = math.sqrt(1 + (slope * slope))
    #             if(flipped):
    #                 x2 = int(x1 - 1000/r)
    #                 y2 = int(y1 - (1000 * slope/r))
    #             else:
    #                 x2 = int(x1 + 1000/r)
    #                 y2 = int(y1 + (1000 * slope/r))

    #             cv2.line(houghLines, (x1, y1), (x2, y2), (255, 255, 255), 1)
    #             cv2.line(img2Copy, (x1, y1), (x2, y2), (255, 255, 255), 1)
               # count = count + 1

        data = [f.split("\\")[-1], str(median), str(mean), str(0), str(0), str(0)]
        resultsFile.writerow(data)

        #cv2.imwrite(OUTPUT_PATH + f.split("\\")[-1], bitwise)
        WriteImage(bitwise, "originalImageLine", f)
    except Exception as e:
        print(e)
        pass

csvFile.close()