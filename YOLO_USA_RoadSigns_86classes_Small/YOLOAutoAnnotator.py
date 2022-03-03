#
# YOLOAutoAnnotator.py
# BoundingBoxDetector
# 2022/02/27 toshiyuki.arai

import os
import sys
import glob
import cv2
import numpy as np
import traceback

class YOLOAutoAnnotator:

  def __init__(self, classes_file):
    self.classes = []
    with open(classes_file, "r") as file:
      for i in file.read().splitlines():
        print(i)
        self.classes.append(i)  
    print(self.classes)
    
    
  def get_class_index(self, label):
    index = -1
    if label == None or len(label) == 0:
      return index
      
    for i, class_name in enumerate(self.classes):
      if class_name == label:
        index = i
        break
      
    return index
       

  def run(self, images_dir, output_dir, debug=True):
    pattern = images_dir + "/*.jpg"
    files = glob.glob(pattern)

    for file in files:
      try:    
        # Load image, grayscale, Otsu's threshold 
        image = cv2.imread(file)
        fname = os.path.basename(file)
        name = fname
        i = fname.find(".jpg")
        if i>-1:
          name = fname[0:i]

        label = None
        if name.find("_0_") >0:
          label = name.split("_0_")[0]

        if name.find("_1_") >0:
          label = name.split("_1_")[0]
        if name.find("_2_") >0:
          label = name.split("_2_")[0]

        if name.find("_3_") >0:
          label = name.split("_3_")[0]
        if name.find("_4_") >0:
          label = name.split("_4_")[0]

        if name.find("_5_") >0:
          label = name.split("_5_")[0]
        if name.find("_6_") >0:
          label = name.split("_6_")[0]

        index = self.get_class_index(label)
        if index == -1:
          print("---------- Not found class {}".format(fname))
          os.remove(file)
          continue    

        print("=== file {} class {}".format(file, label))

        (height, width, _) = image.shape
        #https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3,3))
        med_val = np.median(gray)
        sigma = 0.33  # 0.33
        min_val = int(max(0, (1.0 - sigma) * med_val))
        max_val = int(max(255, (1.0 + sigma) * med_val))
        
        canny_output = cv2.Canny(gray, threshold1 = min_val, threshold2 = max_val)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(contours)

        cx = 0.5
        cy = 0.5
            
        cw = 0.8
        ch = 0.8
        minx = width
        miny = height
        minw = 0
        minh = 0

        for i, c in enumerate(contours):
          contours_poly[i] = cv2.approxPolyDP(c, 3, True)
          x, y, w, h = cv2.boundingRect(contours_poly[i])

          if x <minx:
            minx = x
          if y <miny:
            miny = y
          if w >minw:
            minw = w
          if h >minh:
            minh = h
        cv2.rectangle(image, (minx, miny), (minx + minw, miny + minh), (36,255,12), 2)
        cx = (float(minx) + float(minw)/2.0)/float(width)
        cy = (float(miny) + float(minh)/2.0)/float(height)
        cw = float(minw)/float(width)
        ch = float(minh)/float(height)

        cx = round(cx, 4)
        cy = round(cy, 4)
            
        cw = round(cw, 4)
        ch = round(ch, 4)
            
        print("Best YOLO BoundingBox------- cx {} cy {} cw {} ch {}".format( cx, cy, cw, ch))

        outputfile = os.path.join(output_dir, name + ".txt")

        SP = " "
        NL = "\n"

        with open(outputfile, "w") as txt:
          line = str(index) + SP + str(cx) + SP + str(cy) + SP + str(cw) + SP + str(ch)
          print(line) 
          txt.write(line+NL)
        print("----------- Saved {}".format(outputfile))
        if debug:
          cv2.imshow('image', image)
          cv2.waitKey()

      except:    
        traceback.print_exc()
        break
        

# python YOLOAutoAnnotator.py train
# python YOLOAutoAnnotator.py valid
if __name__ == "__main__":
  images_dir  = "./train"
  output_dir  = "./train"
  classes_file = "./classes.txt"
  
  try:
    if len(sys.argv) ==2:
        images_dir = sys.argv[1]
        output_dir = sys.argv[1]
    else:
      raise Exception("Usage: python {} dataset ".format(sys.argv[0]))

    if os.path.exists(images_dir) == False:
      raise Exception("Not found "+ images_dir)
    bb = YOLOAutoAnnotator(classes_file)
    bb.run(images_dir, output_dir, debug=False)
    
  except:
    traceback.print_exc()

