import os
import re
import cv2
import numpy as np

data = {}

char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

folder_name = "CCPD_SubDataset"
subfolder_name = "train"
saving_folder_name = "transformed_images"

coodinates_pattern = r'\d+-\d+_\d+-\d+&\d+_\d+&\d+-(?P<x1>\d+)&(?P<y1>\d+)_(?P<x2>\d+)&(?P<y2>\d+)_(?P<x3>\d+)&(?P<y3>\d+)_(?P<x4>\d+)&(?P<y4>\d+)-.*'
coodinates_regex = re.compile(coodinates_pattern)

last_character_pattern = r'\d+-\d+_\d+-\d+&\d+_\d+&\d+-\d+&\d+_\d+&\d+_\d+&\d+_\d+&\d+-\d+_\d+_(?P<d1>\d+)_(?P<d2>\d+)_(?P<d3>\d+)_(?P<d4>\d+)_(?P<d5>\d+)-'
last_character_regex = re.compile(last_character_pattern)

current_directory = os.getcwd()
folder_path = os.path.join(current_directory, folder_name)
subfolder_path = os.path.join(folder_path, subfolder_name)
saving_path = os.path.join(current_directory, saving_folder_name)

files_names = os.listdir(subfolder_path)

for file  in files_names:
  data[file] = {}
  match = coodinates_regex.search(file)
  if match:
    for coordinate in match.groupdict():
      data[file][coordinate] = match.group(coordinate)

    last_char = last_character_regex.search(file)
    if last_char:
      data[file]["last_char"] = last_char.groupdict()  
      
      src_points = np.array([[data[file]["x1"], data[file]["y1"]], [data[file]["x2"], data[file]["y2"]], [data[file]["x3"], data[file]["y3"]], [data[file]["x4"], data[file]["y4"]]], dtype=np.float32)
      dst_points = np.array([[110, 35], [0, 35], [0, 0], [110, 0]], dtype=np.float32)
      
      data[file]["src_points"] = src_points
      data[file]["dst_points"] = dst_points
      
      homography_matrix, status = cv2.findHomography(src_points, dst_points)
      data[file]["homography_matrix"] = homography_matrix
      
      image = cv2.imread(os.path.join(subfolder_path, file))
      result = cv2.warpPerspective(image, homography_matrix, (111, 36))
      
      cropped_image = result[6:30, 35:105]
      blurred_image = cv2.GaussianBlur(cropped_image,(1,1),cv2.BORDER_DEFAULT)
      gray_scaled_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
      binary_image = cv2.threshold(gray_scaled_image, 140, 255, cv2.THRESH_BINARY)[1]
      morphological_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, np.ones((1, 1), np.uint8), iterations=2)
      morphological_image = cv2.morphologyEx(morphological_image, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8),iterations=4)     

      cv2.imwrite(os.path.join(saving_path, file), morphological_image)
      # data[file]["image"] = morphological_image
    
            
      
  else:
    data[file] = None


cv2.imshow("Image Processing Steps", morphological_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(data[next(iter(data))])
# cv2.imshow('Result', data[next(iter(data))]["image"])
# cv2.waitKey(0)

    