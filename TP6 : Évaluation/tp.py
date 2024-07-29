import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
import math
from collections import Counter


def get_normalised_histogram(img):

  #compute histogram for red , green and blue channel
  hist_red = cv2.calcHist([img], [0], None, [256], [0, 255])
  hist_green = cv2.calcHist([img], [1], None, [256], [0, 255])
  hist_blue = cv2.calcHist([img], [2], None, [256], [0, 255])
  #normalise the histogram
  hist_red = hist_red / hist_red.sum()
  hist_green = hist_green / hist_green.sum()
  hist_blue = hist_blue / hist_blue.sum()
  #concatenate the histogram
  hist = (hist_red, hist_green, hist_blue)
  return hist
  
def get_histograms(path):
  histograms = []
  files = os.listdir(path)
  for file in files:
    img = cv2.imread(os.path.join(path, file))
    histograms.append(get_normalised_histogram(img))
  return histograms
  
  
def hough_detect_circles(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1,
                               param1=100, param2=30,
                               minRadius=30, maxRadius=35)
  
  if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    circle = circles[0]
    (x, y, r) = circle
    # cv2.circle(img, (x, y), r, (0, 255, 0), 4)
    new_img = extract_panneaux(img, (x, y), r)
    # cv2.imshow("img", new_img)
    # cv2.waitKey(0)
    return (True, new_img)
  else:
    return (False, None)

    
    
def extract_panneaux(img, centre, rayon):
  new_image = np.zeros(img.shape)
  x, y = centre
  new_image = img[y-rayon:y+rayon, x-rayon:x+rayon]
  # for i in range(new_image.shape[0]):
  #   for j in range(new_image.shape[1]):
  #     print(((y-i)**2 +(x-j)**2))
  #     print( rayon**2)
  #     if ((i-y)**2 + (j-x)**2) < rayon**2:
  #       new_image[i][j] = 0
  # cv2.imshow("img", new_image)
  # cv2.waitKey()
  return new_image
  
  
def compute_Bhatt_distance(image, histogram):
  # cv2.imshow('img', image)
  # cv2.waitKey(0)
  img_hist = get_normalised_histogram(image)
  distance = 0
  summ = 0
  for i in range(3):
    hx_ = np.average(img_hist[i])
    hx_prim = np.average(histogram[i])
    
    for j in range(256):
      summ += math.sqrt(img_hist[i][j][0] * histogram[i][j][0])
  
    temp = math.sqrt(hx_*hx_prim*(256**2))*summ
    bhatt = math.sqrt(1 + (1/(temp)))
    distance += bhatt
  return distance


def compute_probability(img, histograms):
  distances = []
  panneaux_50, panneaux_120, depassement, giratoire, sens = histograms
  for histogram in panneaux_50:
      distance = compute_Bhatt_distance(img, histogram)
      temp = ("50", distance)
      distances.append(temp)
  for histogram in panneaux_120:
    distance = compute_Bhatt_distance(img, histogram)
    temp = ("120", distance)
    distances.append(temp)
  for histogram in depassement:
    distance = compute_Bhatt_distance(img, histogram)
    temp = ("depassement", distance)
    distances.append(temp)
  for histogram in giratoire:
      distance = compute_Bhatt_distance(img, histogram)
      temp = ("giratoire", distance)
      distances.append(temp)
  for histogram in sens:
      distance = compute_Bhatt_distance(img, histogram)
      temp = ("sens", distance)
      distances.append(temp)
  
  distances.sort(key=lambda a: a[1])
  distances = distances[:5]
  
  most_frequent = get_most_frequent(distances)
  return most_frequent[0]

def get_most_frequent(distances):
  res = max(Counter(distances).items(), key=lambda ele: ele[0])
  return res[0]

def main():
  
  current_folder = os.path.join(os.getcwd(), "ressources")
  test_folder = os.path.join(current_folder, "test")
  app_folder = os.path.join(current_folder, "app")
  panneaux_folder = os.listdir(app_folder)

  panneaux_50 = []
  panneaux_120 = []
  depassement = []
  giratoire =  []
  sens = []
  
  for folder in panneaux_folder:
    total_path = os.path.join(app_folder, folder)
    if "50" in folder:
      panneaux_50=get_histograms(total_path)
    elif "120" in folder:
      panneaux_120 = get_histograms(total_path)
    elif "depassement" in folder:
      depassement=get_histograms(total_path)
    elif "giratoire" in folder:
      giratoire = get_histograms(total_path)
    elif "sens" in folder:
      sens = get_histograms(total_path)
      
    all_histograms = (panneaux_50, panneaux_120, depassement, giratoire, sens)
      
      
  # img = cv2.imread(os.path.join(test_folder, "50/002.png"))
  # compute_probability(img, all_histograms)
  
  results = []
  count = 0
  total = 0
  folders_test_list = os.listdir(test_folder)
  for folder in folders_test_list:
    total_path = os.path.join(test_folder, folder)
    files = os.listdir(total_path)
    for file in files:
      final_path = os.path.join(total_path, file)
      img = cv2.imread(final_path)
      # cv2.imshow("img", img)
      # cv2.waitKey(0)
      panneaux_detected, panneaux = hough_detect_circles(img)
      if panneaux_detected:
        highest_k = compute_probability(panneaux, all_histograms)
        results.append((file, folder, highest_k))
        if str(highest_k) == folder:
          count += 1
      total += 1
      

  print(results)
  print("Le taux de réussite final est de :", count*100/total,"%")
  
  
            
if __name__ == "__main__":
  main()