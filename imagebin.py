# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2
import os


def read_image(image_file, gray_scale=False):
  image_src = cv2.imread(image_file)
  if gray_scale:
      image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
  else:
      image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
  return image_src

def write_image(file_path, img):
  cv2.imwrite(file_path, img)
  return  

# image binarization using Niblack method
def image_binariation(img):
  img_shape = img.shape
  print("image shape", img_shape)
  bin_img = np.zeros(img_shape, dtype=np.uint8)

  window_size = 25
  rows = img_shape[0]
  cols = img_shape[1]

  for i in range(rows):
    for j in range(cols):
      # get window size of neigboring pixels
      row_top = i - 2
      row_bottom = i + 2
      col_left = j - 2
      col_right = j + 2

      if row_top < 0:
        row_top = 0
        row_bottom = 5
      
      if row_bottom > rows - 1:
        row_bottom = rows - 1
        row_top = rows - 6

      if col_left < 0:
        col_left = 0
        col_right = 5
      
      if col_right > cols - 1:
        col_right = cols - 1
        col_left = cols - 6
      
      nbr_pixels = bin_img[row_top:row_bottom, col_left:col_right]
      std_dev = np.std(nbr_pixels)
      mean_pix = np.mean(nbr_pixels)

      threshold = mean_pix * (1 + (0.5 * (std_dev/128 - 1)))
      # print("threshold ", threshold)
      # print("img[i][j] ", img[i][j])
      if img[i][j] > threshold:
        bin_img[i][j] = 255


  return bin_img

cwd = os.getcwd()

if os.path.exists(cwd + "/binscale.png"):
  os.remove(cwd + "/binscale.png")

img = read_image(cwd + "/grayscale_girl.png", True)

print("original image matrix ", img)

bin_img = image_binariation(img)
print("converted binary image matrix", bin_img)

write_image(cwd + "/binscale_girl.png", bin_img)