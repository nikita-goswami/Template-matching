"""
Character Detection

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

"""


import argparse
import json
import os

import utils
from task1 import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # Thresholding the image to make it cleaner 
    def thresholding(img,c):
      img_threshold = copy.deepcopy(img)
      for i in range(len(img_threshold)):
        for j in range(len(img_threshold[0])):
          if(img[i][j]>c):
            img_threshold[i][j] = 255
          else:
            img_threshold[i][j] = 0
      return img_threshold


    # resize template
    def resize(img, dim):
      img = np.array(img).astype('float32')
      resized = cv2.resize(img, dim)
      return resized


    # Normalized cross correlation
    def correlation_coef(a, b):
        prod = np.mean((a - a.mean()) * (b - b.mean()))
        stds = a.std() * b.std()
        if stds == 0:
            return 0
        else:
            return prod/stds
            
                    
            
    def norm_cross_cor(image,template):
      image=np.array(image)
      l1,b1=np.array(template).shape
      l2,b2=np.array(image).shape
      ncc=np.zeros((l2-l1+1,b2-b1+1)).tolist()
      for i in range(l2-l1+1):
        for j in range(b2-b1+1):
          img= image[i:i+l1,j:j+b1]
          ncc[i][j]=correlation_coef(img,np.array(template))
      return ncc

    # Drop points for the same character coming more than once
    def drop_same_points(m):
      store=[]
      for idx in m:
        p,q=idx[0],idx[1]
        if (p-1,q) in store or (p+1,q) in store or (p,q-1) in store or (p,q+1) in store or (p+1,q+1) in store or (p-1,q-1) in store or (p+1,q-1) in store or (p-1,q+1) in store:
          pass 
        else: 
          store+=([idx])
      return store


    # Threshold for selection 
    def find_max(arr,c):
      m=[]
      (l,b)=arr.shape
      for i in range(l):
        for j in range(b):
          if arr[i][j]>=c:
            #m+=[(i,j)]
            m+=[(i,j)]
      return(m)

    def template_match(img,template):
      dim=(14, 14)
      resized_template = resize(template,dim)
      threshold_img = thresholding(resized_template,150)
      matches=[]
      img_ncc=norm_cross_cor(img,threshold_img)
      img_ncc_np=np.array(img_ncc)
      matches += find_max(img_ncc_np,0.85)

      dim=(11, 11)
      resized_template = resize(template,dim) 
      threshold_img = thresholding(resized_template,170)
      img_ncc=norm_cross_cor(img,threshold_img)
      img_ncc_np=np.array(img_ncc)
      matches += find_max(img_ncc_np,0.67)

      if(len(matches)==1):
        matches += find_max(img_ncc_np,0.54)

      return drop_same_points(matches)

    # smoothen the image
    gaussian=np.matmul(np.array([[1],[2],[1]]),np.array([[1,2,1]]))
    gauss_image=convolve2d(img,gaussian)
    coordinates = template_match(gauss_image,template)
    print(coordinates)

    #raise NotImplementedError
    return coordinates

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
