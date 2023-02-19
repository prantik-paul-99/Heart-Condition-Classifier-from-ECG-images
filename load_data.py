import numpy as np
import pandas as pd
import os
import cv2 as cv
import PIL as pil

folders = [
          'dataset_cropped/Covid',
          'dataset_cropped/HB',
          'dataset_cropped/Normal',
          'dataset_cropped/MI',
          'dataset_cropped/PMI'
          ]

filenames = [
             'npy_files/covid.npy',
             'npy_files/hb.npy',
             'npy_files/normal.npy',
             'npy_files/mi.npy',
             'npy_files/pmi.npy'
             ]


def load_data():
    for folder_path in folders:
        images = []
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
        # print("len of image_paths: {}".format(len(image_paths)))
        for image_path in image_paths:
            img = cv.imread(image_path)
            if img is not None:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                invert = cv.bitwise_not(gray)
                dilate = cv.dilate(invert, None, iterations=2)
                # erode = cv.erode(dilate, None, iterations=2)

                img = cv.resize(dilate, (28, 28))
                img = np.array(img)
                
                images.append(img)
        
        images = np.array(images)
        print("done loading images from {}".format(folder_path))
        print("total images loaded: {}".format(len(images)))

        # # if the file does not exist, create it
        # if not os.path.exists(filenames[folders.index(folder_path)]):
        #     open(filenames[folders.index(folder_path)], 'w').close()

        np.save(filenames[folders.index(folder_path)], images)

if __name__ == '__main__':
    load_data()