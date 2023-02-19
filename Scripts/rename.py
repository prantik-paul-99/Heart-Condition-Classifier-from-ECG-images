# reanme all images of this folder

import os
import sys

def rename():
    src_folders = [
        'ECGImagesofMyocardialInfarctionPatients',
        'ECGImagesofPatientthathaveabnormalheartbeat',
        'ECGImagesofPatientthathaveHistoryofMI',
        'NormalPersonECGImages'
    ]
    dest_folders = [
        'new/MI',
        'new/HB',
        'new/PMI',
        'new/Normal'
    ]
    new_nums = [
        77,
        548,
        203,
        859
    ]


    for folder in src_folders:
        # read all files in this folder and sort them
        files = os.listdir(folder)

        for file in files:
            # get the file name
            filename = os.path.splitext(file)[0]

            # get the number of this file
            num = int(filename.split('_')[1])
            # add the corresponding value from new_num
            new_num = num + new_nums[src_folders.index(folder)]

            # rename the file
            new_name = filename.split('_')[0] + '_' + str(new_num) + '.jpg'
            # print(filename)
            # print(new_name)
            
            # save the file with new name to the corresponding folder
            os.rename(folder + '/' + file, dest_folders[src_folders.index(folder)] + '/' + new_name)
            # break

        


if __name__ == '__main__':
    rename()