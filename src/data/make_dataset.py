# -*- coding: utf-8 -*-
import os
import pandas as pd
import shutil


def move_images(dataframe, selected_classes, dest_base):
    # Moving images into train  val test folders
    failed_list = []
    for i, row in dataframe.iterrows():
        image_class = row['class']

        if image_class in selected_classes:
            src = row.orig_img_path
            dest_folder = os.path.join(dest_base, row.target, row['class'])
            img_name = os.path.basename(src)
            dest = os.path.join(dest_folder, img_name)
            target = row.target

            if os.path.exists(dest_folder) is False:
                print("Creating Destination Folder", dest_folder)
                os.makedirs(dest_folder)

            try:
                shutil.copyfile(src, dest)

            # If file missing at src
            except FileNotFoundError:
                print('FileNotFound')
                failed_list.append(["FileMissing",
                                    src,
                                    dest,
                                    target,
                                    image_class])
                continue

            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
                failed_list.append(["SameFile",
                                    src,
                                    dest,
                                    target,
                                    image_class])

            # If destination is a directory.
            except IsADirectoryError:
                print("Destination is a directory.")
                failed_list.append(["Directory",
                                    src,
                                    dest,
                                    target,
                                    image_class])

            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")
                failed_list.append(["Denied",
                                    src,
                                    dest,
                                    target,
                                    image_class])

    print(f'Total files failed to move: {len(failed_list)}')


dataframe = pd.read_csv('data\\interim\\speedlimit_signs.csv')
selected_classes = ['Speed Limit 100',
                    'Speed Limit 120',
                    'Speed Limit 20',
                    'Speed Limit 30',
                    'Speed Limit 40',
                    'Speed Limit 50',
                    'Speed Limit 60',
                    'Speed Limit 70',
                    'Speed Limit 80',
                    'Speed Limit 90']
dest_base = 'data\\processed'

move_images(dataframe=dataframe,
            selected_classes=selected_classes,
            dest_base=dest_base)

# Walk through dataset directory and list number of filed
for dirpath, dirnames, filenames in os.walk(dest_base):
    print(f'There are  {len(dirnames)} directories and {len(filenames)}\
           files in {dirpath}')
