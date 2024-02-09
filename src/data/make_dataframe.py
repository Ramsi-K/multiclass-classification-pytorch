import os
import pandas as pd
import yaml


def get_class_names(data_loaded):
    keys = data_loaded['names']
    class_names = {i: keys[i] for i in range(len(keys))}
    return class_names


def build_df(raw_datadir, target, class_names):

    labels_dir = os.path.join(raw_datadir, target, 'labels')
    image_dir = os.path.join(raw_datadir, target, 'images')
    label_files = os.listdir(labels_dir)

    df = []
    failed = []

    for file_name in label_files:
        img_name = file_name.replace(".txt", ".jpg")

        label_file_path = os.path.join(labels_dir, file_name)
        orig_img_path = os.path.join(image_dir, img_name)

        with open(label_file_path, 'r') as label_file:
            lines = label_file.readlines()
        if len(lines) > 0:  # removing bad data
            img_class = int(lines[0].split()[0])
        else:
            failed.append([target, orig_img_path, label_file_path])
            continue

        dest_folder = class_names[img_class]

        df.append([target, img_class, dest_folder, orig_img_path])

    dataframe = pd.DataFrame(df,
                             columns=['target',
                                      'class_num',
                                      'class',
                                      'orig_img_path'])
    failed_df = pd.DataFrame(failed,
                             columns=['target',
                                      'orig_img_path',
                                      'label_file_path'])

    return dataframe, failed_df


dataset_dir = 'data'
raw_datadir = os.path.join(dataset_dir, 'raw')
interim_datadir = os.path.join(dataset_dir, 'interim')

with open(f'{raw_datadir}\\data.yaml', "r") as stream:
    try:
        data_loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class_names = get_class_names(data_loaded)
class_names

train_df, train_failed = build_df(raw_datadir=raw_datadir,
                                  target='train',
                                  class_names=class_names)
test_df, test_failed = build_df(raw_datadir=raw_datadir,
                                target='test',
                                class_names=class_names)
val_df, val_failed = build_df(raw_datadir=raw_datadir,
                              target='valid',
                              class_names=class_names)

print('Total entries in train: ', len(train_df), '\n',
      'Failed entries in train: ', len(train_failed), '\n',
      'Total entries in test: ', len(test_df), '\n',
      'Failed entries in test: ', len(test_failed), '\n',
      'Total entries in val: ', len(val_df), '\n',
      'Failed entries in val: ', len(val_failed)
      )

combined_df = pd.concat(
    [train_df,
     test_df,
     val_df]).reset_index().drop('index',
                                 axis=1)
combined_df.groupby(['target', 'class'])['class'].count()
combined_df.to_csv(f'{interim_datadir}\\speedlimit_signs.csv', index=False)
