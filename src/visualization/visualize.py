import os
import random
import yaml
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import logging


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and
    validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.figure()
    plt.plot(epochs, loss, label='training loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()  # Everytime you want a new plot
    plt.plot(epochs, accuracy, label='accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.legend()


def plot_random_images(c, images_path, labels_path, class_names):
    """
    Plot a random set of images from dataset
    """
    num_rows = 3
    num_cols = 3

    # Get a list of image files
    image_files = os.listdir(images_path)
#     label_files = os.listdir(labels_path)

    # Randomly select num_rows * num_cols images
    selected_images = random.sample(image_files, num_rows * num_cols)

    # Plot images in a grid
    fig, axes = plt.subplots(num_rows, num_cols)

    for i, image_file in enumerate(selected_images):
        # Load and plot the image
        image_path = os.path.join(images_path, image_file)
        img = mpimg.imread(image_path)
        axes[i // num_cols, i % num_cols].imshow(img)
        axes[i // num_cols, i % num_cols].axis('off')

        # Load and print the corresponding label
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_path, label_file)
        with open(label_path, 'r') as label_file:
            label_info = label_file.read().split()[0]  # Extract a substring

        axes[i // num_cols, i % num_cols].\
            set_title(class_names[int(label_info)])

    plt.savefig(f"reports\\figures\\RandomImages_{c}")
    # plt.show()


def main(image_dir, label_dir, yaml_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting Random Images')

    with open(yaml_filepath, "r") as stream:
        try:
            data_loaded = yaml.safe_load(stream)
    #         print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    keys = data_loaded['names']
    class_names = {i: keys[i] for i in range(len(keys))}

    for i in range(10):
        plot_random_images(i, image_dir, label_dir, class_names)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    mpl.style.use("seaborn-v0_8-deep")
    mpl.rcParams["figure.figsize"] = (20, 5)
    mpl.rcParams["figure.dpi"] = 100
    image_dir = "data\\raw\\train\\images"
    label_dir = "data\\raw\\train\\labels"
    yaml_filepath = "data\\raw\\data.yaml"

    main(image_dir, label_dir, yaml_filepath)
