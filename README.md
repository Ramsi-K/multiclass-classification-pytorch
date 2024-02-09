# Road Sign Classifier

Multiclass classification of road signs using diverse methods, featuring from-scratch, transfer learning, and transformer architectures. Demonstrates model training and monitoring with DVC.

## Dataset Details

- **Dataset Source**: [Roboflow - Self-Driving Cars](https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou)
- **License**: CC BY 4.0
- **Annotation Format**: YOLOv8

This dataset, provided by a Roboflow user, contains road sign images for multiclass classification tasks in self-driving car applications. It was exported via Roboflow on October 20, 2023, at 9:18 AM GMT.

## Methodology

### Raw Data

- Image Count: 4969
- Image Dimensions: 416x416 (Resized)
- Image Augmentation: NoneS

Preliminary EDA revealed that the ideal categories for conducting classification would be the following:
```
[   'Speed Limit 100',
    'Speed Limit 120',
    'Speed Limit 20',
    'Speed Limit 30',
    'Speed Limit 40',
    'Speed Limit 50',
    'Speed Limit 60',
    'Speed Limit 70',
    'Speed Limit 80',
    'Speed Limit 90']
```
[def1]: reports\figures\RandomImages_0.png
[def2]: reports\figures\RandomImages_9.png

![eda1][def1]
![eda2][def2]

Accordingly, the data was processed into the standard image classification folder style with the above mentioned labels.

### Data Augmentations

For the experimental stage, only image resizing and normalizing transformations have been used. 
If these do not result in adequate prediction accuracies, further augmentations will be considered.

### Training

1. TinyVGG network - I am currently working on this. I have trained the TinyVGG model in Jupyter Notebook. I am currently working on moving the code to .py files and setting up experiment tracking.
2. TODO

### Experiment Tracking

For this project I am using DVC to track my data and projects. I am currently building reproducible training pipelines.

### Evaluation
TODO

### References
TODO

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


