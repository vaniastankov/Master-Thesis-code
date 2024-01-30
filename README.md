# Natural Audio Data Augmentation Techniques

**Type:** Master's Thesis

**Author:** Ivan Stankov

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Benjamin Fabian

![results](/Box%20Plot%20for%20EfficientNetV2B1%20Accuracies.png)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
  - [Dependencies](#Dependencies)
  - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
  - [Training code](#Training-code)
  - [Evaluation code](#Evaluation-code)
  - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

In various machine learning and data science projects, audio data stands aside as a separate yet understudied domain. Quite often, one might come across a study that treats audio as image data without taking its specificity into consideration. As it is hard to label audio recordings, various data augmentation techniques are crucial in this domain, just like in others. However, research on this topic is limited. In this work, multiple, supposedly most beneficial, audio data techniques are compared with each other and with a few novel ones, which consider the nature of an audio signal.

**Keywords**: Data Augmentation, Audio Data, Machine Learning, Audio Classification, Model Training.

**Full text**: The full text for this work is available [here](https://doi.org/10.18452/28010).

## Working with the repo

### Dependencies

The project was built using Python 3.9.6. Nonetheless, there should be little to no problem in reproducing it with a different version.

Code dependencies are stated in [requirements.txt](requirements.txt). All imports are managed in [setup.py](src/py/setup.py).

### Setup

1. Clone this repository
2. Install requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
(In some cases pip has to be substituted for pip3 or conda).

3. Run code in desired Jupyter Notebook

## Reproducing results

In order to reproduce the results you should do the following:

1. Upon cloning this repository, you should create a folder called Data. In it, you should place a copy of the [dataset](https://urbansounddataset.weebly.com/urbansound8k.html) used in this work.


2. After that, you can start experimenting with augmentation techniques and visualizations used in this work


3. In order to train a model, you can run [Models](src/Models.ipynb) Notebook. Note that in this work it was executed in Google Colab. In order to reuse it on your account, a copy of the preprocessed dataset (which is built after running [Preprocessing](src/Preprocessing.ipynb)Notebook) has to be uploaded to your Google Drive and made accessible to the Notebook execution environment. Execution on alternative platforms and/or local machines should follow similar steps.

4. As a result of executing [Models](src/Models.ipynb) Notebook, you will obtain a pickled dataframe (for example, like [this](EfNetV2B1Res.pkl)) that stores true and predicted labels for each augmentation technique. The datafame looks as follows:


|    | slice_file_name   |   fold | class           | source   | dist             | mixup           | imixup          | room            | spectrum         | warp             | delay           | all             |
|---:|:------------------|-------:|:----------------|:---------|:-----------------|:----------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|:----------------|
|  0 | 57320-0-0-39.wav  |      1 | air_conditioner | o        | children_playing | air_conditioner | air_conditioner | air_conditioner | children_playing | children_playing | dog_bark        | air_conditioner |
|  1 | 134717-0-0-6.wav  |      1 | air_conditioner | o        | street_music     | air_conditioner | air_conditioner | air_conditioner | air_conditioner  | engine_idling    | air_conditioner | engine_idling   |
|  2 | 57320-0-0-31.wav  |      1 | air_conditioner | o        | dog_bark         | air_conditioner | air_conditioner | dog_bark        | dog_bark         | dog_bark         | dog_bark        | air_conditioner |

5. Final metrics can be recalculated using [Evaluation](src/Evaluation.ipynb) Notebook.

### Training code

The training code is provided as part of the [Models](src/Models.ipynb) Notebook.

### Evaluation code

[Evaluation](src/Evaluation.ipynb) Notebook is responsible for metrics and resulting plots.

### Pretrained models

Models used in this work can benefit from fine-tuning pre-trained weights or built from scratch. In either case, there is no need to separately load weights, these can be downloaded automatically by a library used in this work. For more details, see:[TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

## Results

Proper evaluation of the prediction result is provided in thesis itself. However, [Evaluation](src/Evaluation.ipynb) Notebook contains excessive data that you can draw your conclusion from.

## Project structure
The project has a following structure:

```bash
├── Box Plot for EfficientNetV2B1 Accuracies.png    -- Example of the results
├── EfNetV2B1Res.pkl                                -- EfficientNetV2B1 predictions
├── EfNetV2B2Res.pkl                                -- EfficientNetV2B2 predictions
├── Examples                                     -- Real audio data examples subfolder
│   ├── Hall Recording Example.mp3                  
│   ├── New Recording.mp3                           
│   ├── Original Recording Example.mp3              
│   └── Street Recording Example.mp3               
├── README.md                                    
├── imgs                                         -- Subfolder storing high res plots
├── Data                                         -- subfolder storing data
│   ├── pkl                                      -- Subfolder with preprocessed data copies
│   └── UrbanSound8K                             -- Subfolder with original data
├── requirements.txt                                -- Library requirements
└── src                                          -- Code folder
    ├── Evaluation.ipynb                            -- Evaluation of model predictions
    ├── Models.ipynb                                -- Model training notebook
    ├── Motivation.ipynb                            -- Notebook explaining the domain
    ├── Preprocessing.ipynb                         -- Data preprocessing Notebook
    └── py                                       -- Subfolder for supporting .py files
        ├── augmenters.py                           -- Augmentation Techniques
        ├── batchproc.py                            -- Functions for Batch processing
        ├── helpers.py                              -- Plots and supporting functions
        └── setup.py                                -- Imports and global variable definitions
```
Note that Data and imgs folders are excluded from the repo due to their size.