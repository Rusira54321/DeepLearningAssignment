# Brain Tumor MRI Classification Models

This repository contains four TensorFlow/Keras Jupyter notebooks for classifying brain MRI images into four categories. The notebooks compare a custom convolutional neural network with three ImageNet-pretrained architectures.

## Models

| Notebook | Architecture | Approach |
|---|---|---|
| `custom_cnn.ipynb` | Custom CNN | Three convolution and max-pooling blocks (32, 64, and 128 filters), followed by a 256-unit dense layer and four-class softmax output. |
| `EfficientNet-B0.ipynb` | EfficientNet-B0 | ImageNet-pretrained feature extractor with global average pooling and a four-class head trains the head and includes a fine-tuning stage. |
| `DenseNet.ipynb` | DenseNet121 | ImageNet-pretrained feature extractor with global average pooling and a four-class head includes head training and fine-tuning. The notebook also contains an optional Ray Tune hyperparameter-search section. |
| `brain_tumor_resnet50_V4.ipynb` | ResNet50 | ImageNet-pretrained backbone, global average pooling, dense/dropout classification head, then fine-tuning of the final backbone layers. |

All four notebooks resize images to 224 x 224 and train/evaluate against the same labels: `glioma`, `meningioma`, `pituitary`, and `notumor`.

## Dataset

Download the **Brain Tumor MRI Dataset** from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Accept Kaggle's dataset terms and sign in if prompted. The archive should contain `Training/` and `Testing/` directories, each with one subdirectory per class:

```text
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

The notebooks use a preprocessing workflow that crops MRI backgrounds, creates a validation split from the training images, resizes images, and loads them from class-named folders. Their expected processed structure is:

```text
Cropped/
├── Train/<class-name>/
├── Validation/<class-name>/
└── Testing/<class-name>/
```

## Run in Google Colab

The notebooks currently contain Google Colab upload cells and Colab-specific file paths. Google Colab is the easiest way to run them, and a GPU runtime is recommended for training.

1. Download or clone this repository, then upload one notebook to [Google Colab](https://colab.research.google.com/) using **File > Upload notebook**.
2. In Colab, select **Runtime > Change runtime type** and choose a GPU accelerator if one is available.
3. Download the dataset ZIP from Kaggle to your computer.
4. Run the notebook from the first cell, in order. When its `files.upload()` cell appears, upload the dataset ZIP.
5. Check the ZIP filename expected by that notebook (table below). If needed, rename your downloaded ZIP before uploading, or update the filename in the notebook's ZIP extraction cell.
6. Continue running the cells in order through preprocessing, training, and test-set evaluation. Allow the notebook to install any packages requested by its cells.

| Notebook | ZIP filename expected by its extraction cell |
|---|---|
| `custom_cnn.ipynb` | `archive (2).zip` |
| `DenseNet.ipynb` | `Data set.zip` |
| `EfficientNet-B0.ipynb` | `archive.zip` |
| `brain_tumor_resnet50_V4.ipynb` | `archive.zip` |

### Dataset path note

The notebooks were developed in Colab and their hard-coded paths are not fully consistent. If a dataset-loading cell reports that it cannot find `Training` or `Testing`, inspect where the ZIP extracted and update that notebook's `dataset_path` variables to match. For example, if extraction creates `/content/Data_set/Training` and `/content/Data_set/Testing`, set the raw dataset path to `/content/Data_set`. After the notebook's crop and split steps, point the image-loading cell to the directory containing `Train`, `Validation`, and `Testing` (typically `/content/Cropped` or `/Data_set/Cropped`). Run the notebook's diagnostic folder-listing cells to confirm these directories exist before continuing.

The ResNet50 notebook attempts to detect an extra nested `Data set` directory. The DenseNet notebook assumes that nested directory exists. Check the extracted layout and adjust its raw `dataset_path` if Kaggle's ZIP has a different folder nesting.

## Run locally

These notebooks are not standalone Python scripts and do not include a pinned `requirements.txt`. To run locally, install Jupyter Notebook or JupyterLab and the dependencies used by the notebooks (TensorFlow/Keras, NumPy, Pillow, OpenCV, `imutils`, `tqdm`, scikit-learn, Matplotlib, and Seaborn). DenseNet's optional tuning section additionally uses Ray Tune (`ray`). A compatible TensorFlow installation and sufficient RAM/GPU memory are needed. Consult the [TensorFlow installation guide](https://www.tensorflow.org/install) for your operating system and hardware.

Then launch Jupyter from the repository folder:

```bash
jupyter lab
```

Open the notebook you want to run and follow **Run in Google Colab**, adapting the dataset paths to your local extracted dataset. Remove or skip Colab-only `google.colab.files` upload cells and provide a local path to the downloaded dataset instead. Run cells from top to bottom in a fresh kernel.

## Training and evaluation

Each notebook builds and trains its own model. Training may take a while a GPU is recommended. The notebooks use training, validation, and test data, and report classification metrics and/or plots near the end. Compare the models using the test-set results from each notebook, keeping the same dataset split and class ordering when interpreting the comparison. Do not compare training accuracy as if it were test performance.

The notebooks do not all save outputs in the same way. Some create a best-model checkpoint during training; the ResNet50 notebook also writes result files under `results/` and its checkpoint under `saved_models/`. Colab files are temporary unless downloaded or copied to persistent Drive storage. Download any checkpoint or result artifacts you want to retain before the Colab session ends.

## Notes

- Run each notebook from the beginning in a fresh runtime. Later cells depend on variables and preprocessing produced earlier.
- The dataset archive is not included in this repository. Download it directly from Kaggle.
- Pretrained Keras backbones may download ImageNet weights on first use, requiring an internet connection.
- This is an academic image-classification project, not a medical diagnostic device. Model predictions should not be used for clinical decisions.
