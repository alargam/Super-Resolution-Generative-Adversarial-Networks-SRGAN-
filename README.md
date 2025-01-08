# Swift SRGAN Project

This project implements a Swift SRGAN (Super-Resolution Generative Adversarial Network) for image super-resolution. The goal is to enhance the resolution of low-resolution images using deep learning techniques. The project is structured into several Python files, each handling a specific part of the pipeline, from data preparation to model training and evaluation.

## Project Structure

The project consists of the following files:

1. **custom_loss.py**: Defines the custom loss functions used for training the generator and discriminator in the SRGAN model.
2. **make_dataset.py**: Handles the downloading and extraction of the dataset.
3. **model_architecture.py**: Contains the architecture definitions for the generator and discriminator models.
4. **model_metrics.py**: Implements the Structural Similarity Index (SSIM) metric for evaluating the quality of super-resolved images.
5. **prepare_data.py**: Prepares the dataset by applying transformations and creating data loaders for training and validation.
6. **requirements.txt**: Lists the Python dependencies required to run the project.
7. **split_data.py**: Splits the dataset into training and validation sets.
8. **train_model.py**: Contains the main training loop for the SRGAN model, including checkpointing and evaluation.
9. **config.py**: Contains configuration settings for the Streamlit app, including page names and sample image paths.
10. **streamlit_app.py**: The main Streamlit application file that handles the UI and user interactions.

## File Descriptions

### 1. custom_loss.py
This file defines the custom loss functions used in the SRGAN model:
- **GeneratorLoss**: Combines adversarial loss, perceptual loss, naive image loss, and total variation (TV) loss to train the generator.
- **TVLoss**: Implements the total variation loss, which encourages smoothness in the generated images.

### 2. make_dataset.py
This script handles the downloading and extraction of the dataset from a zip file. It checks if the zip file exists and extracts it to the specified directory.

### 3. model_architecture.py
This file defines the architecture of the SRGAN model:
- **SeperableConv2d**: Implements a depthwise separable convolutional layer.
- **ConvBlock**: Defines a convolutional block with optional batch normalization and activation.
- **UpsampleBlock**: Implements an upsampling block using pixel shuffle.
- **ResidualBlock**: Defines a residual block with two convolutional layers.
- **Generator**: The generator network that takes a low-resolution image and outputs a high-resolution image.
- **Discriminator**: The discriminator network that distinguishes between real and generated high-resolution images.

### 4. model_metrics.py
This file implements the Structural Similarity Index (SSIM) metric, which is used to evaluate the quality of the super-resolved images. SSIM measures the similarity between two images based on luminance, contrast, and structure.

### 5. prepare_data.py
This script prepares the dataset for training and validation:
- **TrainDataset**: Loads and transforms the training dataset.
- **ValDataset**: Loads and transforms the validation dataset.
- **train_hr_transform** and **train_lr_transform**: Define the transformations applied to high-resolution and low-resolution images, respectively.

### 6. requirements.txt
This file lists the Python dependencies required to run the project, including libraries like `torch`, `torchvision`, `Pillow`, and `tqdm`.

### 7. split_data.py
This script splits the dataset into training and validation sets. It shuffles the dataset and saves the split indices into pickle files.

### 8. train_model.py
This file contains the main training loop for the SRGAN model:
- **save_checkpoint**: Saves the model state and training metrics to a checkpoint file.
- **load_checkpoint**: Loads the model state and training metrics from a checkpoint file.
- **run_pipeline**: Initializes the models, loss functions, and optimizers, and runs the training loop. It also handles validation, checkpointing, and early stopping.

### 9. config.py
This file contains configuration settings for the Streamlit app:
- **PAGES**: A list of page names for the Streamlit sidebar.
- **SAMPLE_IMAGES**: A list of sample image paths that can be used in the app.

### 10. streamlit_app.py
This is the main Streamlit application file:
- **load_data()**: Loads sample images and initializes the model for inference.
- **run_UI()**: Displays the UI based on the selected page, including the home page, image enhancer examples, and the option to try your own image.

## Usage

To train the SRGAN model, run the following command:

```bash
python train_model.py --upscale_factor 4 --num_epochs 100 --batch_size 32
