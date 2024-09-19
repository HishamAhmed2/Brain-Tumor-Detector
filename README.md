# Brain-Tumor-Detector
This project focuses on building a Convolutional Neural Network (CNN) to classify MRI images of brain tumors. The model differentiates between healthy individuals and patients with brain tumors based on their MRI scans. Data retrieved from open dataset on kaggle (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?resource=download)

## Model Architecture
1. **Convolutional Layers**: Two convolutional layers, each followed by a non-linear activation function (Tanh) and average pooling.
2. **Fully Connected Layers**: Three fully connected layers with Tanh activation functions, with the final output using a sigmoid activation to classify the image as either healthy or tumor.

### Model Layers:
- **Conv2D** (3 → 6 channels, kernel size = 5)
- **Tanh Activation**
- **AvgPool2D** (kernel size = 2, stride = 5)
- **Conv2D** (6 → 16 channels, kernel size = 5)
- **Tanh Activation**
- **AvgPool2D** (kernel size = 2, stride = 5)
- **Fully Connected Layer** (256 → 120)
- **Tanh Activation**
- **Fully Connected Layer** (120 → 84)
- **Tanh Activation**
- **Fully Connected Layer** (84 → 1)
- **Sigmoid Activation**
