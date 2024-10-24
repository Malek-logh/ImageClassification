# Image Classification with TensorFlow and OpenCV

This repository demonstrates an image classification project using TensorFlow and OpenCV. The goal is to classify images into different categories (e.g., "Happy" or "Sad"). The project uses a convolutional neural network (CNN) for binary classification, with OpenCV for image preprocessing.

## Requirements

To run this project, you will need the following dependencies:

- `opencv-python`: For image processing and manipulation.
- `matplotlib`: For visualizing training results and images.
- `tensorflow`: For building, training, and evaluating the machine learning model.

You can install the necessary libraries using the following commands:

```bash
pip install opencv-python
pip install matplotlib
pip install tensorflow
```

## Project Structure

- **`ImageClassification.ipynb`**: Jupyter notebook containing the code for data preprocessing, model building, training, and evaluation.
- **`Data/`**: Directory that contains the images for training and testing the model. Ensure the images are categorized into subdirectories by class (e.g., `Data/Happy` and `Data/Sad`).
- **`models/`**: Directory where the trained model is saved after training.

## Steps to Run the Model

1. **Install Dependencies**: Make sure you have installed all the required Python packages.

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**: Add your image dataset to the `Data/` directory, with subdirectories for each class.

3. **Run the Notebook**: Open `ImageClassification.ipynb` and execute the cells to:
   - Load and preprocess the images.
   - Define and compile the CNN model.
   - Train the model on the dataset.
   - Evaluate its performance on the test set.
   - Use the trained model to make predictions on new images.

4. **Visualize Training Results**: After training, the notebook plots the accuracy and loss during training and validation.

## Model Architecture

The Convolutional Neural Network (CNN) used in this project consists of:

- **3 Conv2D layers** with ReLU activation and MaxPooling.
- **Flatten** and **Dense** layers for classification.
- **Sigmoid activation** in the final layer for binary classification.

The model is compiled with the Adam optimizer and binary cross-entropy loss.

## Contributions

Feel free to fork this repository, create issues, or submit pull requests to contribute to this project.

## License

This project is licensed under the MIT License.

