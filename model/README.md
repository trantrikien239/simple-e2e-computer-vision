# Project Title

This project contains a Convolutional Neural Network (CNN) model built with PyTorch, suitable for learning the MNIST classification task.

## File Structure

```
.
├── mnist_cnn.py
├── mnist_dataset.py 
└── train.py
```

- `mnist_cnn.py`: This file defines the CNN model using PyTorch. The model is designed to be suitable for the MNIST classification task.

- `mnist_dataset.py`: This file defines the MNIST dataset and the data loader. The data loader is responsible for loading the MNIST dataset in a format that can be used by the CNN model.

- `train.py`: This file defines a trainer class for training the classifier. After training, the model is saved as a safe tensor.

## Usage

To use these scripts, first ensure you have PyTorch installed. Then, you can run the `train.py` script to train the model on the MNIST dataset.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details