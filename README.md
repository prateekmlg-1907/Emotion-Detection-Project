
# Emotion Detection Project 🎭

This project enables real-time emotion detection using deep learning techniques. It includes scripts for training an emotion detection model and a main script for real-time emotion detection from a webcam feed. 🚀

## Project Structure 📂

```
Project
|-- data
|   |-- train
|   |   |-- class1
|   |   |-- class2
|   |   ...
|   |-- test
|       |-- class1
|       |-- class2
|       ...
|-- models
|   |-- emotion_detector.h5
|-- main.py
|-- metrics.py
|-- train.py
|-- requirements.txt
```

- **data**: Contains subdirectories for training and testing data. Each class has its own subdirectory, facilitating efficient data organization.
- **models**: Stores the trained models. The `emotion_detector.h5` file contains the trained model for emotion detection.
- **main.py**: The main script for real-time emotion detection using the trained model.
- **train.py**: Script for training the emotion detection model.

## Usage 🛠️

1. **Clone the Repository**: Run the following command in your terminal:

```bash
git clone https://github.com/yourusername/emotion-detection-project.git
```

2. **Install Dependencies**: Navigate to the project directory and install the required dependencies:

```bash
cd emotion-detection-project
pip install -r requirements.txt
```

3. **Train the Model**: Execute the `train.py` script to train the emotion detection model using the provided training data.

```bash
python train.py
```

4. **Real-time Emotion Detection**: Run the `main.py` script to start real-time emotion detection from the webcam feed.

```bash
python main.py
```

## Requirements 📋

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Keras

## Contribute 🤝

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## Contact 📧

For any questions or suggestions, feel free to reach out to [prateekmalagund@gmail.com](mailto:prateekmalagund@gmail.com).

Feel free to further customize the README.md as needed! 🌟
