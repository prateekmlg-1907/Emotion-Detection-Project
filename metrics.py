from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reinitialize the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False  # Ensure the order is consistent for metrics evaluation
)

# Load the saved model
model = load_model('models/emotion_detector.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_set, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Get true labels and predicted labels
y_true = test_set.classes  # True labels

# Get class indices to map predictions back to labels
class_indices = test_set.class_indices
class_labels = list(class_indices.keys())

# Predict the probabilities and get predicted classes
y_pred = model.predict(test_set, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
