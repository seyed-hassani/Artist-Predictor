# Artist Style Classification with Deep Learning

This project investigates how well neural networks can interpret artistic style. Using transfer learning and image data augmentation, we train a deep convolutional neural network to classify paintings by famous artists.

The notebook `artist.ipynb` walks through every step — from preprocessing to prediction — using the **ResNet50** architecture and TensorFlow/Keras.

---

##  Dataset

The dataset consists of hundreds of paintings by 25 renowned painters. Each image filename encodes the artist's name (e.g., `Vincent_van_Gogh_13.jpg`). There is a separate test set for evaluation.

###  Download Dataset

Download the dataset using the command below in your Google Colab environment:

!gdown 1-0d315aj7Ai8NNqat65XDvaOcHDcHiUD
!unzip famous_paintings.zip > /dev/null 2>&1

##  Key Features

- **  Dataset Organization**: Images are automatically organized into class-based folders to streamline training.
- **  Preprocessing**: Utilizes `image_dataset_from_directory` and `LabelEncoder` for efficient data loading and label encoding.
- **  Transfer Learning**: Built upon the pre-trained `ResNet50` model for effective feature extraction.
- **  Data Augmentation**: Enhances model robustness through random flipping, rotation, brightness, zoom, and contrast transformations.
- **  Fine-Tuning**: Supports partial or full fine-tuning of the top `ResNet` layers to improve performance.
- **  Evaluation Metrics**: Measures performance using Accuracy, F1-score, Precision, and Recall.
- **  Class Mapping**: Ensures correct label-to-artist name mapping using `class_names` for accurate prediction outputs.

---

##  Model Architecture

- **Base Model**: `ResNet50` with `include_top=False` and `imagenet` pre-trained weights.
- **Classification Head**:
  - `GlobalAveragePooling2D`
  - Dense layers with `ReLU` activation
  - `BatchNormalization` and `Dropout` for regularization
  - Final output layer with `softmax` activation for multi-class classification
- **Optimizer**: Adam with:
  - Learning rate scheduling (`ReduceLROnPlateau`)
  - Early stopping (`EarlyStopping`) for optimal training convergence

---

##  Training Visualization

Training progress is visualized using Plotly for both loss and accuracy metrics:


##  Evaluation & Submission

-  Achieved validation accuracy: **~60%**
-  Predictions on the test set are mapped to artist names using `class_names`.
-  Output saved to `submission.csv` in the following format:

-  All results are compressed into `result.zip` for submission or sharing.

---

##  Project Files

- `artist.ipynb` – Full Jupyter Notebook containing code for training, evaluation, and prediction  
- `submission.csv` – Output predictions for the test dataset  
- `result.zip` – Compressed archive for final submission  

---

##  Artists Covered

The model is trained to recognize works from 25 celebrated artists, including:

- Pablo Picasso  
- Vincent van Gogh  
- Rembrandt  
- Leonardo da Vinci  
- Frida Kahlo  
- Salvador Dali  
- Andy Warhol  
- Edgar Degas  
- Henri Matisse  
- Paul Klee  
- *...and more*

---

##  How to Run

1. Open `artist.ipynb` in Google Colab  
2. Download and unzip the dataset  
3. Execute training and validation cells  
4. Generate predictions and export the submission  

---

##  License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

##  Acknowledgments

Gratitude to open-source painting archives, Kaggle, and the deep learning community for the inspiration and datasets that made this project possible.
