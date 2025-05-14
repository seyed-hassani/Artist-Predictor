# 🎨 Artist Style Classification with Deep Learning

This project investigates how well neural networks can interpret artistic style. Using transfer learning and image data augmentation, we train a deep convolutional neural network to classify paintings by famous artists.

The notebook `artist.ipynb` walks through every step — from preprocessing to prediction — using the **ResNet50** architecture and TensorFlow/Keras.

---

## 📂 Dataset

The dataset consists of hundreds of paintings by 25 renowned painters. Each image filename encodes the artist's name (e.g., `Vincent_van_Gogh_13.jpg`). There is a separate test set for evaluation.

### 📥 Download Dataset

Download the dataset using the command below in your Google Colab environment:

```bash
!gdown 1-0d315aj7Ai8NNqat65XDvaOcHDcHiUD
!unzip famous_paintings.zip > /dev/null 2>&1
🛠️ Key Features
Data preprocessing using image_dataset_from_directory and label encoding

Transfer learning with ResNet50

Data augmentation (rotation, brightness, contrast, etc.)

Fine-tuning last layers of the pre-trained model

Metrics: Accuracy, F1-score, Precision, Recall

Prediction output converted to artist names with proper label mapping

🧠 Model Architecture
Base model: ResNet50 (include_top=False)

Custom classification head: Dense layers + BatchNorm + Dropout

Optimization: Adam optimizer with learning rate decay and early stopping

📈 Visualization
Training history is visualized using interactive Plotly charts for:

Loss vs Epochs

Accuracy vs Epochs

python
Copy
Edit
display_curves(history, 'loss')
display_curves(history, 'accuracy')
🧪 Evaluation & Submission
Validation accuracy: ~60%

Test predictions converted to readable artist names using class_names

Generates a submission.csv in the following format:

file	artist
test_001	Vincent_van_Gogh
test_002	Claude_Monet
...	...

Final outputs are zipped into result.zip for submission.

📦 Contents
artist.ipynb: Full Jupyter notebook

submission.csv: Output predictions for test images

result.zip: Compressed file for evaluation or upload

🧑‍🎨 Artists Included
Some of the painters in this dataset:

Pablo Picasso

Vincent van Gogh

Rembrandt

Leonardo da Vinci

Frida Kahlo

Salvador Dali

Andy Warhol

and many more...

🚀 Usage
Run artist.ipynb in Google Colab

Download and unzip the dataset

Train the model and evaluate performance

Generate predictions and save the submission

📄 License
MIT License. See LICENSE file for details.

🙌 Acknowledgments
Special thanks to open-source painting archives and Kaggle datasets that inspired this project.

