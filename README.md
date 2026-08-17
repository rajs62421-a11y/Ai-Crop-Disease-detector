# Crop Disease Prediction 🌱

A beginner-friendly crop disease prediction project using Python, TensorFlow/Keras, and Streamlit.

## Features

- Upload a crop leaf image
- Train a CNN image classifier
- Predict disease from a trained model
- Display prediction confidence
- Simple Streamlit web interface

## Project structure

```text
crop_disease_prediction/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── data/
│   ├── train/
│   └── validation/
└── models/
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare the dataset

Place your images into folders like:

```text
data/train/Healthy/
data/train/Tomato___Early_blight/
data/train/Tomato___Late_blight/

data/validation/Healthy/
data/validation/Tomato___Early_blight/
data/validation/Tomato___Late_blight/
```

Each folder represents one class.

## 3. Train the model

```bash
python train.py
```

The trained model will be saved as:

```text
models/crop_disease_model.keras
```

## 4. Test a single image

```bash
python predict.py path/to/leaf.jpg
```

## 5. Run the web app

```bash
streamlit run app.py
```

Then open the Streamlit address shown in the terminal.

## Important

The ZIP intentionally does not include a large image dataset or a pre-trained model. Add a legally obtained crop-leaf dataset and train the model before using it for real predictions.

For a college project, you can extend this with disease descriptions, treatment suggestions, accuracy graphs, confusion matrix, login page, database storage, and deployment.
