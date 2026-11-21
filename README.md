# FoodAI – Intelligent Food Photo and Recipe Finder

## Project Overview
FoodAI is an intelligent application that predicts food categories from uploaded images and recommends recipes based on text queries.

This project was developed as part of the SWE015 course in Computer Engineering.

## Features
- Food image classification using CNN and VGG16.
- Prediction confidence scores.
- Grad-CAM heatmap visualization.
- Text-based recipe search.
- TF-IDF and Transformer-based recipe retrieval.
- Quantized models for efficient inference.
- Model performance comparison using accuracy, model size, and latency.

## Technologies
- Python
- PyTorch
- Torchvision
- Streamlit
- Scikit-learn
- Sentence Transformers
- NumPy

## Dataset
The image classification task uses a 10-class subset of Food-101.

The recipe retrieval task uses a subset of the RecipeNLG dataset.

## Project Structure
- `app.py`: Streamlit application.
- `recipe_search.py`: Recipe retrieval and text processing.
- `benchmark.py`: Performance benchmarking.
- `download_data.py`: Dataset download utilities.
- `src/`: Model training and evaluation scripts.
- `models/`: Trained and quantized models.
- `data/`: Image and recipe datasets.
- `results/`: Evaluation results.

## How to Run

1. Clone the repository:

   `git clone https://github.com/zeynepsertbas10-commits/FoodAI.git`

2. Open the project folder:

   `cd FoodAI`

3. Create and activate a virtual environment.

4. Install dependencies:

   `pip install -r requirements.txt`

5. Run the application:

   `streamlit run app.py`

## Evaluation
Image classification models are evaluated using Top-1 and Top-5 accuracy.

Recipe retrieval models are evaluated using Recall@5 and Mean Reciprocal Rank (MRR).

Model size and inference latency are also measured.

## Student Information
- **Name:** Ayşe Zeynep Sertbaş
- **Student ID:** 220901588
- **Department:** Computer Engineering
