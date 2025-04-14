# Video Violence Detection Project

This project aims to classify videos into two categories: **Violent** and **Non-Violent**. The model utilizes a combination of **Temporal Convolutional Networks (TCN)**, **ResNet-50** for feature extraction, and a **BiLSTM (Bidirectional Long Short-Term Memory)** network for classification.

The pipeline consists of feature extraction from video frames, followed by the use of two models (TCN/ResNet50 and BiLSTM) to classify the video as either violent or non-violent. 

The model has achieved **96% accuracy** in predicting violent and non-violent videos.

## Key Features

- **Custom Video Prediction**: A cell in the notebook allows users to input custom videos for real-time predictions.
- **Best Model Files**: The models used are:
  - `best_model.pth`: A combination of TCN/ResNet50 architecture used for feature extraction.
  - `trained_model.pth`: A BiLSTM-based model for classification.
  
The project has been compared with several other datasets (details below) to benchmark its performance.

## Pipeline Overview

1. **Feature Extraction**:  
   - Videos are processed frame-by-frame, and features are extracted using a pre-trained **ResNet-50** model.
   - These features capture the spatial information within each frame.
   
2. **Temporal Modeling**:  
   - A **TCN (Temporal Convolutional Network)** is used to model the temporal dynamics between frames in the video. This model captures the motion patterns over time, crucial for understanding actions in the video.
   
3. **Classification**:  
   - The temporal features from the TCN are passed to a **BiLSTM** model that classifies the video as **Violent** or **Non-Violent** based on the extracted features. The BiLSTM captures both forward and backward dependencies in the video sequence for better context understanding.

4. **Prediction**:  
   - Once the model is trained, the `best_model.pth` and `trained_model.pth` are used to predict the class of a video, either **Violent** or **Non-Violent**.

## How the Approach Works

Our approach is unique because it integrates **spatial-temporal feature extraction** using **ResNet-50** and **TCN**, followed by **BiLSTM** for classification. This combination allows the model to capture both **short-term motions** and **long-term temporal dependencies**, providing a context-aware prediction that improves classification accuracy.

- The **ResNet-50** model is pre-trained on ImageNet and is used to extract robust spatial features from individual frames of the video.
- The **TCN** component helps capture motion patterns across frames, identifying sequences and temporal relationships in the video.
- The **BiLSTM** model further enhances the ability to understand sequences by looking at the video in both forward and backward directions.

This combination allows our model to consider the context of actions in the video, rather than simply analyzing isolated frames, which leads to more accurate predictions.

## Dataset Comparison

Our model was tested on several datasets to validate its performance, and the results were compared with the following:

- [UCF Crimes Dataset](https://www.kaggle.com/datasets/tesisjulinwilson/ucf-crimes-dataset) - Testing
- [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) - Training BiLSTM model for violence and non-violence classification
- [UCF101 - Action Recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) - Training TCN/Resnet50 model for acion prediction

These datasets provided a diverse set of videos from different crime-related activities, allowing us to benchmark our model and compare its performance against other state-of-the-art models.

## Model Pipeline

```text
Step 1: Video Input
  Input: Raw video file (.mp4 / .avi)
  Output: Video stream to be processed

Step 2: Frame Extraction
  Process: Extract 1–2 frames per second
  Output: Sequence of image frames

Step 3: Frame Preprocessing
  Resize: 224 × 224 pixels
  Normalize: Using ImageNet mean and std
  Output: Cleaned & normalized frames

Step 4: Feature Extraction (ResNet50)
  Model: Pretrained ResNet50 (final layer removed)
  Output: Each frame → 2048-D feature vector → (N, 2048)

Step 5: Sliding Window Segmentation
  Window Size: 10 frames
  Stride: 5 frames
  Output: Segments shaped (10, 2048)

Step 6: Temporal Modeling
  Options:
    - BiLSTM: Learns bidirectional time dependencies
    - TCN: Uses dilated causal convolutions
  Output: Encoded segment representations

Step 7: Segment Classification
  Each segment is classified as:
    - Violent → 1
    - Non-Violent → 0

Step 8: Video-Level Decision
  Method: Majority vote over segments
  Final Output: "Violent" or "Non-Violent"

# 📘 How to Run This Notebook

This section provides detailed instructions to run `violence_detection.ipynb` in different environments: **Kaggle**, **Google Colab**, and **locally**. Each platform has its own setup method depending on file access, environment persistence, and dependencies.

---

## 🟢 Running on Kaggle (Recommended)

Kaggle offers a seamless experience with persistent environments and automatic handling of files and variables.

### ✅ Why Kaggle is Ideal:

- **Built-in persistence**: Files, variables, and outputs are saved throughout the session.
- **No manual download required**: When you upload the dataset, model weights (`trained_model.pth`, `best_model.pth`), and notebook in the same session, Kaggle retains them.
- **Pre-installed dependencies**: Most common packages (e.g., PyTorch, OpenCV, tqdm) are already installed.

### ▶️ Steps:

1. Visit [https://www.kaggle.com](https://www.kaggle.com).
2. Create a new notebook.
3. Upload the following files:
   - `violence_detection.ipynb`
   - `trained_model.pth` (BiLSTM)
   - `best_model.pth` (TCN)
   - Dataset folder (videos or preprocessed frames)
4. Set your runtime to GPU for faster performance (optional but recommended).
5. Click **“Run All”** — no additional setup needed.

> 📌 **Note:** As long as all files are uploaded together, **no path changes** are required. Kaggle automatically handles storage and file access.

---

## ☁️ Running on Google Colab

Colab is a flexible option but requires some manual setup for file and dependency management.

### ⚠️ Setup Required:

- **Dependencies must be installed manually** using `pip`.
- **No file persistence** between sessions (unless you mount Google Drive).
- **Dataset and model weights must be downloaded** from the provided link.

### ▶️ Steps:

1. Go to [https://colab.research.google.com](https://colab.research.google.com).
2. Upload or open `violence_detection.ipynb`.
3. Install required dependencies (run this in the first cell):
   ```python
   !pip install torch torchvision opencv-python tqdm

