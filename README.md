
# 🌿 Crop Disease Detection using Hybrid CNN-SVM Model

## 📌 Project Overview
This project focuses on the development of an intelligent plant disease detection system using a hybrid deep learning and machine learning approach. The primary aim is to assist farmers in the early identification of crop diseases by analyzing plant leaf images and providing accurate classification results.

Agriculture plays a vital role in the global economy, yet plant diseases lead to significant losses in crop yield and quality. Traditional disease detection methods rely on manual inspection by experts, which is time-consuming, subjective, and not scalable for large agricultural fields. To overcome these limitations, this project leverages advancements in Artificial Intelligence, Machine Learning, and Computer Vision.

The proposed system combines the strengths of Convolutional Neural Networks (CNN) and Support Vector Machines (SVM). The CNN model is used as a feature extractor, capturing important visual characteristics such as texture, color patterns, and leaf structure from input images. Instead of performing direct classification, the extracted features are passed to an SVM classifier, which creates an optimal decision boundary for accurate prediction.

The model is trained and evaluated using the PlantVillage dataset, consisting of approximately 52,000 labeled images. A strict leakage-free 80–20 training-validation split is maintained to ensure reliable performance evaluation. The hybrid model achieves a high classification accuracy of 99.50%, outperforming the standalone CNN model.

Additionally, the system is tested under noisy and distorted conditions to evaluate robustness, making it suitable for real-world agricultural environments. The overall solution is computationally efficient and can be deployed on standard systems, making it practical for widespread use.

This project demonstrates how integrating deep learning with traditional machine learning techniques can result in a more accurate, robust, and scalable solution for smart agriculture applications.

---

## 🎯 Objectives
- Develop an automated plant disease detection system  
- Use CNN for feature extraction  
- Use SVM for classification  
- Improve accuracy and generalization  
- Ensure leakage-free dataset handling  
- Evaluate robustness under real-world conditions  

---

## 🧠 Methodology

### 🔹 1. Data Preprocessing
- Image resizing  
- Pixel normalization (0–1 range)  
- 80–20 train-validation split  

### 🔹 2. CNN Feature Extraction
- 3 Convolutional layers  
- ReLU activation  
- Pooling layers  
- Output: **128-dimensional feature vector**  

### 🔹 3. SVM Classification
- Kernel: **RBF (Radial Basis Function)**  
- Classifies images into:
  - Healthy  
  - Diseased  

<img width="676" height="502" alt="{7AB21EDC-1DBD-4C7F-9365-6E0496564985}" src="https://github.com/user-attachments/assets/f0382c43-0278-4a2f-a219-4eafb0553e6b" />


## 📊 Results

| Model        | Accuracy |
|-------------|---------|
| CNN         | 99.17%  |
| CNN-SVM     | **99.50%** |

<img width="610" height="468" alt="{8433A3A6-8907-49B1-AC36-86F0BBB77B38}" src="https://github.com/user-attachments/assets/859488cf-650d-496f-a11c-c3a57b672b07" />
<img width="570" height="474" alt="{C0AE3993-7CFA-4F54-B373-FCBBE1D63E7F}" src="https://github.com/user-attachments/assets/fed50523-28ba-4cf2-bc50-da5fb53ca9f2" />
<img width="920" height="609" alt="{0AADCAFA-3B82-43B4-A0E1-9152E392DF16}" src="https://github.com/user-attachments/assets/0c64d1c1-4cad-4165-bf6e-e917e6091d29" />


### 🔍 Additional Performance
- Cross-validation accuracy: ~99.51%  
- Accuracy under noise: 99.37%  
- ROC-AUC Score: **> 0.999**
  <img width="511" height="425" alt="{D8309DA3-52D2-4ED9-964B-E67627B383E4}" src="https://github.com/user-attachments/assets/6ee288bf-5914-45fa-a344-3f968ac0abb4" />


---

## 📁 Project Structure
<img width="250" height="831" alt="{9A6263AD-C2F9-47F0-913C-A30EBE75B7A4}" src="https://github.com/user-attachments/assets/1f703e48-7203-4a87-9f57-f26de4b8d053" />
