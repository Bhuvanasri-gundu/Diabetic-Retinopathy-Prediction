# Diabetic Retinopathy Prediction

## Project Overview
DR Vision is a Streamlit-based application for diabetic retinopathy severity assessment using retinal fundus images and patient clinical information. The application integrates image-only and multimodal prediction workflows with Grad-CAM++ explainability and PDF report generation into a unified interface for research and educational purposes.

## Prediction Classes

| Grade | Severity |
|-------|----------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative |

## Key Features
- Interactive Streamlit-based web application
- Retinal fundus image analysis
- Image-only and multimodal prediction workflows
- Clinical data collection through structured forms
- Grad-CAM++ visual explanations
- Prediction confidence scores
- Probability distribution visualization
- PDF report generation
- Responsive and intuitive user interface

## Project Structure
```text
.
├── app.py
├── dr_backend.py
├── requirements.txt
├── data/
│   ├── clinical_data_processed.csv
│   └── clinical_data.csv
├── Models/
│   ├── best_model.pth
│   └── best_multimodal_model.pth
├── notebooks/
├── processed_images/
└── .streamlit/
```

## Technologies Used
- Python
- Streamlit
- PyTorch
- TorchVision
- OpenCV
- Pillow
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- fpdf2

## Installation Instructions
Create and activate a Python environment, then install the required dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application
From the project root, run:

```bash
streamlit run app.py
```

This launches the interactive web application for image upload, clinical data entry, inference, and report generation.

## Application Workflow
1. Launch the Streamlit application.
2. Choose the analysis mode:
   - Multimodal (Image + Clinical)
   - Image Only
3. Upload a retinal image.
4. Enter the relevant clinical information in the form.
5. Run the analysis.
6. Review the predicted severity class, confidence, and Grad-CAM++ overlay.
7. Generate or view the PDF report for documentation.

## Model Information
The project includes two trained model variants:

- Image-only model: built around an EfficientNet-B4-based classifier and loaded from Models/best_model.pth
- Multimodal model: combines image features with a clinical feature encoder and is loaded from Models/best_multimodal_model.pth

The backend defines five classes for diabetic retinopathy severity:
- No DR
- Mild
- Moderate
- Severe
- Proliferative

## Explainability (Grad-CAM++)
Grad-CAM++ is used to generate visual explanations for the model prediction. The system produces a heatmap overlay on the processed retinal image so that the image regions most influential to the prediction can be inspected. These heatmaps are displayed in the interface and included in the generated PDF report.

## Dataset Information
The repository includes clinical data in the data directory, including:
- data/clinical_data_processed.csv
- data/clinical_data.csv

The preprocessing pipeline uses categorical and numerical clinical fields such as Gender, Smoking_Status, Hypertension, Age, Duration_DM_Years, HbA1c, Fasting_Glucose_mg_dL, BMI, Systolic_BP, Diastolic_BP, Total_Cholesterol_mg_dL, and Serum_Creatinine_mg_dL. The image preprocessing workflow includes border removal, center-focused cropping, circular masking, CLAHE-based enhancement, and resizing.

## Project Requirements
This project requires a Python environment with the packages listed in requirements.txt. In practice, the application depends on:
- A working Python installation
- The packages from requirements.txt
- Access to the model weight files in the Models directory
- Retinal image input files and clinical data files stored in the repository directories
