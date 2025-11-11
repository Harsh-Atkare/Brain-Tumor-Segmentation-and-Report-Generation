# 🧠 AI-Powered Automated Radiology Report Generator

This project provides an end-to-end pipeline for **automated brain tumor analysis and clinical report generation** using MRI scans.  
It combines **deep learning segmentation**, **radiomic feature extraction**, and **LLM-based report generation** to assist radiologists in rapid and accurate diagnosis.

---

## 🚀 Key Features

### 1. Automated 3D Tumor Segmentation
- **Multi-modal MRI analysis:** FLAIR, T1CE, T2  
- **Model:** 3D U-Net architecture for precise segmentation  
- **Sub-region identification:**  
  - 🔵 Enhancing tumor  
  - 🔴 Necrotic core  
  - 🟢 Peritumoral edema  
- **Processing time:** ~2 minutes per case (vs. 30–45 minutes manually)  
- **Accuracy:** 87%+ Dice coefficient  

---

### 2. Comprehensive Clinical Feature Extraction
- **19 quantitative parameters** automatically extracted  
- **Volumetric measurements:** Total tumor, core, enhancing, necrotic, edema  
- **Dimensional analysis:** Maximum diameter, centroid coordinates  
- **Enhancement characteristics:** Pattern, intensity, percentage  
- **WHO grade indicators:** Automated assessment based on imaging features  
- **Anatomical localization:** Hemisphere, region, and functional implications  

---

### 3. AI-Powered Medical Report Generation
- **Model:** Qwen3-Coder-30B-A3B-Instruct (via Hugging Face API)  
- **Structured medical formatting includes:**
  - Professional headings (**SECTION NAME**)  
  - Bullet points with indentation  
  - Clinical tables and structured assessments  
  - WHO grade analysis  
- **Clinical priority stratification:** HIGH / MODERATE / ROUTINE  
- **Treatment urgency recommendations:** Suggested intervention timeline  
- **Differential diagnosis:** Based on imaging characteristics  
- **Generation time:** 30–60 seconds  

---

### 4. Advanced Visualization Suite
- **Color-coded segmentation overlays:**
  - 🔴 Red: Necrotic core  
  - 🟢 Green: Peritumoral edema  
  - 🔵 Blue: Enhancing tumor  
- **Multi-slice visualization:** Automatic selection of most relevant slices  
- **3D volume rendering:** Multi-planar (sagittal, coronal, axial) views  
- **Clinical dashboards:** Interactive charts and graphs  
- **Medical-grade quality:** 150–200 DPI output  

---

### 5. Professional PDF Reports
- Comprehensive documentation including:
  - Embedded visualizations and charts  
  - Quantitative analysis tables  
  - Clinical recommendations  
  - Technical specifications  
  - Medical disclaimers for clinical use  

---

## 🧩 Tech Stack
- **Segmentation Model:** 3D U-Net  
- **Frameworks:** PyTorch, MONAI  
- **Visualization:** Matplotlib, Plotly, Mayavi  
- **LLM Integration:** Hugging Face Transformers API  
- **Report Generation:** ReportLab / FPDF  

---

## ⚙️ Installation

Follow these steps to set up the project locally:

---

### 1. Clone the Repository
```bash
git clone https://github.com/Harsh-Atkare/Brain-Tumor-Segmentation-and-Report-Generation.git
cd brain-tumor-analysis
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
```
Activate the environment:

Linux / macOS:
```bash
source venv/bin/activate
```
Windows:
```bash
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Download the Segmentation Model
 [Download Model from Google Drive](https://drive.google.com/file/d/1TQ5HFyZOmfTCsKIP2fS50M1AXflBTtw0/view?usp=sharing)

After downloading, place the model file in:

```bash
brain-tumor-analysis/data/models/
```
---
 
### 5. LLM API Integration
	1.	Get your Hugging Face API key from the Hugging Face website￼.
	2.	Create a new file named .env in the project root directory.
	3.	Copy all contents from .env.example into .env.
	4.	Paste your API key into the following variable:

### 6. Run the Application
```bash
uvicorn app.main:app --reload
```
## 7. Access the System

After running, Uvicorn will display a local host link in the terminal.
Open it in your browser to access the system interface.


## ⚠️ Disclaimer
This system is intended **for research and educational purposes only**.  
It should **not** be used as a substitute for professional medical judgment or diagnosis.

---
