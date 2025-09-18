# Brain-Tumor-Segmentation-and-Report-Generation

## Overview

This repository contains a project for brain tumor segmentation using deep learning techniques on MRI images and automatic generation of medical reports based on the segmentation results. The goal is to assist in the diagnosis and documentation of brain tumors by automating the segmentation process and creating standardized reports for clinical use.

The project leverages convolutional neural networks (CNNs) for accurate tumor detection and segmentation, followed by natural language processing or template-based methods to generate comprehensive reports including tumor location, size, type, and recommendations.

## Features

- Automatic segmentation of brain tumors from MRI scans (T1, T2, FLAIR modalities).
- Classification of tumor types (e.g., glioma, meningioma, pituitary).
- Generation of PDF or text-based medical reports with visualizations.
- Evaluation metrics such as Dice coefficient, IoU, and accuracy.
- Support for popular datasets like BraTS (Brain Tumor Segmentation Challenge).

## Technologies Used

- **Programming Language**: Python 3.8+
- **Deep Learning Frameworks**: TensorFlow/Keras or PyTorch
- **Libraries**:
  - NumPy, Pandas for data handling
  - OpenCV or scikit-image for image processing
  - Matplotlib/Seaborn for visualization
  - ReportLab or similar for PDF report generation
  - NiBabel for handling NIfTI files (medical imaging format)
- **Hardware**: GPU recommended (NVIDIA with CUDA support)

## Dataset

The project uses the BraTS dataset, which includes multi-modal MRI scans (T1, T1ce, T2, FLAIR) of patients with brain tumors. Ground truth segmentations are provided for training and evaluation.

- Download from: [BraTS Challenge Website](https://www.med.upenn.edu/cbica/brats/)
- Preprocessing: Skull stripping, normalization, and augmentation (rotation, flipping).


## Results

- Achieved Dice Score: ~0.85 for whole tumor segmentation on BraTS validation set.
- Report generation includes tumor volume, location (e.g., frontal lobe), and confidence scores.
- Visualizations: Overlay of segmentation mask on original MRI slices.

See `results/` directory for sample outputs and metrics plots.
<img width="838" height="690" alt="image" src="https://github.com/user-attachments/assets/046f0ff6-1159-4269-86e9-5acd1149ecc1" />
<img width="790" height="506" alt="image" src="https://github.com/user-attachments/assets/406136a0-d9f5-47ea-91b9-0adbc404391d" />
<img width="793" height="547" alt="image" src="https://github.com/user-attachments/assets/782e5338-4ce1-4154-9881-dbf95a7ce859" />

<img width="682" height="340" alt="image" src="https://github.com/user-attachments/assets/7b9c9921-dd6e-4a5c-be2f-875c3a16bbe4" />
<img width="816" height="347" alt="image" src="https://github.com/user-attachments/assets/b4affd1e-c46d-42f7-94b2-d6800c5c0ead" />


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue first.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BraTS dataset providers
- Open-source libraries and pre-trained models
- Harsh Atkare for developing the initial project

For questions or issues, open a GitHub issue or contact harsh.atkare@example.com.

