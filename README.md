# CranioDental AI — Forensic Diagnostic Workstation

**Live Application:** [https://craniodental-ai.streamlit.app](https://craniodental-ai.streamlit.app)

CranioDental AI (IPCTI Diagnostic Engine) is a multimodal deep learning forensic workstation designed for independent and combined **chronological age estimation** and **biological sex classification** using **Dental CBCT views** and **3D Cranial CT scans**.

---

## Key Features

- **Independent & Multimodal Inference:** Run Dental Only, Skull Only, or Multimodal (Both) analysis from a unified dark-themed forensic dashboard.
- **Dental 4-View CBCT Engine:** Processes Coronal (`11`, `21`) and Sagittal (`11`, `21`) dental views using an EfficientNet-B3 backbone enhanced with Convolutional Block Attention Modules (CBAM).
- **Cranial 3D Volumetric Engine:** Supports pre-processed PyTorch tensors (`.pt`), NIfTI volumes (`.nii`, `.nii.gz`), single/multi-slice DICOMs (`.dcm`), and zipped DICOM series (`.zip`), applying Hounsfield Unit (HU) bone thresholding (`300–400 HU`) and 3-plane Maximum Intensity Projections (Axial, Coronal, Sagittal).
- **Explainable AI (XAI):** Generates gradient-based saliency heatmaps for dental scans with an interactive live thermal overlay opacity slider.
- **Clinical PDF Export:** One-click downloadable clinical forensic report (`CranioDent_Forensic_Report.pdf`) via `html2pdf.js`.

---

## Model Architecture & Evaluation Metrics

| Modality | Architecture | Backbone & Attention | Input Format | Accuracy | MAE | MSE | F1-Score |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Dental** | `DentalMultiTaskNet` | EfficientNet-B3 + CBAM | 4-View 2.5D CBCT (`.png`, `.jpg`) | 89.0% | ±2.6 Yrs | 10.24 | 83.6% |
| **Cranial** | `DecoupledForensicHead` | EfficientNet-V2-S + Spatial Attention | 3D CT Volume (`.pt`, `.nii.gz`, `.dcm`, `.zip`) | 100% | ±4.18 Yrs | 25.50 | 100% |

---

## Project Structure

```text
craniodental-ai/
├── app.py                              # Main Streamlit forensic workstation
├── model.py                            # DentalMultiTaskNet & CBAM architecture
├── core/
│   └── model.py                        # DecoupledForensicHead (Cranial architecture)
├── weights/
│   ├── age_expert_engine_2d.pth        # Cranial age prediction weights
│   └── sex_expert_engine_2d.pth        # Cranial sex classification weights
├── dental_multitask_model.pth          # Dental multi-task weights
└── requirements.txt                    # Python & DICOM codec dependencies
```

---

## Local Setup

```bash
git clone https://github.com/Anushka1813/craniodental-ai.git
cd craniodental-ai
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision (EfficientNet-B3, EfficientNet-V2-S, CBAM)
- **Medical Imaging:** PyDICOM, pylibjpeg, NiBabel, OpenCV, Pillow
- **Frontend & Deployment:** Streamlit, Custom HTML/CSS/JS, `html2pdf.js`, Streamlit Community Cloud

---

## Author

**Anushka Charpe**  
GitHub: [@Anushka1813](https://github.com/Anushka1813)
