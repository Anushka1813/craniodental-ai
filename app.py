import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import math
import os
import cv2
import numpy as np
import tempfile
import nibabel as nib
import time
import pydicom

from model import DentalMultiTaskNet
from core.model import DecoupledForensicHead

st.set_page_config(page_title="CranioDental AI Workstation", layout="wide")

device = torch.device("cpu")

@st.cache_resource
def load_models():
    dental_model = DentalMultiTaskNet(pretrained=False).to(device)
    dental_model.load_state_dict(
        torch.load("dental_multitask_model.pth", map_location=device, weights_only=True),
        strict=False
    )
    dental_model.eval()

    skull_model = DecoupledForensicHead().to(device)
    skull_model.load_state_dict(
        torch.load("weights/age_expert_engine_2d.pth", map_location=device, weights_only=True),
        strict=False
    )
    skull_model.eval()

    return dental_model, skull_model

dental_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_skull_scan(file_path):
    if file_path.endswith('.pt'):
        scan_tensor = torch.load(file_path, map_location='cpu', weights_only=True).to(torch.float32)
        if scan_tensor.dim() == 3:
            scan_tensor = scan_tensor.unsqueeze(0)
    elif file_path.endswith(('.nii', '.nii.gz', '.dcm')):
        if file_path.endswith('.dcm'):
            dicom_data = pydicom.dcmread(file_path)
            scan_tensor = torch.tensor(dicom_data.pixel_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            nii_img = nib.load(file_path)
            scan_tensor = torch.tensor(nii_img.get_fdata(), dtype=torch.float32)
            if scan_tensor.dim() == 3:
                scan_tensor = scan_tensor.unsqueeze(0)

        if scan_tensor.shape[1:] != (128, 128, 128):
            scan_tensor = scan_tensor.unsqueeze(0)
            scan_tensor = F.interpolate(scan_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
            scan_tensor = scan_tensor.squeeze(0)
    else:
        raise ValueError("Unsupported format. Upload .nii, .nii.gz, .dcm, or .pt")

    # HU Thresholding (Isolating bone)
    bone_mask = (scan_tensor >= 300) & (scan_tensor <= 400)
    scan_tensor = scan_tensor * bone_mask.float()

    vol = scan_tensor.squeeze(0)
    axial_proj = torch.max(vol, dim=0)[0]
    coronal_proj = torch.max(vol, dim=1)[0]
    sagittal_proj = torch.max(vol, dim=2)[0]

    multi_view_tensor = torch.stack([axial_proj, coronal_proj, sagittal_proj], dim=0)
    v_min, v_max = multi_view_tensor.min(), multi_view_tensor.max()
    multi_view_tensor = (multi_view_tensor - v_min) / (v_max - v_min + 1e-8)

    return multi_view_tensor.unsqueeze(0), [axial_proj, coronal_proj, sagittal_proj]


# ================= UI LAYOUT =================
st.title("CranioDental AI — Independent Dual-Modal Forensic Workstation")
st.caption("Upload Dental CBCT views, a 3D Cranial CT scan, or both for independent forensic profiling.")

col_d, col_s = st.columns(2)

with col_d:
    st.subheader("Dental Engine (CBCT 2.5D Views)")
    c1, c2 = st.columns(2)
    coronal_11 = c1.file_uploader("Coronal 11", type=["png", "jpg", "jpeg"], key="c11")
    coronal_21 = c2.file_uploader("Coronal 21", type=["png", "jpg", "jpeg"], key="c21")
    sagittal_11 = c1.file_uploader("Sagittal 11", type=["png", "jpg", "jpeg"], key="s11")
    sagittal_21 = c2.file_uploader("Sagittal 21", type=["png", "jpg", "jpeg"], key="s21")

with col_s:
    st.subheader("Cranial Engine (3D Skull CT)")
    skull_file = st.file_uploader(
        "Upload Skull Scan (.nii, .nii.gz, .dcm, .pt)",
        type=["nii", "gz", "dcm", "pt"],
        key="skull"
    )

run_btn = st.button("Run Forensic Analysis", type="primary", use_container_width=True)

if run_btn:
    dental_files = {
        'Coronal 11': coronal_11,
        'Coronal 21': coronal_21,
        'Sagittal 11': sagittal_11,
        'Sagittal 21': sagittal_21
    }

    has_dental = any(f is not None for f in dental_files.values())
    has_skull = skull_file is not None

    if not has_dental and not has_skull:
        st.error("Please upload Dental scans, a Skull scan, or both.")
    else:
        try:
            with st.spinner("Running independent neural inference..."):
                start_time = time.time()
                dental_model, skull_model = load_models()

                # ================= DENTAL PIPELINE =================
                if has_dental:
                    processed_images = []
                    active_files = {}

                    for label, file in dental_files.items():
                        if file is not None:
                            img_bytes = file.getvalue()
                            active_files[label] = img_bytes
                            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                            processed_images.append(dental_transform(img))

                    dental_batch = torch.stack(processed_images).unsqueeze(0).to(device)
                    dental_batch.requires_grad = True

                    outputs = dental_model(dental_batch)
                    d_age = outputs['age'].item()
                    d_sex_logit = outputs['gender'].item()
                    d_prob = 1.0 / (1.0 + math.exp(-d_sex_logit))

                    dental_age = round(d_age, 1)
                    dental_sex = "Male" if d_prob > 0.5 else "Female"
                    dental_conf = round((d_prob if d_prob > 0.5 else (1 - d_prob)) * 100, 1)

                    # Saliency Heatmap
                    dental_model.zero_grad()
                    outputs['age'].backward()
                    raw_saliency = dental_batch.grad.abs().squeeze(0).cpu().numpy()
                    raw_saliency = np.max(raw_saliency, axis=1)

                    st.divider()
                    st.subheader("Dental Engine Results")
                    m1, m2 = st.columns(2)
                    m1.metric("Estimated Age (Dental)", f"{dental_age} Years")
                    m2.metric("Predicted Biological Sex (Dental)", f"{dental_sex} ({dental_conf}%)")

                    img_cols = st.columns(len(active_files))
                    for idx, (label, img_bytes) in enumerate(active_files.items()):
                        img_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        img_cv = cv2.resize(img_cv, (224, 224))

                        saliency = raw_saliency[idx]
                        s_max = np.max(saliency)
                        if s_max > 0:
                            saliency = saliency / s_max
                        saliency = cv2.GaussianBlur(saliency, (15, 15), 0)
                        s_max = np.max(saliency)
                        if s_max > 0:
                            saliency = saliency / s_max

                        heatmap_color = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
                        blended = cv2.addWeighted(img_cv, 0.4, heatmap_color, 0.6, 0)
                        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                        orig_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                        with img_cols[idx]:
                            st.image(orig_rgb, caption=f"{label} (Original)", use_container_width=True)
                            st.image(blended_rgb, caption=f"{label} (Saliency)", use_container_width=True)

                # ================= SKULL PIPELINE =================
                if has_skull:
                    temp_path = os.path.join(tempfile.gettempdir(), skull_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(skull_file.getvalue())

                    skull_tensor, projections = process_skull_scan(temp_path)
                    skull_tensor = skull_tensor.to(device)
                    os.remove(temp_path)

                    with torch.no_grad():
                        s_age_norm, s_sex_logits = skull_model(skull_tensor)
                        s_age = s_age_norm.item() * 100.0
                        s_prob = torch.sigmoid(s_sex_logits).item()

                    skull_age = round(s_age, 1)
                    skull_sex = "Male" if s_prob >= 0.5 else "Female"
                    skull_conf = round((s_prob if s_prob >= 0.5 else (1 - s_prob)) * 100, 1)

                    st.divider()
                    st.subheader("Cranial Engine Results")
                    m1, m2 = st.columns(2)
                    m1.metric("Estimated Age (Cranial)", f"{skull_age} Years")
                    m2.metric("Predicted Biological Sex (Cranial)", f"{skull_sex} ({skull_conf}%)")

                    view_names = ['Axial Projection', 'Coronal Projection', 'Sagittal Projection']
                    v_cols = st.columns(3)
                    for idx, proj in enumerate(projections):
                        proj_img = proj.cpu().numpy()
                        proj_norm = cv2.normalize(proj_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        proj_colored = cv2.applyColorMap(proj_norm, cv2.COLORMAP_BONE)
                        proj_rgb = cv2.cvtColor(proj_colored, cv2.COLOR_BGR2RGB)
                        v_cols[idx].image(proj_rgb, caption=view_names[idx], use_container_width=True)

                exec_time = round((time.time() - start_time) * 1000)
                st.success(f"Inference completed in {exec_time} ms")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")