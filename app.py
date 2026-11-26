import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import math
import base64
import os
import cv2
import numpy as np
import tempfile
import shutil
import zipfile
import nibabel as nib
import time
import pydicom

from model import DentalMultiTaskNet
from core.model import DecoupledForensicHead

st.set_page_config(
    page_title="CranioDental AI | Forensic Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --bg-base: #0b0f19;
        --bg-panel: #151b2b;
        --bg-viewport: #000000;
        --accent: #0ea5e9;
        --text-main: #f8fafc;
        --text-muted: #64748b;
        --border: #1e293b;
        --success: #10b981;
        --skull-accent: #8b5cf6;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }

    header[data-testid="stHeader"] {
        background-color: #0b0f19;
    }

    section[data-testid="stSidebar"] {
        background-color: #151b2b;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 100%;
    }

    .brand-logo {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #ffffff;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .brand-logo span {
        color: #0ea5e9;
    }

    .status-box {
        font-size: 12px;
        padding: 15px;
        background: #000000;
        border-radius: 6px;
        border: 1px solid #064e3b;
        color: #34d399;
        line-height: 1.65;
        margin-top: 40vh;
    }

    .ws-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid #1e293b;
    }

    .ws-header h1 {
        margin: 0;
        font-size: 23px;
        font-weight: 400;
        color: #e2e8f0;
    }

    .panel-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 10px;
        font-weight: 700;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #151b2b;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        padding: 16px;
    }

    div[data-testid="stRadio"] > div {
        gap: 12px;
    }

    div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.03);
        padding: 8px 14px;
        border-radius: 6px;
        border: 1px solid #1e293b;
        color: #f8fafc !important;
        font-weight: 600;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #000000;
        border: 1px dashed #334155;
        border-radius: 6px;
        padding: 8px;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        width: 100%;
        padding: 14px;
        margin-top: 10px;
        background: linear-gradient(135deg, #0284c7, #2563eb);
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1, #1d4ed8);
        color: #ffffff;
    }

    .vp-label {
        font-size: 11px;
        color: #cbd5e1;
        font-family: monospace;
        margin-bottom: 4px;
        display: block;
    }

    .mounted-badge {
        background: #000000;
        border: 1px solid #064e3b;
        color: #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

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

def load_dicom_series(dcm_paths):
    slices = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p)
            if hasattr(ds, 'pixel_array'):
                slices.append((ds, p))
        except Exception:
            continue

    if not slices:
        raise ValueError("No valid DICOM (.dcm) slices found.")

    if len(slices) == 1:
        arr = slices[0][0].pixel_array.astype(np.float32)
        if arr.ndim == 2:
            return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    def sort_key(item):
        ds, path = item
        if hasattr(ds, 'ImagePositionPatient'):
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        if hasattr(ds, 'InstanceNumber'):
            return float(ds.InstanceNumber)
        return path

    slices.sort(key=sort_key)

    arrays = []
    target_shape = slices[0][0].pixel_array.shape
    for ds, _ in slices:
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim > 2:
            arr = arr[0]
        if arr.shape != target_shape[:2]:
            arr = cv2.resize(arr, (target_shape[1], target_shape[0]))
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        arr = arr * slope + intercept
        arrays.append(arr)

    vol_np = np.stack(arrays, axis=0)
    return torch.tensor(vol_np, dtype=torch.float32).unsqueeze(0)

def process_skull_inputs(uploaded_files):
    work_dir = tempfile.mkdtemp()
    try:
        saved_paths = []
        for uf in uploaded_files:
            out_path = os.path.join(work_dir, os.path.basename(uf.name))
            with open(out_path, "wb") as f:
                f.write(uf.getvalue())
            saved_paths.append(out_path)

        if len(saved_paths) == 1 and saved_paths[0].lower().endswith('.zip'):
            extract_dir = os.path.join(work_dir, "extracted")
            with zipfile.ZipFile(saved_paths[0], 'r') as zf:
                zf.extractall(extract_dir)
            extracted_files = []
            for root, _, files in os.walk(extract_dir):
                for fname in files:
                    if not fname.startswith('.'):
                        extracted_files.append(os.path.join(root, fname))
            saved_paths = extracted_files

        dcm_paths = [p for p in saved_paths if p.lower().endswith('.dcm')]
        nii_paths = [p for p in saved_paths if p.lower().endswith(('.nii', '.nii.gz'))]
        pt_paths = [p for p in saved_paths if p.lower().endswith('.pt')]

        if dcm_paths:
            scan_tensor = load_dicom_series(dcm_paths)
        elif nii_paths:
            nii_img = nib.load(nii_paths[0])
            scan_tensor = torch.tensor(nii_img.get_fdata(), dtype=torch.float32)
            if scan_tensor.dim() == 3:
                scan_tensor = scan_tensor.unsqueeze(0)
        elif pt_paths:
            scan_tensor = torch.load(pt_paths[0], map_location='cpu', weights_only=True).to(torch.float32)
            if scan_tensor.dim() == 3:
                scan_tensor = scan_tensor.unsqueeze(0)
        else:
            raise ValueError("Unsupported format. Upload .zip DICOM folder, .dcm files, .nii/.nii.gz, or .pt.")

        if scan_tensor.dim() == 3:
            scan_tensor = scan_tensor.unsqueeze(0)

        if scan_tensor.shape[1:] != (128, 128, 128):
            scan_tensor = scan_tensor.unsqueeze(0)
            scan_tensor = F.interpolate(scan_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
            scan_tensor = scan_tensor.squeeze(0)

        bone_mask = (scan_tensor >= 300) & (scan_tensor <= 400)
        if bone_mask.sum() > 0:
            scan_tensor = scan_tensor * bone_mask.float()

        vol = scan_tensor.squeeze(0)
        axial_proj = torch.max(vol, dim=0)[0]
        coronal_proj = torch.max(vol, dim=1)[0]
        sagittal_proj = torch.max(vol, dim=2)[0]

        multi_view_tensor = torch.stack([axial_proj, coronal_proj, sagittal_proj], dim=0)
        v_min, v_max = multi_view_tensor.min(), multi_view_tensor.max()
        multi_view_tensor = (multi_view_tensor - v_min) / (v_max - v_min + 1e-8)

        return multi_view_tensor.unsqueeze(0), [axial_proj, coronal_proj, sagittal_proj]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

def build_report_html(results):
    if not results:
        return """
        <!DOCTYPE html>
        <html><head><style>
            body { margin:0; padding:0; background:#151b2b; color:#f8fafc; font-family:'Segoe UI',system-ui,sans-serif; }
            .panel { padding:20px; border:1px solid #1e293b; border-radius:8px; background:#151b2b; min-height:420px; box-sizing:border-box; }
            .title { font-size:14px; text-transform:uppercase; letter-spacing:1px; color:#64748b; font-weight:bold; display:flex; justify-content:space-between; }
            .empty { height:340px; display:flex; align-items:center; justify-content:center; color:#475569; text-align:center; font-size:14px; }
        </style></head>
        <body>
            <div class="panel">
                <div class="title"><span>Clinical Output Report</span><span style="font-weight:normal;font-size:11px;">IPCTI Diagnostic Engine</span></div>
                <div class="empty">Awaiting scans for independent inference.</div>
            </div>
        </body></html>
        """

    cards_html = ""
    if results.get("has_dental"):
        d_color = "#3b82f6" if results["dental_sex"] == "Male" else "#ec4899"
        cards_html += f"""
        <div class="metric-card" style="border-top: 3px solid #0ea5e9;">
            <div class="metric-header"><span class="metric-title">Dental Evaluator</span></div>
            <div class="metric-value">{results['dental_age']} Yrs</div>
            <div class="sub-val">{results['dental_sex']} ({results['dental_conf']}%)</div>
            <div class="gauge-track"><div class="gauge-fill" style="width:{results['dental_conf']}%; background:{d_color};"></div></div>
            <div class="static-grid">
                <div>ACCURACY: <span>89.0%</span></div>
                <div>MAE: <span>&plusmn;2.6 Yrs</span></div>
                <div>MSE: <span>10.24</span></div>
                <div>F1-SCORE: <span>83.6%</span></div>
            </div>
        </div>
        """

    if results.get("has_skull"):
        s_color = "#3b82f6" if results["skull_sex"] == "Male" else "#ec4899"
        cards_html += f"""
        <div class="metric-card" style="border-top: 3px solid #8b5cf6;">
            <div class="metric-header"><span class="metric-title">Cranial Evaluator</span></div>
            <div class="metric-value">{results['skull_age']} Yrs</div>
            <div class="sub-val">{results['skull_sex']} ({results['skull_conf']}%)</div>
            <div class="gauge-track"><div class="gauge-fill" style="width:{results['skull_conf']}%; background:{s_color};"></div></div>
            <div class="static-grid">
                <div>ACCURACY: <span>100%</span></div>
                <div>MAE: <span>&plusmn;4.18 Yrs</span></div>
                <div>MSE: <span>25.50</span></div>
                <div>F1-SCORE: <span>100%</span></div>
            </div>
        </div>
        """

    visuals_html = ""
    if results.get("has_dental") and results.get("dental_heatmaps"):
        tiles = ""
        for k, heat_b64 in results["dental_heatmaps"].items():
            orig_b64 = results["dental_originals"][k]
            tiles += f"""
            <div class="img-tile" style="border-color:#0ea5e9;">
                <img src="{orig_b64}" class="layer-img">
                <img src="{heat_b64}" class="layer-img heatmap-overlay">
                <span class="tile-tag">{k.upper()}</span>
            </div>
            """
        visuals_html += f"""
        <div class="section-label">Saliency &amp; Features</div>
        <div class="tiles-grid">{tiles}</div>
        <div class="slider-container" id="slider-box">
            <div class="slider-labels"><span>Scan Only</span><span>Thermal Overlay</span></div>
            <input type="range" min="0" max="100" value="100" oninput="document.querySelectorAll('.heatmap-overlay').forEach(el => el.style.opacity = this.value / 100);">
        </div>
        """

    if results.get("has_skull") and results.get("skull_views"):
        skull_tiles = ""
        for v_name, v_b64 in results["skull_views"].items():
            skull_tiles += f"""
            <div class="img-tile" style="border-color:#8b5cf6;">
                <img src="{v_b64}" class="layer-img">
                <span class="tile-tag">{v_name.upper()}</span>
            </div>
            """
        visuals_html += f"""
        <div class="section-label" style="color:#8b5cf6;">Cranial 3-View MIP Projections</div>
        <div class="tiles-grid">{skull_tiles}</div>
        """

    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            body { margin:0; padding:0; background:#151b2b; color:#f8fafc; font-family:'Segoe UI',system-ui,sans-serif; }
            .panel { padding:20px; border:1px solid #1e293b; border-radius:8px; background:#151b2b; box-sizing:border-box; }
            .top-bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:15px; }
            .title { font-size:14px; text-transform:uppercase; letter-spacing:1px; color:#64748b; font-weight:bold; }
            .btn-export { background:#1e293b; color:#fff; border:1px solid #334155; padding:7px 14px; border-radius:4px; cursor:pointer; font-size:11px; font-weight:bold; }
            .btn-export:hover { background:#334155; }
            .cards-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:14px; margin-bottom:16px; }
            .metric-card { background:#0f131f; border:1px solid #1e293b; padding:16px; border-radius:6px; }
            .metric-header { margin-bottom:8px; }
            .metric-title { font-size:12px; color:#64748b; text-transform:uppercase; font-weight:bold; }
            .metric-value { font-size:26px; font-weight:300; color:#fff; }
            .sub-val { font-size:13px; color:#94a3b8; margin-top:6px; }
            .gauge-track { width:100%; height:6px; background:#1e293b; border-radius:3px; margin-top:8px; overflow:hidden; }
            .gauge-fill { height:100%; border-radius:3px; }
            .static-grid { font-size:10.5px; color:#64748b; margin-top:14px; border-top:1px solid #1e293b; padding-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:6px; }
            .static-grid span { color:#fff; font-weight:bold; }
            .section-label { margin-top:16px; font-size:12px; color:#64748b; text-transform:uppercase; font-weight:bold; }
            .tiles-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(95px, 1fr)); gap:10px; margin-top:10px; }
            .img-tile { position:relative; aspect-ratio:1; border-radius:4px; border:1px solid #0ea5e9; overflow:hidden; background:#000; }
            .layer-img { position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; }
            .heatmap-overlay { transition:opacity 0.15s ease; }
            .tile-tag { position:absolute; bottom:4px; left:4px; font-size:9px; font-family:monospace; background:rgba(0,0,0,0.7); color:#e2e8f0; padding:2px 5px; border-radius:3px; }
            .slider-container { margin-top:14px; background:#0f131f; padding:10px; border-radius:6px; border:1px solid #1e293b; }
            .slider-labels { font-size:11px; color:#64748b; margin-bottom:5px; display:flex; justify-content:space-between; }
            input[type=range] { width:100%; cursor:pointer; accent-color:#0ea5e9; }
        </style>
    </head>
    <body>
        <div class="panel" id="report-content">
            <div class="top-bar">
                <div>
                    <div class="title">Clinical Output Report</div>
                    <div style="font-size:11px; color:#64748b; margin-top:2px;">IPCTI Diagnostic Engine</div>
                </div>
                <button class="btn-export" id="pdf-btn" onclick="exportReport()">📥 EXPORT CLINICAL PDF</button>
            </div>
            <div class="cards-grid">__CARDS__</div>
            __VISUALS__
        </div>
        <script>
            function exportReport() {
                var slider = document.getElementById('slider-box');
                var btn = document.getElementById('pdf-btn');
                if (slider) slider.style.display = 'none';
                if (btn) btn.style.display = 'none';
                var element = document.getElementById('report-content');
                var opt = {
                    margin: 10,
                    filename: 'CranioDent_Forensic_Report.pdf',
                    image: { type: 'jpeg', quality: 0.98 },
                    html2canvas: { scale: 2, useCORS: true, backgroundColor: '#151b2b' },
                    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
                };
                html2pdf().set(opt).from(element).save().then(function() {
                    if (slider) slider.style.display = 'block';
                    if (btn) btn.style.display = 'inline-block';
                });
            }
        </script>
    </body>
    </html>
    """
    return template.replace("__CARDS__", cards_html).replace("__VISUALS__", visuals_html)


if "results" not in st.session_state:
    st.session_state.results = None

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown('<div class="brand-logo">💀🦷 CRANIO<span>DENT</span></div>', unsafe_allow_html=True)
    exec_block = ""
    if st.session_state.results:
        exec_block = f"<br><br><strong style='color:#0ea5e9;'>LATEST INFERENCE:</strong><br>{st.session_state.results['exec_time']} ms"
    st.markdown(f"""
    <div class="status-box">
        <strong>SYSTEM STATUS:</strong> ONLINE<br>
        <strong>MODELS ACTIVE:</strong><br>
        - Teeth: EffNet-B3<br>
        - Skull: EffNetV2-S{exec_block}
    </div>
    """, unsafe_allow_html=True)

# ================= MAIN WORKSPACE =================
st.markdown("""
<div class="ws-header">
    <h1>Craniofacial &amp; Dental Diagnostic Workstation</h1>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1.35, 1], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Select Diagnostic Mode</div>', unsafe_allow_html=True)
        diag_mode = st.radio(
            "Select Diagnostic Mode",
            ["🦷 Dental Only", "💀 Skull Only", "⚕️ Multimodal (Both)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown("<hr style='border-color:#1e293b; margin:14px 0 18px 0;'>", unsafe_allow_html=True)

        coronal_11 = coronal_21 = sagittal_11 = sagittal_21 = None
        skull_files = None

        show_dental = diag_mode in ("🦷 Dental Only", "⚕️ Multimodal (Both)")
        show_skull = diag_mode in ("💀 Skull Only", "⚕️ Multimodal (Both)")

        if show_dental and show_skull:
            d_box, s_box = st.columns(2, gap="medium")
        elif show_dental:
            d_box = st.container()
            s_box = None
        else:
            d_box = None
            s_box = st.container()

        if d_box is not None:
            with d_box:
                st.markdown('<div class="panel-title" style="color:#0ea5e9;">Teeth (4-View CBCT)</div>', unsafe_allow_html=True)
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    st.markdown('<span class="vp-label">CORONAL_11</span>', unsafe_allow_html=True)
                    coronal_11 = st.file_uploader("CORONAL_11", type=["png", "jpg", "jpeg"], key="c11", label_visibility="collapsed")
                    if coronal_11:
                        st.image(coronal_11, use_container_width=True)
                with r1c2:
                    st.markdown('<span class="vp-label">CORONAL_21</span>', unsafe_allow_html=True)
                    coronal_21 = st.file_uploader("CORONAL_21", type=["png", "jpg", "jpeg"], key="c21", label_visibility="collapsed")
                    if coronal_21:
                        st.image(coronal_21, use_container_width=True)

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    st.markdown('<span class="vp-label">SAGITTAL_11</span>', unsafe_allow_html=True)
                    sagittal_11 = st.file_uploader("SAGITTAL_11", type=["png", "jpg", "jpeg"], key="s11", label_visibility="collapsed")
                    if sagittal_11:
                        st.image(sagittal_11, use_container_width=True)
                with r2c2:
                    st.markdown('<span class="vp-label">SAGITTAL_21</span>', unsafe_allow_html=True)
                    sagittal_21 = st.file_uploader("SAGITTAL_21", type=["png", "jpg", "jpeg"], key="s21", label_visibility="collapsed")
                    if sagittal_21:
                        st.image(sagittal_21, use_container_width=True)

        if s_box is not None:
            with s_box:
                st.markdown('<div class="panel-title" style="color:#8b5cf6;">Skull (3D Volumetric)</div>', unsafe_allow_html=True)
                st.markdown('<span class="vp-label">UPLOAD .ZIP FOLDER / MULTIPLE .DCM (Ctrl+A) / .NII.GZ / .PT</span>', unsafe_allow_html=True)
                skull_files = st.file_uploader(
                    "Skull Volume",
                    type=["zip", "dcm", "nii", "gz", "pt"],
                    accept_multiple_files=True,
                    key="skull",
                    label_visibility="collapsed"
                )
                if skull_files:
                    st.markdown(f'<div class="mounted-badge">🩻 Volume Mounted ({len(skull_files)} file(s))</div>', unsafe_allow_html=True)

        run_btn = st.button("Execute Multimodal Analysis", type="primary", use_container_width=True)

        if run_btn:
            dental_files = {
                'coronal_11': coronal_11 if show_dental else None,
                'coronal_21': coronal_21 if show_dental else None,
                'sagittal_11': sagittal_11 if show_dental else None,
                'sagittal_21': sagittal_21 if show_dental else None,
            }
            active_skull = skull_files if (show_skull and skull_files) else None
            has_dental = any(f is not None for f in dental_files.values())
            has_skull = bool(active_skull)

            if not has_dental and not has_skull:
                st.error("Please upload Dental scans, a Skull scan, or both.")
            else:
                try:
                    with st.spinner("Executing independent forensic inference..."):
                        start_time = time.time()
                        dental_model, skull_model = load_models()
                        results = {'has_dental': False, 'has_skull': False}

                        if has_dental:
                            processed_images = []
                            display_images = {}
                            ordered_keys = []

                            for key, file in dental_files.items():
                                if file is not None:
                                    img_bytes = file.getvalue()
                                    display_images[key] = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
                                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                                    processed_images.append(dental_transform(img))
                                    ordered_keys.append((key, img_bytes))

                            dental_batch = torch.stack(processed_images).unsqueeze(0).to(device)
                            dental_batch.requires_grad = True

                            outputs = dental_model(dental_batch)
                            d_age = outputs['age'].item()
                            d_sex_logit = outputs['gender'].item()
                            d_prob = 1.0 / (1.0 + math.exp(-d_sex_logit))

                            results['dental_age'] = round(d_age, 1)
                            results['dental_sex'] = "Male" if d_prob > 0.5 else "Female"
                            results['dental_conf'] = round((d_prob if d_prob > 0.5 else (1 - d_prob)) * 100, 1)
                            results['has_dental'] = True
                            results['dental_originals'] = display_images

                            heatmap_display = {}
                            dental_model.zero_grad()
                            outputs['age'].backward()
                            raw_saliency = dental_batch.grad.abs().squeeze(0).cpu().numpy()
                            raw_saliency = np.max(raw_saliency, axis=1)

                            for idx, (key, img_bytes) in enumerate(ordered_keys):
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
                                _, buffer = cv2.imencode('.jpg', blended)
                                heatmap_display[key] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                            results['dental_heatmaps'] = heatmap_display

                        if has_skull:
                            skull_tensor, projections = process_skull_inputs(active_skull)
                            skull_tensor = skull_tensor.to(device)

                            with torch.no_grad():
                                s_age_norm, s_sex_logits = skull_model(skull_tensor)
                                s_age = s_age_norm.item() * 100.0
                                s_prob = torch.sigmoid(s_sex_logits).item()

                            results['skull_age'] = round(s_age, 1)
                            results['skull_sex'] = "Male" if s_prob >= 0.5 else "Female"
                            results['skull_conf'] = round((s_prob if s_prob >= 0.5 else (1 - s_prob)) * 100, 1)
                            results['has_skull'] = True

                            skull_views = {}
                            view_names = ['Axial', 'Coronal', 'Sagittal']
                            for idx, proj in enumerate(projections):
                                proj_img = proj.cpu().numpy()
                                proj_norm = cv2.normalize(proj_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                proj_colored = cv2.applyColorMap(proj_norm, cv2.COLORMAP_BONE)
                                _, buffer = cv2.imencode('.jpg', proj_colored)
                                skull_views[view_names[idx]] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                            results['skull_views'] = skull_views

                        results['exec_time'] = round((time.time() - start_time) * 1000)
                        st.session_state.results = results
                        st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with right_col:
    res = st.session_state.results
    report_height = 460
    if res:
        report_height = 560
        if res.get("has_dental") and res.get("has_skull"):
            report_height = 840
        elif res.get("has_dental"):
            report_height = 680
        elif res.get("has_skull"):
            report_height = 600
    components.html(build_report_html(res), height=report_height, scrolling=True)
