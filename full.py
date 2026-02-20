import streamlit as st
import plotly.graph_objects as go
import nibabel as nib
import numpy as np
import tempfile
import os

# --- APP CONFIGURATION & CSS ---
st.set_page_config(page_title="Neuro Viewer", layout="wide")

# Force crisp, nearest-neighbor rendering for all images (prevents blurry NIfTI slices)
st.markdown(
    """
    <style>
    [data-testid="stImage"] img {
        image-rendering: pixelated;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Neuroimaging Viewer (NIfTI & GIFTI)")
st.write("Upload a NIfTI volume or a GIFTI mesh to explore your data.")

# --- CACHED NIFTI LOADER ---
@st.cache_data
def load_nifti_data(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
    finally:
        os.remove(tmp_path)
        
    return data

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🩻 NIfTI Viewer (2D Slices)", "🌐 GIFTI Viewer (3D Mesh)"])

# ==========================================
# TAB 1: NIFTI VIEWER
# ==========================================
with tab1:
    st.header("NIfTI Volume Viewer")
    nifti_file = st.file_uploader("Choose a NIfTI file (.nii or .nii.gz)", type=["nii", "gz"], key="nifti")
    
    if nifti_file is not None:
        with st.spinner("Loading NIfTI data into memory..."):
            data = load_nifti_data(nifti_file.getvalue())
            
            st.divider()
            
            # Handle 4D data
            if data.ndim >= 4:
                t_idx = st.slider("Volume (Time / 4th Dimension)", 0, data.shape[3] - 1, 0)
                vol_data = data[:, :, :, t_idx]
            else:
                vol_data = data
                
            # Intensity / Contrast Control
            min_val, max_val = float(np.min(vol_data)), float(np.max(vol_data))
            if min_val == max_val: 
                max_val = min_val + 1.0
                
            vmin, vmax = st.slider(
                "Intensity Range (Contrast)", 
                min_value=min_val, 
                max_value=max_val, 
                value=(min_val, max_val)
            )
            
            # Slice Position Controls
            st.markdown("### View Slices")
            c1, c2, c3 = st.columns(3)
            x_idx = c1.slider("Sagittal (X)", 0, vol_data.shape[0] - 1, vol_data.shape[0] // 2)
            y_idx = c2.slider("Coronal (Y)", 0, vol_data.shape[1] - 1, vol_data.shape[1] // 2)
            z_idx = c3.slider("Axial (Z)", 0, vol_data.shape[2] - 1, vol_data.shape[2] // 2)
            
            def normalize_slice(slice_2d, v_min, v_max):
                clipped = np.clip(slice_2d, v_min, v_max)
                if v_max == v_min:
                    return np.zeros_like(clipped, dtype=np.uint8)
                normalized = (clipped - v_min) / (v_max - v_min)
                return (normalized * 255).astype(np.uint8)
                
            # Extract and orient slices
            sagittal_slice = normalize_slice(np.rot90(vol_data[x_idx, :, :]), vmin, vmax)
            coronal_slice = normalize_slice(np.rot90(vol_data[:, y_idx, :]), vmin, vmax)
            axial_slice = normalize_slice(np.rot90(vol_data[:, :, z_idx]), vmin, vmax)
            
            # Render images (CSS will keep them crisp)
            c1.image(sagittal_slice, caption=f"Sagittal (X={x_idx})", use_container_width=True)
            c2.image(coronal_slice, caption=f"Coronal (Y={y_idx})", use_container_width=True)
            c3.image(axial_slice, caption=f"Axial (Z={z_idx})", use_container_width=True)


# ==========================================
# TAB 2: GIFTI VIEWER
# ==========================================
with tab2:
    st.header("3D GIFTI Mesh Viewer")
    
    # Visual settings
    with st.expander("⚙️ Visualization Settings", expanded=True):
        colarmaps = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'coolwarm', 'RdBu', 'icefire']
        colormap = st.selectbox("Colormap", colarmaps, index=0)

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        mesh_file = st.file_uploader("1. Choose a Mesh .gii file", type=["gii"], key="mesh")
    with col2:
        texture_file = st.file_uploader("2. Choose a Texture .gii file (Optional)", type=["gii"], key="texture")

    if mesh_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gii") as tmp_mesh:
            tmp_mesh.write(mesh_file.getvalue())
            mesh_path = tmp_mesh.name
            
        texture_path = None
        intensity_values = None
        
        if texture_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gii") as tmp_tex:
                tmp_tex.write(texture_file.getvalue())
                texture_path = tmp_tex.name

        try:
            with st.spinner("Processing 3D mesh data..."):
                gii_mesh = nib.load(mesh_path)
                vertices = gii_mesh.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
                faces = gii_mesh.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
                
                x, y, z = vertices.T
                i, j, k = faces.T

                if texture_path:
                    gii_tex = nib.load(texture_path)
                    intensity_values = gii_tex.darrays[0].data.astype(float)
                    
                    if len(intensity_values) != len(x):
                        st.error(f"Mismatch: Mesh has {len(x)} vertices, but texture has {len(intensity_values)} values.")
                        intensity_values = None

                lighting_effects_without_texture = dict(ambient=0.15, diffuse=1.0, specular=0.5, roughness=0.4, fresnel=0.2)
                lighting_effects_with_texture = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
                
                # Switch lighting based on whether intensity values are successfully loaded
                current_lighting = lighting_effects_with_texture if intensity_values is not None else lighting_effects_without_texture
                    
                lightposition = dict(x=100, y=100, z=100)
                camera = dict(eye=dict(x=1.5, y=1.5, z=0.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))
                
                mesh_args = dict(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    opacity=1.0,
                    flatshading=False,
                    lighting=current_lighting,
                    lightposition=lightposition,
                    name="Cortical Surface"
                )

                if intensity_values is not None:
                    mesh_args['intensity'] = intensity_values
                    mesh_args['colorscale'] = colormap
                    mesh_args['showscale'] = True
                    mesh_args['colorbar'] = dict(title="Intensity", thickness=20, len=0.75)
                    mesh_args['text'] = intensity_values
                    mesh_args['hovertemplate'] = (
                        "<b>Value:</b> %{text:.4f}<extra></extra>"
                    )
                else:
                    mesh_args['color'] = 'silver'
                    mesh_args['hovertemplate'] = (
                        "<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<br><b>Z:</b> %{z:.2f}<extra></extra>"
                    )

                fig = go.Figure(data=[go.Mesh3d(**mesh_args)])
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        zaxis=dict(visible=False), camera=camera, aspectmode='data'          
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=700
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
        
        finally:
            if os.path.exists(mesh_path):
                os.remove(mesh_path)
            if texture_path and os.path.exists(texture_path):
                os.remove(texture_path)