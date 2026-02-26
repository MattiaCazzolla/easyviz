import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nibabel as nib
import numpy as np
import tempfile
import os
import math

# --- APP CONFIGURATION & CSS ---
st.set_page_config(
    page_title="Neuro Viewer Pro", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished, enterprise look
st.markdown(
    """
    <style>
    /* Force crisp rendering for NIfTI slices and add modern styling */
    [data-testid="stImage"] img {
        image-rendering: pixelated;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    /* Clean up main container padding */
    .block-container {
        padding-top: 2rem;
    }
    /* Subtle metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- CACHED DATA LOADERS (FOR PERFORMANCE) ---
@st.cache_data(show_spinner=False)
def load_nifti_data(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
        zooms = img.header.get_zooms()
    finally:
        os.remove(tmp_path)
        
    return data, zooms

@st.cache_data(show_spinner=False)
def load_gifti_mesh(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gii") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        gii_mesh = nib.load(tmp_path)
        vertices = gii_mesh.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
        faces = gii_mesh.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
        x, y, z = vertices.T
        i, j, k = faces.T
    finally:
        os.remove(tmp_path)
        
    return x, y, z, i, j, k

@st.cache_data(show_spinner=False)
def load_gifti_texture(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gii") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        gii_tex = nib.load(tmp_path)
        intensity = gii_tex.darrays[0].data.astype(float)
    finally:
        os.remove(tmp_path)
        
    return intensity

# --- HELPER FUNCTIONS ---
def normalize_slice(slice_2d, v_min, v_max):
    clipped = np.clip(slice_2d, v_min, v_max)
    if v_max == v_min:
        return np.zeros_like(clipped, dtype=np.uint8)
    normalized = (clipped - v_min) / (v_max - v_min)
    return (normalized * 255).astype(np.uint8)


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🧠 Neuro Viewer")
st.sidebar.markdown("Explore NIfTI volumes and GIFTI meshes.")
st.sidebar.divider()

view_mode = st.sidebar.radio(
    "Select Modality",
    options=["🩻 NIfTI (2D Volume)", "🌐 GIFTI (3D Mesh)"],
    label_visibility="collapsed"
)

st.sidebar.divider()

# ==========================================
# VIEW 1: NIFTI VIEWER
# ==========================================
if "NIfTI" in view_mode:
    st.title("🩻 NIfTI Volume Viewer")
    
    nifti_file = st.sidebar.file_uploader("Upload NIfTI file (.nii or .nii.gz)", type=["nii", "gz"])
    
    if nifti_file is not None:
        with st.spinner("Loading NIfTI data..."):
            data, zooms = load_nifti_data(nifti_file.getvalue())
            
            st.sidebar.subheader("Volume Controls")
            
            t_idx = 0
            if data.ndim >= 4:
                t_idx = st.sidebar.slider("Time / 4th Dimension", 0, data.shape[3] - 1, 0)
                vol_data = data[:, :, :, t_idx]
            else:
                vol_data = data
                
            min_val, max_val = float(np.min(vol_data)), float(np.max(vol_data))
            if min_val == max_val: 
                max_val = min_val + 1.0
                
            vmin, vmax = st.sidebar.slider(
                "Intensity Range", 
                min_value=min_val, 
                max_value=max_val, 
                value=(min_val, max_val)
            )

            st.sidebar.subheader("Coordinates")
            x_idx = st.sidebar.slider("Sagittal (X)", 0, vol_data.shape[0] - 1, vol_data.shape[0] // 2)
            y_idx = st.sidebar.slider("Coronal (Y)", 0, vol_data.shape[1] - 1, vol_data.shape[1] // 2)
            z_idx = st.sidebar.slider("Axial (Z)", 0, vol_data.shape[2] - 1, vol_data.shape[2] // 2)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dimensions", f"{data.ndim}D")
            m2.metric("Shape", f"{data.shape}")
            voxel_size = " × ".join([f"{z:.2f}" for z in zooms[:3]])
            m3.metric("Voxel Size (mm)", voxel_size)
            m4.metric("Intensity Range", f"{min_val:.1f} to {max_val:.1f}")
            
            st.divider()
            
            c1, c2, c3 = st.columns(3)
            
            sagittal_slice = normalize_slice(np.rot90(vol_data[x_idx, :, :]), vmin, vmax)
            coronal_slice = normalize_slice(np.rot90(vol_data[:, y_idx, :]), vmin, vmax)
            axial_slice = normalize_slice(np.rot90(vol_data[:, :, z_idx]), vmin, vmax)
            
            c1.markdown(f"**Sagittal** (X={x_idx})")
            c1.image(sagittal_slice, use_container_width=True)
            
            c2.markdown(f"**Coronal** (Y={y_idx})")
            c2.image(coronal_slice, use_container_width=True)
            
            c3.markdown(f"**Axial** (Z={z_idx})")
            c3.image(axial_slice, use_container_width=True)
            
    else:
        st.info("👈 Please upload a NIfTI file in the sidebar to begin.")

# ==========================================
# VIEW 2: GIFTI VIEWER
# ==========================================
elif "GIFTI" in view_mode:
    st.title("🌐 GIFTI Mesh Viewer")
    
    mesh_file = st.sidebar.file_uploader("1. Upload Mesh (.gii)", type=["gii"])
    texture_files = st.sidebar.file_uploader(
        "2. Upload Textures (.gii) (Optional)", 
        type=["gii"], 
        accept_multiple_files=True
    )
    
    if mesh_file is not None:
        active_texture_file = None
        intensity_values = None
        raw_intensity = None
        
        if texture_files:
            st.sidebar.subheader("Texture Controls")
            texture_dict = {f.name: f for f in texture_files}
            selected_texture_name = st.sidebar.selectbox("Active Texture:", list(texture_dict.keys()))
            active_texture_file = texture_dict[selected_texture_name]
            
        st.sidebar.subheader("Visualization Settings")
        colarmaps = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'coolwarm', 'RdBu', 'icefire']
        colormap = st.sidebar.selectbox("Colormap", colarmaps, index=0)

        try:
            with st.spinner("Rendering 3D mesh..."):
                
                # Fetch data from cache
                x, y, z, i, j, k = load_gifti_mesh(mesh_file.getvalue())

                st.markdown("### Mesh Properties")
                m1, m2 = st.columns(2)
                m1.metric("Vertices", f"{len(x):,}")
                m2.metric("Faces", f"{len(i):,}")
                st.divider()

                # Calculate intensity and apply masking threshold
                if active_texture_file:
                    raw_intensity = load_gifti_texture(active_texture_file.getvalue())
                    
                    if len(raw_intensity) != len(x):
                        st.error(f"Mismatch: Mesh has {len(x)} vertices, but selected texture has {len(raw_intensity)} values.")
                        raw_intensity = None
                    else:
                        # Masking Slider Configuration
                        min_tex, max_tex = float(np.min(raw_intensity)), float(np.max(raw_intensity))
                        if min_tex == max_tex: max_tex += 1.0
                        
                        thresh_min, thresh_max = st.sidebar.slider(
                            "Intensity Masking (Threshold)", 
                            min_value=min_tex, 
                            max_value=max_tex, 
                            value=(min_tex, max_tex),
                            step=(max_tex - min_tex) / 100.0
                        )
                        
                        # Apply mask: Set values outside threshold to NaN so they aren't colored
                        intensity_values = np.where(
                            (raw_intensity >= thresh_min) & (raw_intensity <= thresh_max), 
                            raw_intensity, 
                            np.nan
                        )

                # Dictionary of standard camera views
                camera_views = {
                    "Isometric": dict(eye=dict(x=1.5, y=1.5, z=0.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                    "Lateral (Right)": dict(eye=dict(x=1.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                    "Lateral (Left)": dict(eye=dict(x=-1.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                    "Superior": dict(eye=dict(x=0, y=0, z=1.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                    "Inferior": dict(eye=dict(x=0, y=0, z=-1.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                    "Anterior": dict(eye=dict(x=0, y=1.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                    "Posterior": dict(eye=dict(x=0, y=-1.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))
                }

                # Construct Plotly built-in buttons (Client-side fast rendering)
                plotly_buttons = []
                for view_name, cam in camera_views.items():
                    plotly_buttons.append(
                        dict(
                            label=view_name,
                            method="relayout",
                            args=[{"scene.camera": cam}]
                        )
                    )

                # Lighting Setup
                lighting_effects_without_texture = dict(ambient=0.15, diffuse=1.0, specular=0.5, roughness=0.4, fresnel=0.2)
                lighting_effects_with_texture = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
                current_lighting = lighting_effects_with_texture if intensity_values is not None else lighting_effects_without_texture
                lightposition = dict(x=100, y=100, z=100)
                
                # Build Base Mesh Arguments
                mesh_args = dict(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    opacity=1.0,
                    flatshading=False,
                    lighting=current_lighting,
                    lightposition=lightposition,
                    name="Cortical Surface",
                    color='#cbd5e1' # Professional silver base color
                )

                if intensity_values is not None:
                    mesh_args['intensity'] = intensity_values
                    mesh_args['colorscale'] = colormap
                    mesh_args['showscale'] = True
                    mesh_args['colorbar'] = dict(title="Intensity", thickness=15, len=0.6, xpad=30)
                    mesh_args['text'] = raw_intensity
                    mesh_args['hovertemplate'] = "<b>Value:</b> %{text:.4f}<extra></extra>"
                else:
                    mesh_args['hovertemplate'] = "<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<br><b>Z:</b> %{z:.2f}<extra></extra>"

                # Render Main Interactive Plot
                st.markdown("**Interactive View:**")
                fig = go.Figure(data=[go.Mesh3d(**mesh_args)])
                
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        zaxis=dict(visible=False), camera=camera_views["Isometric"], aspectmode='data'          
                    ),
                    margin=dict(l=0, r=0, b=0, t=50),  # Increased top margin to fit buttons
                    height=700,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    updatemenus=[dict(
                        type="buttons",
                        direction="right",
                        x=0.5,
                        y=1.05,
                        xanchor="center",
                        yanchor="bottom",
                        showactive=True,
                        buttons=plotly_buttons
                    )]
                )

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # --- PUBLICATION FIGURE & HTML GENERATOR ---
                st.divider()
                st.markdown("### 💾 Export Options")
                
                # 1. HTML Interactive Export
                st.markdown("**Interactive 3D HTML**")
                st.caption("Download the 3D isometric model as an interactive webpage to share with colleagues.")
                
                # Generate HTML string from the main figure
                # include_plotlyjs="cdn" keeps the file size small by loading the JS library from the web
                html_string = fig.to_html(include_plotlyjs="cdn", full_html=True)
                
                st.download_button(
                    label="📥 Download Interactive HTML",
                    data=html_string,
                    file_name="neuro_mesh_isometric.html",
                    mime="text/html"
                )
                
                st.divider()
                
                # 2. Static Multi-view Export (Your existing code)
                st.markdown("**📸 Static Multi-view Figure**")
                # Allow user to select which views to include in the publication figure
                default_views = ["Lateral (Right)", "Superior", "Anterior", "Lateral (Left)", "Inferior", "Posterior"]
                selected_views = st.multiselect(
                    "Select views to include in the figure:",
                    options=list(camera_views.keys()),
                    default=default_views
                )
                
                if st.button("Generate Publication Figure"):
                    n_views = len(selected_views)
                    
                    if n_views == 0:
                        st.warning("Please select at least one view to generate a figure.")
                    else:
                        with st.spinner("Rendering multi-view figure (this may take a moment)..."):
                            
                            # Dynamically calculate grid layout based on selection
                            if n_views == 4:
                                cols = 2
                                rows = 2
                            else:
                                cols = min(3, n_views)
                                rows = math.ceil(n_views / cols)

                            # Build the dynamic grid specs
                            specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
                                      
                            fig_pub = make_subplots(
                                rows=rows, cols=cols,
                                specs=specs,
                                subplot_titles=selected_views,
                                vertical_spacing=0.05,
                                horizontal_spacing=0.01
                            )

                            scene_layouts = {}
                            
                            for idx, view_name in enumerate(selected_views):
                                r = (idx // cols) + 1
                                c = (idx % cols) + 1
                                
                                # Hide colorbar on all but the last plot to prevent overlap
                                current_args = mesh_args.copy()
                                if idx < n_views - 1 and 'showscale' in current_args:
                                    current_args['showscale'] = False
                                    
                                fig_pub.add_trace(go.Mesh3d(**current_args), row=r, col=c)
                                
                                # Setup scene layout for this specific subplot
                                scene_name = f'scene{idx+1}' if idx > 0 else 'scene'
                                scene_layouts[scene_name] = dict(
                                    xaxis=dict(visible=False), 
                                    yaxis=dict(visible=False), 
                                    zaxis=dict(visible=False), 
                                    camera=camera_views[view_name], 
                                    aspectmode='data'
                                )

                            # Apply layout updates and adjust height dynamically
                            fig_pub.update_layout(
                                **scene_layouts,
                                height=400 * rows,  # Scale height by number of rows
                                margin=dict(l=10, r=10, b=10, t=30),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig_pub, use_container_width=True, config={
                                'toImageButtonOptions': {
                                    'format': 'png', 
                                    'filename': 'neuro_mesh_multiview',
                                    'height': 600 * rows, # Higher resolution based on rows
                                    'width': 1800,
                                    'scale': 2
                                }
                            })
                            st.caption("Hover over the plot and click the 📷 icon in the top right corner to download this as a high-resolution PNG.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            
    else:
        st.info("👈 Please upload a GIFTI mesh file in the sidebar to begin.")