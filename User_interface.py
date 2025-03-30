import os
import re
import io
import base64
import zipfile
import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL

# Global cache for faster loading
nifti_cache = {}         # Cache for nifti files
thumbnail_cache = {}     # Cache for thumbnail images
full_image_cache = {}    # Cache for high resolution images

# Global variable storing the path of the uploaded data folder
uploaded_base_path = None

# List patient directories based on the new base_path if the file is already uploaded
def get_patient_dirs():
    global uploaded_base_path
    if uploaded_base_path and os.path.exists(uploaded_base_path):
        dirs = [d for d in os.listdir(uploaded_base_path) if os.path.isdir(os.path.join(uploaded_base_path, d))]
        dirs.sort()
        return dirs
    return []

# List data and group by sequence in a given patient folder
def get_sequence_files(patient_dir):
    global uploaded_base_path
    full_path = os.path.join(uploaded_base_path, patient_dir)
    nii_files = [f for f in os.listdir(full_path) if f.endswith('.nii')]
    patterns = {
        "t1ce": re.compile(r't1ce', re.IGNORECASE),
        "t1": re.compile(r't1(?!ce)', re.IGNORECASE),
        "t2": re.compile(r't2', re.IGNORECASE),
        "flair": re.compile(r'flair', re.IGNORECASE),
        "seg": re.compile(r'seg', re.IGNORECASE)
    }
    seq_files = {}
    for f in nii_files:
        for key, pattern in patterns.items():
            if pattern.search(f):
                seq_files[key] = os.path.join(full_path, f)
                break
    return seq_files

# Load nifti file with caching
def load_nifti(file_path):
    if file_path in nifti_cache:
        return nifti_cache[file_path]
    img = nib.load(file_path)
    data = img.get_fdata()
    nifti_cache[file_path] = data
    return data

# Center the 3D mesh
def center_mesh(verts):
    min_vals = np.min(verts, axis=0)
    max_vals = np.max(verts, axis=0)
    center = (min_vals + max_vals) / 2
    return verts - center

# Generate 3D mesh using the marching cubes algorithm
def create_mesh(data, level=1.5):
    verts, faces, _, _ = measure.marching_cubes(data, level=level)
    verts = center_mesh(verts)
    return verts, faces

# Create 3D plot using Plotly
def create_figure(verts, faces, tumor_verts=None, tumor_faces=None):
    brain_mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5, color='lightgrey'
    )
    fig_data = [brain_mesh]
    if tumor_verts is not None and tumor_faces is not None:
        tumor_mesh = go.Mesh3d(
            x=tumor_verts[:, 0], y=tumor_verts[:, 1], z=tumor_verts[:, 2],
            i=tumor_faces[:, 0], j=tumor_faces[:, 1], k=tumor_faces[:, 2],
            opacity=1, color='red'
        )
        fig_data.append(tumor_mesh)
    camera = dict(eye=dict(x=0.8, y=0.8, z=0.8))
    fig = go.Figure(data=fig_data)
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=camera,
            xaxis=dict(showbackground=False, visible=False, showticklabels=False),
            yaxis=dict(showbackground=False, visible=False, showticklabels=False),
            zaxis=dict(showbackground=False, visible=False, showticklabels=False),
        ),
        width=1600,
        height=800
    )
    return fig

# Generate PNG image from slice
def create_png_from_slice(data, slice_index, dpi=80):
    if dpi <= 50:
        slice_data = data[:, :, slice_index]
        normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        image_array = (normalized * 255).astype(np.uint8)
        img = Image.fromarray(image_array, mode='L')
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return "data:image/png;base64," + encoded
    else:
        fig, ax = plt.subplots()
        ax.imshow(data[:, :, slice_index], cmap='gray')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return "data:image/png;base64," + encoded

# Create DASH application with suppress_callback_exceptions parameter
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ----------------------
# MAIN LAYOUT
# ----------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # These Store components are always available regardless of the page
    dcc.Store(id='store-base-path'),
    html.Div(id='page-content')
])

# ----------------------
# LANDING PAGE LAYOUT
# ----------------------
landing_layout = html.Div(
    style={
        'backgroundImage': "url('/assets/doctor_with_films_.jpg')",
        'backgroundSize': 'cover',
        'height': '100vh',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'flexDirection': 'column',
        'color': 'white',
        'textAlign': 'center',
        'textShadow': '2px 2px 4px rgba(0, 0, 0, 0.8)'
    },
    children=[
        html.H1("Welcome to the Diagnostic System", style={'fontSize': '48px', 'marginBottom': '20px'}),
        html.P("Please upload the folder in ZIP format (containing the patient folders and the CSV with the metrics).",
               style={'fontSize': '24px', 'marginBottom': '40px'}),
        dcc.Upload(
            id='upload-zip',
            children=html.Div("Drag and drop here or click to upload the ZIP file."),
            style={
                'width': '60%',
                'margin': '10px auto',
                'padding': '20px',
                'border': '2px dashed #fff',
                'textAlign': 'center',
                'cursor': 'pointer',
                'textShadow': '2px 2px 4px rgba(0, 0, 0, 0.9)'
            },
            multiple=False
        ),
        html.Div(id='proceed-link-container')
    ]
)

# ----------------------
# DASHBOARD LAYOUT (3D, 2D, Metrics)
# ----------------------
dashboard_layout = html.Div([
    html.H1("Brain MRI Diagnostic Interface", style={'textAlign': 'center'}),
    html.P("Select from the following menu options:", style={'textAlign': 'center'}),
    dcc.Tabs(id="main-tabs", value='tab-3d', children=[
        # 3D TAB
        dcc.Tab(label="3D", value='tab-3d', children=[
            html.H2("3D Brain MRI Visualization"),
            html.Div([
                html.Div([
                    html.Label("Select Patient:"),
                    dcc.Dropdown(
                        id="patient-dropdown-3d",
                        options=[],
                        value=None
                    )
                ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '5%'}),
                html.Div([
                    html.Label("Select Sequence:"),
                    dcc.Dropdown(
                        id="sequence-dropdown-3d",
                        options=[],
                        value=None
                    )
                ], style={'width': '45%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'}),
            html.Div([
                dcc.Graph(id="brain-graph", style={"height": "900px", "width": "100%"})
            ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'})
        ]),
        # 2D TAB
        dcc.Tab(label="2D", value='tab-2d', children=[
            html.H2("2D Brain MRI Visualization"),
            html.Div([
                html.Label("Select Patient:"),
                dcc.Dropdown(
                    id="patient-dropdown-2d",
                    options=[],
                    value=None,
                    style={'width': '50%'}
                )
            ], style={'margin': 'auto', 'paddingBottom': '20px'}),
            html.Button("Loading", id="load-slices-button", n_clicks=0, style={
                'display': 'block',
                'margin': '10px auto',
                'padding': '10px 20px',
                'fontSize': '16px'
            }),
            html.Div(
                id="thumbnails-container",
                style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}
            )
        ]),
        # METRICS TAB
        dcc.Tab(label="Metrics", value='tab-metrics', children=[
            html.H2("Metrics"),
            html.Div([
                html.Label("Metric display mode:"),
                dcc.RadioItems(
                    id="metrics-mode",
                    options=[
                        {"label": "Given patient", "value": "patient"},
                        {"label": "All patients", "value": "all"}
                    ],
                    value="patient",
                    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                )
            ], style={'width': '60%', 'margin': 'auto', 'paddingBottom': '20px', 'textAlign': 'center'}),
            html.Div([
                html.Label("Select Patient:"),
                dcc.Dropdown(
                    id="patient-dropdown-metrics",
                    options=[],
                    value=None,
                    style={'width': '40%', 'margin': 'auto'}
                )
            ], id="metrics-patient-dropdown-container", style={'marginBottom': '20px', 'textAlign': 'center'}),
            dcc.Graph(id="metrics-graph", style={'height': '900px'})
        ]),
    ]),
    # 2D enlarged image modal
    html.Div(
        id="modal",
        style={
            "display": "none",
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0,0,0,0.5)",
            "justifyContent": "center",
            "alignItems": "center",
            "zIndex": 1000
        },
        children=html.Div([
            html.Button("X", id="close-modal", n_clicks=0, style={
                'position': 'absolute',
                'top': '10px',
                'right': '10px',
                'zIndex': 1001,
                'fontSize': '24px',
                'color': 'white',
                'backgroundColor': 'red',
                'border': 'none',
                'borderRadius': '4px',
                'padding': '5px 10px',
                'cursor': 'pointer'
            }),
            html.Img(
                id="modal-image",
                style={
                    'width': '100%',
                    'height': '100%',
                    'objectFit': 'contain'
                }
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'width': '90%',
            'height': '90%',
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center'
        })
    )
])

# ----------------------
# URL-BASED ROUTING
# ----------------------
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dashboard':
        return dashboard_layout
    else:
        return landing_layout

# ----------------------
# CALLBACKS FOR ZIP UPLOAD TO LANDING PAGE
# ----------------------
@app.callback(
    [Output('store-base-path', 'data'),
     Output('proceed-link-container', 'children')],
    [Input('upload-zip', 'contents')],
    [State('upload-zip', 'filename')]
)
def handle_zip_upload(contents, filename):
    global uploaded_base_path
    if contents is None:
        return dash.no_update, ""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Create the "uploaded_data" folder (if it does not already exist)
        upload_folder = os.path.join(os.getcwd(), "uploaded_data")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        # Extract the contents of the ZIP into the upload_folder
        with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
            zip_ref.extractall(upload_folder)
        # If there is exactly one subdirectory inside the upload_folder, use it as the base_path
        subdirs = [d for d in os.listdir(upload_folder) if os.path.isdir(os.path.join(upload_folder, d))]
        if len(subdirs) == 1:
            uploaded_base_path = os.path.join(upload_folder, subdirs[0])
        else:
            uploaded_base_path = upload_folder
        return uploaded_base_path, dcc.Link("Continue to the system", href="/dashboard", style={
            'fontSize': '24px',
            'color': 'white',
            'textDecoration': 'underline',
            'border': '2px solid white',
            'padding': '10px 20px',
            'borderRadius': '5px'
        })
    except Exception as e:
        return dash.no_update, html.Div(f"Error during upload: {str(e)}", style={'color': 'red'})

# ----------------------
# CALLBACKS FOR DASHBOARD ELEMENTS
# ----------------------

# Update patient dropdown options based on the uploaded folder
@app.callback(
    [Output("patient-dropdown-3d", "options"),
     Output("patient-dropdown-3d", "value"),
     Output("patient-dropdown-2d", "options"),
     Output("patient-dropdown-2d", "value"),
     Output("patient-dropdown-metrics", "options"),
     Output("patient-dropdown-metrics", "value")],
    [Input('store-base-path', 'data')]
)
def update_patient_dropdowns(base_path_data):
    dirs = get_patient_dirs()
    options = [{"label": p, "value": p} for p in dirs]
    default = options[0]["value"] if options else None
    return options, default, options, default, options, default

# 3D CALLBACK
@app.callback(
    [Output("sequence-dropdown-3d", "options"),
     Output("sequence-dropdown-3d", "value")],
    [Input("patient-dropdown-3d", "value")]
)
def update_sequence_dropdown_3d(selected_patient):
    if not selected_patient:
        return [], None
    seq_files = get_sequence_files(selected_patient)
    options = [{"label": key, "value": key} for key in seq_files.keys()]
    default = options[0]["value"] if options else None
    return options, default

@app.callback(
    Output("brain-graph", "figure"),
    [Input("patient-dropdown-3d", "value"),
     Input("sequence-dropdown-3d", "value")]
)
def update_3d_graph(selected_patient, selected_sequence):
    if not selected_patient or not selected_sequence:
        return go.Figure()
    seq_files = get_sequence_files(selected_patient)
    file_path = seq_files.get(selected_sequence)
    tumor_path = seq_files.get("seg")
    if not file_path:
        return go.Figure()
    try:
        data = load_nifti(file_path)
        verts, faces = create_mesh(data, level=0.5)
        tumor_verts, tumor_faces = None, None
        if tumor_path:
            tumor_data = load_nifti(tumor_path)
            tumor_verts, tumor_faces = create_mesh(tumor_data, level=0.5)
        fig = create_figure(verts, faces, tumor_verts, tumor_faces)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
    return fig

# 2D CALLBACKS
@app.callback(
    Output("thumbnails-container", "children"),
    [Input("load-slices-button", "n_clicks")],
    [State("patient-dropdown-2d", "value")]
)
def load_slices(n_clicks, selected_patient):
    if not n_clicks or not selected_patient:
        return []
    seq_files = get_sequence_files(selected_patient)
    file_path = None
    for key in seq_files:
        if key.lower() != "seg":
            file_path = seq_files[key]
            break
    if not file_path:
        return ["No available 2D data series."]
    try:
        data = load_nifti(file_path)
        n_slices = data.shape[2]
        thumbnails = []
        for idx in range(n_slices):
            cache_key = (selected_patient, file_path, idx, 'thumb')
            if cache_key in thumbnail_cache:
                png_src = thumbnail_cache[cache_key]
            else:
                png_src = create_png_from_slice(data, idx, dpi=50)
                thumbnail_cache[cache_key] = png_src
            thumbnails.append(html.Img(
                src=png_src,
                id={'type': 'thumbnail', 'index': idx},
                n_clicks=0,
                style={'width': '100px', 'margin': '5px', 'cursor': 'pointer'}
            ))
        return thumbnails
    except Exception as e:
        return [f"Error: {str(e)}"]

@app.callback(
    [Output("modal", "style"), Output("modal-image", "src")],
    [Input({'type': 'thumbnail', 'index': ALL}, 'n_clicks'),
     Input("close-modal", "n_clicks")],
    [State("patient-dropdown-2d", "value")]
)
def display_modal(thumbnail_clicks, close_click, patient):
    ctx = dash.callback_context
    if not thumbnail_clicks or all(n is None or n == 0 for n in thumbnail_clicks):
        return {"display": "none"}, ""
    if not ctx.triggered:
        return {"display": "none"}, ""
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if "close-modal" in trigger_id:
        return {"display": "none"}, ""
    if "type" not in trigger_id or "thumbnail" not in trigger_id:
        return {"display": "none"}, ""
    try:
        clicked = eval(trigger_id)
        index = clicked['index']
    except Exception:
        return {"display": "none"}, ""
    seq_files = get_sequence_files(patient)
    file_path = None
    for key in seq_files:
        if key.lower() != "seg":
            file_path = seq_files[key]
            break
    if not file_path:
        return {"display": "none"}, ""
    try:
        data = load_nifti(file_path)
        cache_key = (patient, file_path, index, 'full')
        if cache_key in full_image_cache:
            src = full_image_cache[cache_key]
        else:
            src = create_png_from_slice(data, index, dpi=300)
            full_image_cache[cache_key] = src
        modal_style = {
            "display": "flex",
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0,0,0,0.5)",
            "justifyContent": "center",
            "alignItems": "center",
            "zIndex": 1000
        }
        return modal_style, src
    except Exception as e:
        return {"display": "none"}, ""

# METRICS CALLBACK
@app.callback(
    Output("metrics-graph", "figure"),
    [Input("metrics-mode", "value"),
     Input("patient-dropdown-metrics", "value")]
)
def update_metrics_graph(mode, selected_patient):
    global uploaded_base_path
    csv_path = os.path.join(uploaded_base_path, "metrics_output.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error reading CSV: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

    # List of all metric names
    all_metrics = ["TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "ACC", "DS"]
    fig = go.Figure()
    if mode == "patient":
        if not selected_patient:
            return go.Figure()
        df_patient = df[df["Name"] == selected_patient]
        if df_patient.empty:
            fig.add_annotation(text="No metric data available for the selected patient.", x=0.5, y=0.5, showarrow=False)
            return fig
        for metric in all_metrics:
            if metric in ["TP", "TN", "FP", "FN"]:
                values = df_patient[metric] / 1000.0
            else:
                values = df_patient[metric]
            fig.add_trace(go.Bar(
                name=metric,
                x=df_patient["TumorType"],
                y=values
            ))
        title_text = f"Patient Metrics: {selected_patient}"
        xaxis_title = "Tumor Type"
    else:
        # All patients: average the metrics for each patient
        grouped = df.groupby("Name")[all_metrics].mean().reset_index()
        for metric in all_metrics:
            if metric in ["TP", "TN", "FP", "FN"]:
                values = grouped[metric] / 1000.0
            else:
                values = grouped[metric]
            fig.add_trace(go.Bar(
                name=metric,
                x=grouped["Name"],
                y=values
            ))
        title_text = "Metrics for All Patients"
        xaxis_title = "Patient"
    fig.update_layout(
        barmode='group',
        title=title_text,
        xaxis_title=xaxis_title,
        yaxis_title="Value",
        title_font_size=22,
        font=dict(size=16),
        yaxis_type="log"
    )
    return fig
#Teszt
if __name__ == '__main__':
    app.run_server(debug=True)
