import atexit
import os
import re
import io
import base64
import shutil
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

# Import the layout definitions
from layout import landing_layout, dashboard_layout, main_layout

# Path to the folder you want to clean up:
UPLOAD_DIR = os.path.join(os.getcwd(), 'uploaded_data')

def cleanup_uploaded_data():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        print(f"Cleaned up folder: {UPLOAD_DIR}")

# Register the cleanup function to run on interpreter exit
atexit.register(cleanup_uploaded_data)

# Global caches and state
nifti_cache = {}
thumbnail_cache = {}
full_image_cache = {}
uploaded_base_path = None


# Returns a sorted list of patient directory names from the uploaded base path if it exists.
def get_patient_dirs():
    global uploaded_base_path
    if uploaded_base_path and os.path.exists(uploaded_base_path):
        dirs = [d for d in os.listdir(uploaded_base_path)
                if os.path.isdir(os.path.join(uploaded_base_path, d))]
        dirs.sort()
        return dirs
    return []

# Returns a dictionary mapping sequence types (e.g. T1, T2, FLAIR, etc.) to their corresponding .nii file paths for a given patient directory.
def get_sequence_files(patient_dir):
    global uploaded_base_path
    full_path = os.path.join(uploaded_base_path, patient_dir)
    nii_files = [f for f in os.listdir(full_path) if f.endswith('.nii')]
    patterns = {
        "t1ce": re.compile(r't1ce', re.IGNORECASE),
        "t1":  re.compile(r't1(?!ce)', re.IGNORECASE),
        "t2":  re.compile(r't2', re.IGNORECASE),
        "flair": re.compile(r'flair', re.IGNORECASE),
        "seg": re.compile(r'seg', re.IGNORECASE)
    }
    seq_files = {}
    for f in nii_files:
        for key, pat in patterns.items():
            if pat.search(f):
                seq_files[key] = os.path.join(full_path, f)
                break
    return seq_files

# Loads a NIfTI file and caches the result for future access.
def load_nifti(fp):
    if fp in nifti_cache:
        return nifti_cache[fp]
    img = nib.load(fp)
    data = img.get_fdata()
    nifti_cache[fp] = data
    return data

#Centers a 3D mesh by shifting vertices to the origin.
def center_mesh(verts):
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    return verts - ((mn + mx) / 2)

# Generates a 3D mesh (vertices and faces) from volumetric data using the marching cubes algorithm.
def create_mesh(data, level=1.5):
    verts, faces, _, _ = measure.marching_cubes(data, level=level)
    return center_mesh(verts), faces

# Creates and returns a 3D Plotly figure from one or two meshes with customizable opacity.
def create_figure(verts, faces, t_verts=None, t_faces=None, opacity=0.6):
    mesh = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        opacity=opacity, color='lightgrey'
    )
    data = [mesh]
    if t_verts is not None:
        tmesh = go.Mesh3d(
            x=t_verts[:,0], y=t_verts[:,1], z=t_verts[:,2],
            i=t_faces[:,0], j=t_faces[:,1], k=t_faces[:,2],
            opacity=1, color='red'
        )
        data.append(tmesh)
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=0.8,y=0.8,z=0.8)),
            xaxis=dict(showbackground=False,visible=False,showticklabels=False),
            yaxis=dict(showbackground=False,visible=False,showticklabels=False),
            zaxis=dict(showbackground=False,visible=False,showticklabels=False),
        ),
        width=1800, height=800
    )
    return fig

# Generates a PNG image (as a base64 string) from a selected 2D slice of 3D data
def create_png_from_slice(data, idx, dpi=80):
    from PIL import Image
    import matplotlib.pyplot as plt
    if dpi <= 50:
        sl = data[:, :, idx]
        norm = (sl - sl.min())/(sl.max()-sl.min()+1e-8)
        arr = (norm*255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
    else:
        fig, ax = plt.subplots()
        ax.imshow(data[:,:,idx], cmap='gray')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')


# --- Dash app setup ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
suppress_callback_exceptions = True,
dev_tools_ui = False,
dev_tools_props_check = False
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Diagnostic System</title>
    {%favicon%}
    {%css%}
    <style>html, body { margin:0; padding:0; width:100%; height:100%; }</style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""
app.layout = main_layout

# --- Routing ---
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(path):
    return dashboard_layout if path == '/dashboard' else landing_layout

# --- ZIP Upload ---
@app.callback(
    [Output('store-base-path', 'data'),
     Output('upload-container', 'children'),
     Output('upload-zip', 'disabled')],
    Input('upload-zip', 'contents'),
    State('upload-zip', 'filename')
)

def upload_zip(contents, filename):
    global uploaded_base_path
    if contents is None:
        return dash.no_update, dash.no_update, False

    processing = html.Div("Processing upload… Please wait.", style={'fontSize': '20px'})
    content_type, b64 = contents.split(',')
    data = base64.b64decode(b64)
    folder = os.path.join(os.getcwd(), 'uploaded_data')
    os.makedirs(folder, exist_ok=True)

    # Sample for file types
    patterns = {
        "t1ce": re.compile(r't1ce', re.IGNORECASE),
        "t1": re.compile(r't1(?!ce)', re.IGNORECASE),
        "t2": re.compile(r't2', re.IGNORECASE),
        "flair": re.compile(r'flair', re.IGNORECASE),
        "seg": re.compile(r'seg', re.IGNORECASE)
    }

    try:
        with zipfile.ZipFile(io.BytesIO(data), 'r') as zp:
            zp.extractall(folder)

        subs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        uploaded_base_path = os.path.join(folder, subs[0]) if len(subs) == 1 else folder

        # Check: is there a suitable .nii file
        found_valid_file = False
        for root, dirs, files in os.walk(uploaded_base_path):
            for file in files:
                if file.endswith('.nii'):
                    for pat in patterns.values():
                        if pat.search(file):
                            found_valid_file = True
                            break
                if found_valid_file:
                    break
            if found_valid_file:
                break

        if not found_valid_file:
            raise ValueError("The ZIP does not contain the expected types of .nii files: t1ce, t1, t2, flair, or seg.")

        cont = dcc.Link("Continue to the system", href="/dashboard",
                        style={'fontSize': '24px', 'color': '#012E40', 'fontWeight': 'bold',
                               'textDecoration': 'none', 'backgroundColor': '#4FC3F7',
                               'padding': '10px 20px', 'borderRadius': '10px'})
        return uploaded_base_path, cont, True

    except Exception as err:
        err_div = html.Div(f" Upload error:  {err}", style={
            'color': 'red', 'fontSize': '25px', 'fontWeight': 'bold', 'paddingBottom': '10px'
        })
        upload_box = dcc.Upload(
            id='upload-zip', children=html.Div("Click here to upload the ZIP file."),
            style={'width': '60%', 'margin': '0 auto', 'padding': '20px', 'border': '2px dashed #fff',
                   'textAlign': 'center', 'cursor': 'pointer'}, multiple=False, disabled=False
        )
        return dash.no_update, html.Div([err_div, upload_box]), False

# Dropdown updates
@app.callback(
    [Output('patient-dropdown-3d', 'options'), Output('patient-dropdown-3d', 'value'),
     Output('patient-dropdown-2d', 'options'), Output('patient-dropdown-2d', 'value'),
     Output('patient-dropdown-metrics', 'options'), Output('patient-dropdown-metrics', 'value')],
    Input('store-base-path', 'data')
)
def update_patient_dropdowns(_):
    dirs = get_patient_dirs()
    opts = [{'label': p, 'value': p} for p in dirs]
    default = opts[0]['value'] if opts else None
    return opts, default, opts, default, opts, default

# Sequence dropdown
@app.callback(
    [Output('sequence-dropdown-3d', 'options'), Output('sequence-dropdown-3d', 'value')],
    Input('patient-dropdown-3d', 'value')
)
def update_seq_3d(patient):
    if not patient:
        return [], None
    seq = get_sequence_files(patient)
    opts = [{'label': k, 'value': k} for k in seq]
    default = opts[0]['value'] if opts else None
    return opts, default

# 3D view
@app.callback(
    Output('brain-graph', 'figure'),
    [Input('patient-dropdown-3d', 'value'), Input('sequence-dropdown-3d', 'value')]
)


def update_3d_graph(patient, seq_key):
    if not patient or not seq_key:
        return go.Figure()
    seq = get_sequence_files(patient)
    fp = seq.get(seq_key)
    seg_fp = seq.get('seg')
    if not fp:
        return go.Figure()
    try:
        data = load_nifti(fp)

        # Dynamic level
        factor_map = {'t1': 0.25, 't2': 0.3, 't1ce': 0.35}
        factor = factor_map.get(seq_key, 0.3)
        v, f = create_mesh(data, factor)

        tv, tf = (None, None)
        if seg_fp:
            td = load_nifti(seg_fp)
            tv, tf = create_mesh(td, 0.4)

        # Dynamic opacity
        opacity_map = {'t1': 0.3, 't2': 0.6, 't1ce': 0.9}
        opacity = opacity_map.get(seq_key, 0.6)

        return create_figure(v, f, tv, tf, opacity=opacity)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=str(e), x=0.5, y=0.5, showarrow=False)
        return fig

# 2D slices: triggered by dropdown
@app.callback(
    Output('thumbnails-container', 'children'),
    Input('patient-dropdown-2d', 'value')
)
def load_slices(patient):
    if not patient:
        return []
    seq = get_sequence_files(patient)
    fp = next((seq[k] for k in seq if k.lower()!='seg'), None)
    if not fp:
        return ['No available 2D data series.']
    data = load_nifti(fp)
    thumbs = []
    for idx in range(data.shape[2]):
        key = (patient, fp, idx, 'thumb')
        src = thumbnail_cache.get(key) or create_png_from_slice(data, idx, dpi=50)
        thumbnail_cache[key] = src
        thumbs.append(html.Img(src=src, id={'type':'thumbnail','index':idx}, n_clicks=0,
                                 style={'width':'100px','margin':'5px','cursor':'pointer'}))
    return thumbs


# Displaying a larger image
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
            src = create_png_from_slice(data, index, dpi=800)
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
            "zIndex": 9999
        }
        return modal_style, src
    except Exception as e:
        return {"display": "none"}, ""

# Updating the METRICS chart
@app.callback(
    Output("metrics-graph", "figure"),
    [
        Input("metrics-mode", "value"),
        Input("patient-dropdown-metrics", "value"),
        Input("metrics-table-type-dropdown", "value")
    ]
)

def update_metrics_graph(mode, selected_patient, table_type):
    global uploaded_base_path
    csv_path = os.path.join(uploaded_base_path, "metrics_output.csv")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[DEBUG] Error reading CSV: {e}")  # Developer log
        fig = go.Figure()
        fig.add_annotation(
            text="Failed to load performance data. Please check the uploaded file.",
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

    fig = go.Figure()

    # --- Performance Metrics by Tumor Type ---
    if table_type == "tumor-metrics":
        all_metrics = ["TPR", "TNR", "PPV", "NPV", "ACC", "DS"]

        if mode == "patient":
            if not selected_patient:
                return fig
            df_patient = df[df["Name"] == selected_patient]
            if df_patient.empty:
                fig.add_annotation(
                    text="No metric data available for the selected patient.",
                    x=0.5, y=0.5,
                    xref='paper', yref='paper',
                    xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color="red")
                )
                return fig
            for metric in all_metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_patient["TumorType"],
                    y=df_patient[metric],
                    text=df_patient[metric].round(3),
                    textposition='outside',
                    hovertemplate='%{x}<br>' + metric + ': %{y:.3f}<extra></extra>'
                ))
            right_header = f"Patient Metrics: {selected_patient}"
        else:
            grouped = df.groupby("Name")[all_metrics].mean().reset_index()
            for metric in all_metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=grouped["Name"],
                    y=grouped[metric]
                ))
            right_header = "All Patients"

        fig.update_layout(
            barmode='group',
            title=None,
            xaxis=dict(title=''),
            yaxis_title="Value",
            title_font_size=22,
            font=dict(size=16),
            yaxis=dict(range=[0, 1.50]),
            annotations=[
                dict(
                    text="Performance Metrics by Tumor Type",
                    x=0, y=1.10,
                    xref='paper', yref='paper',
                    xanchor='left', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=20, color="black", family="Arial")
                ),
                dict(
                    text=right_header,
                    x=1, y=1.10,
                    xref='paper', yref='paper',
                    xanchor='right', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=20, color="black", family="Arial")
                ),
            ],
            legend=dict(
                orientation='h',
                x=0.5, y=1.00,
                xanchor='center', yanchor='bottom'
            ),
            margin=dict(t=120),
        )

    # --- Confusion Matrix & Performance Metrics ---
    elif table_type == "confusion-matrix":
        all_metrics = ["TP", "TN", "FP", "FN"]

        if mode == "patient":
            if not selected_patient:
                return fig
            df_patient = df[df["Name"] == selected_patient]
            if df_patient.empty:
                fig.add_annotation(
                    text="No confusion matrix data available for the selected patient.",
                    x=0.5, y=0.5,
                    xref='paper', yref='paper',
                    xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color="black")
                )
                return fig
            for metric in all_metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_patient["TumorType"],
                    y=df_patient[metric],
                    text=df_patient[metric],
                    textposition='outside',
                    hovertemplate='%{x}<br>' + metric + ': %{y}<extra></extra>'
                ))
            right_header = f"Confusion Matrix: {selected_patient}"
        else:
            grouped = df.groupby("Name")[all_metrics].sum().reset_index()
            for metric in all_metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=grouped["Name"],
                    y=grouped[metric]
                ))
            right_header = "Confusion Matrix: All Patients"

        fig.update_layout(
            barmode='group',
            title=None,
            xaxis=dict(title=''),
            yaxis_title="Count",
            title_font_size=22,
            font=dict(size=16),
            annotations=[
                dict(
                    text="Confusion Matrix & Performance Metrics",
                    x=0, y=1.10,
                    xref='paper', yref='paper',
                    xanchor='left', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=20, color="black", family="Arial")
                ),
                dict(
                    text=right_header,
                    x=1, y=1.10,
                    xref='paper', yref='paper',
                    xanchor='right', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=20, color="black", family="Arial")
                ),
            ],
            legend=dict(
                orientation='h',
                x=0.5, y=1.00,
                xanchor='center', yanchor='bottom'
            ),
            margin=dict(t=120),
        )

    return fig

if __name__ == '__main__':
    try:
        app.run_server(debug=True)
    finally:
        folder = os.path.join(os.getcwd(), 'uploaded_data')
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted folder on shutdown: {folder}")