from dash import dcc, html

# LANDING PAGE LAYOUT

landing_layout = html.Div(
    style={
        'position':'fixed','top':'0','left':'0','right':'0','bottom':'0',
        'backgroundImage':"url('/assets/doctor_with_films_.jpg')",
        'backgroundRepeat':'no-repeat','backgroundSize':'cover','backgroundPosition':'center',
        'display':'flex','flexDirection':'column','justifyContent':'center','alignItems':'center',
        'color':'white','textAlign':'center','textShadow':'2px 2px 4px rgba(0,0,0,0.8)'
    },
    children=[
        html.H1("Welcome to the Diagnostic System", style={'fontSize':'48px','marginBottom':'20px','color': '#B2EBF2'}),
        html.P("Please upload the folder in ZIP format (containing the patient folders and the CSV with the metrics).",
               style={'fontSize':'24px','marginBottom':'40px','color': '#B2EBF2'}),
        dcc.Loading(
            id='loading-container',
            children=html.Div(
                id='upload-container',
                children=[
                    dcc.Upload(
                        id='upload-zip',
                        children=html.Div("Drag & drop ZIP here, or click to select."),
                        style={'width':'60%','margin':'0 auto','padding':'20px','border':'2px dashed #fff',
                               'textAlign':'center','cursor':'pointer','color': '#B2EBF2'},
                        multiple=False,
                        disabled=False
                    )
                ]
            )
        )
    ]
)

# DASHBOARD LAYOUT (3D, 2D, Metrics)
dashboard_layout = html.Div(
    style={
        'backgroundColor': '#ADD8E6',
        'margin': '0',
        'padding': '0'
    },
    children=[
        html.H1(
            "Brain MRI Diagnostic Interface",
            style={
                'textAlign': 'center',
                'margin': '0',
                'padding': '10px 0'
            }
        ),
        html.P(
            "Select from the following menu options:",
            style={
                'textAlign': 'center',
                'margin': '0',
                'padding': '5px 0',
                'fontSize': '18px'
            }
        ),
        # Tab header
        dcc.Tabs(
            id="main-tabs",
            value='tab-3d',
            style={
                'backgroundColor': '#444C56',
                'margin': '0',
                'padding': '0',
                'border': 'none'
            },
            children=[
                # 3D TAB
                dcc.Tab(
                    label="3D Brain MRI Visualization",
                    value='tab-3d',
                    style={'border': 'none','fontWeight': 'bold'},
                    selected_style={'border': 'none'},
                    children=[
                        html.H2(
                            style={'margin': '0', 'padding': '10px', 'fontSize': '24px'}
                        ),
                     html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Label(
                                            "Select Patient:",
                                            style={'fontSize': '18px', 'fontWeight': 'bold'}
                                        ),
                                        dcc.Dropdown(
                                            id="patient-dropdown-3d",
                                            options=[],
                                            value=None,
                                            searchable=False,
                                            clearable=False
                                        )
                                    ],
                                    style={'width': '45%'}
                                ),
                                html.Div(
                                    children=[
                                        html.Label(
                                            "Select Sequence:",
                                            style={'fontSize': '18px', 'fontWeight': 'bold'}
                                        ),
                                        dcc.Dropdown(
                                            id="sequence-dropdown-3d",
                                            options=[],
                                            value=None
                                        )
                                    ],
                                    style={'width': '45%'}
                                )
                            ],
                            style={
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'width': '80%',
                                'margin': '0 auto',
                                'marginBottom': '20px'
                            }
                        ),
                        html.Div(
                            children=[
                                    dcc.Graph(
                                    id="brain-graph",
                                    style={"height": "900px", "width": "100%"},
                                    # ↓ Disable entire modebar
                                    config={"displayModeBar": False}
                              )
                            ],
                            style={'width': '80%', 'margin': '0 auto', 'display': 'block'}
                        )
                    ]
                ),

                # 2D TAB
                dcc.Tab(
                    label="2D Brain MRI Visualization",
                    value='tab-2d',
                    style={'border': 'none','fontWeight': 'bold'},
                    children=[
                        html.Div(
                            style={
                                'backgroundColor': '#ADD8E6',
                                'padding': '20px',
                                'minHeight': 'calc(100vh - 150px)',
                                'textAlign': 'left'
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Label(
                                            "Select Patient:",
                                            style={
                                                'fontSize': '18px',
                                                'fontWeight': 'bold',
                                                'marginTop': '20px'
                                            }
                                        ),
                                        dcc.Dropdown(
                                            id="patient-dropdown-2d",
                                            options=[],
                                            value=None,
                                            searchable=False,
                                            clearable=False,
                                            style={'width': '50%'}
                                        )
                                    ],
                                    style={'paddingBottom': '20px'}
                                ),
                                html.Div(
                                    id="thumbnails-container",
                                    style={
                                        'display': 'flex',
                                        'flexWrap': 'wrap',
                                        'justifyContent': 'center'
                                    }
                                )
                            ]
                        )
                    ]
                ),

                # METRICS TAB
                dcc.Tab(
                    label="Metrics",
                    value='tab-metrics',
                    style={'border': 'none','fontWeight': 'bold'},
                    selected_style={'border': 'none'},
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    "Metric display mode:",
                                    style={'fontSize': '20px', 'fontWeight': 'bold','marginTop': '20px'}
                                ),
                                dcc.RadioItems(
                                    id="metrics-mode",
                                    options=[
                                        {"label": "Given patient", "value": "patient"},
                                        {"label": "All patients", "value": "all"}
                                    ],
                                    value="patient",
                                    labelStyle={'display': 'inline-block', 'margin-right': '20px','fontSize': '20px','marginTop': '20px'}
                                )
                            ],
                            style={
                                'width': '60%',
                                'margin': 'auto',
                                'paddingBottom': '20px',
                                'marginTop': '20px',
                                'textAlign': 'center'
                            }
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Select Patient:",
                                    style={'fontSize': '18px', 'fontWeight': 'bold'}
                                ),
                                dcc.Dropdown(
                                    id="patient-dropdown-metrics",
                                    options=[],
                                    value=None,
                                    searchable=False,
                                    clearable=False,
                                    style={'width': '40%', 'margin': 'auto'}
                                )
                            ],
                            id="metrics-patient-dropdown-container",
                            style={'marginBottom': '20px', 'textAlign': 'center'}
                        ),
                        dcc.Graph(
                            id="metrics-graph",
                            style={'height': '900px'},
                            config={"displayModeBar": False}
                        )
                    ]
                ),
            ]
        ),
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
                "zIndex": 9999
            },
            children=html.Div(
                children=[
                    html.Button(
                        "X",
                        id="close-modal",
                        n_clicks=0,
                        style={
                            'position': 'absolute',
                            'top': '10px',
                            'right': '10px',
                            'zIndex': 10000,
                            'fontSize': '24px',
                            'color': 'white',
                            'backgroundColor': 'red',
                            'border': 'none',
                            'borderRadius': '4px',
                            'padding': '5px 10px',
                            'cursor': 'pointer'
                        }
                    ),
                    html.Img(
                        id="modal-image",
                        style={
                            'width': '100%',
                            'height': '100%',
                            'objectFit': 'contain'
                        }
                    )
                ],
                style={
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'width': '90%',
                    'height': '90%',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center'
                }
            )
        )
    ]
)

# MAIN LAYOUT
main_layout = html.Div(
    style={
        'margin': '0',
        'padding': '0',
        'width': '100%',
        'height': '100%',
        'position': 'relative',
        'overflow': 'visible'
    },
    children=[
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='store-base-path'),
        html.Div(id='page-content', style={'height': '100%', 'width': '100%'})
    ]
)