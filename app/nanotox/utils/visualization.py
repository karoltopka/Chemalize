"""
Visualization utilities for NanoTox predictions.

Generates Plotly charts for concentration-response curves and feature importance.
"""
import json


def create_concentration_curve(curve_data, title="Concentration-Response Curve"):
    """
    Create a Plotly concentration-response curve.

    Args:
        curve_data: List of (concentration, viability) tuples
        title: Chart title

    Returns:
        str: JSON string of Plotly figure
    """
    concentrations = [point[0] for point in curve_data]
    viabilities = [point[1] for point in curve_data]

    # Clip viabilities to 0-100 range for display
    viabilities_clipped = [max(0, min(100, v)) for v in viabilities]

    figure = {
        'data': [
            {
                'x': concentrations,
                'y': viabilities_clipped,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Cell Viability',
                'line': {
                    'color': '#ff6b6b',
                    'width': 3
                },
                'marker': {
                    'size': 8,
                    'color': '#ff6b6b'
                },
                'hovertemplate': (
                    'Concentration: %{x:.3f} ug/ml<br>'
                    'Viability: %{y:.1f}%<br>'
                    '<extra></extra>'
                )
            }
        ],
        'layout': {
            'title': {
                'text': title,
                'font': {'size': 18, 'color': '#ffffff'}
            },
            'xaxis': {
                'title': 'Concentration (ug/ml)',
                'type': 'log',
                'gridcolor': 'rgba(255,255,255,0.1)',
                'color': '#ffffff',
                'tickformat': '.3g'
            },
            'yaxis': {
                'title': 'Cell Viability (%)',
                'range': [0, 110],
                'gridcolor': 'rgba(255,255,255,0.1)',
                'color': '#ffffff'
            },
            'paper_bgcolor': 'rgba(26,26,26,1)',
            'plot_bgcolor': 'rgba(26,26,26,1)',
            'font': {'color': '#ffffff'},
            'hovermode': 'closest',
            'showlegend': False,
            'shapes': [
                {
                    'type': 'line',
                    'x0': min(concentrations),
                    'x1': max(concentrations),
                    'y0': 50,
                    'y1': 50,
                    'line': {
                        'color': 'rgba(255,255,255,0.3)',
                        'width': 1,
                        'dash': 'dash'
                    }
                }
            ],
            'annotations': [
                {
                    'x': max(concentrations) * 0.9,
                    'y': 52,
                    'text': 'IC50',
                    'showarrow': False,
                    'font': {'size': 10, 'color': 'rgba(255,255,255,0.5)'}
                }
            ]
        }
    }

    return json.dumps(figure)


def create_feature_importance_plot(importance_dict, title="Feature Importance"):
    """
    Create a Plotly horizontal bar chart for feature importance.

    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        title: Chart title

    Returns:
        str: JSON string of Plotly figure
    """
    if not importance_dict:
        return json.dumps({'data': [], 'layout': {'title': 'No data available'}})

    # Sort by absolute importance
    sorted_items = sorted(
        importance_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    features = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    # Color based on positive/negative contribution
    colors = ['#4ecdc4' if s >= 0 else '#ff6b6b' for s in scores]

    figure = {
        'data': [
            {
                'y': features,
                'x': scores,
                'type': 'bar',
                'orientation': 'h',
                'marker': {
                    'color': colors
                },
                'hovertemplate': '%{y}: %{x:.3f}<extra></extra>'
            }
        ],
        'layout': {
            'title': {
                'text': title,
                'font': {'size': 18, 'color': '#ffffff'}
            },
            'xaxis': {
                'title': 'Importance Score',
                'gridcolor': 'rgba(255,255,255,0.1)',
                'color': '#ffffff',
                'zeroline': True,
                'zerolinecolor': 'rgba(255,255,255,0.3)'
            },
            'yaxis': {
                'title': '',
                'color': '#ffffff',
                'automargin': True
            },
            'paper_bgcolor': 'rgba(26,26,26,1)',
            'plot_bgcolor': 'rgba(26,26,26,1)',
            'font': {'color': '#ffffff'},
            'margin': {'l': 150, 'r': 20, 't': 60, 'b': 40}
        }
    }

    return json.dumps(figure)


def create_feature_contribution_plot(contributions, prediction, title="Feature Contributions"):
    """
    Create a Plotly waterfall-style chart showing how each feature contributes to the prediction.

    Args:
        contributions: Dictionary mapping feature names to their contributions
        prediction: The final predicted value
        title: Chart title

    Returns:
        str: JSON string of Plotly figure
    """
    if not contributions:
        return json.dumps({'data': [], 'layout': {'title': 'No contribution data available'}})

    # Sort by contribution magnitude
    sorted_items = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Color based on positive/negative contribution
    colors = ['#4ecdc4' if v >= 0 else '#ff6b6b' for v in values]

    figure = {
        'data': [
            {
                'y': features,
                'x': values,
                'type': 'bar',
                'orientation': 'h',
                'marker': {
                    'color': colors,
                    'line': {
                        'color': 'rgba(255,255,255,0.3)',
                        'width': 1
                    }
                },
                'hovertemplate': (
                    '%{y}<br>'
                    'Contribution: %{x:+.2f}<br>'
                    '<extra></extra>'
                )
            }
        ],
        'layout': {
            'title': {
                'text': f'{title}<br><span style="font-size:14px">Predicted Viability: {prediction:.1f}%</span>',
                'font': {'size': 18, 'color': '#ffffff'}
            },
            'xaxis': {
                'title': 'Contribution to Prediction',
                'gridcolor': 'rgba(255,255,255,0.1)',
                'color': '#ffffff',
                'zeroline': True,
                'zerolinecolor': 'rgba(255,255,255,0.5)',
                'zerolinewidth': 2
            },
            'yaxis': {
                'title': '',
                'color': '#ffffff',
                'automargin': True
            },
            'paper_bgcolor': 'rgba(26,26,26,1)',
            'plot_bgcolor': 'rgba(26,26,26,1)',
            'font': {'color': '#ffffff'},
            'margin': {'l': 180, 'r': 20, 't': 80, 'b': 40},
            'annotations': [
                {
                    'x': 0,
                    'y': 1.05,
                    'xref': 'paper',
                    'yref': 'paper',
                    'text': '<span style="color:#4ecdc4">Green = increases viability</span> | <span style="color:#ff6b6b">Red = decreases viability</span>',
                    'showarrow': False,
                    'font': {'size': 11, 'color': '#888888'}
                }
            ]
        }
    }

    return json.dumps(figure)


def estimate_ic50(curve_data):
    """
    Estimate IC50 from concentration-response curve data.

    Args:
        curve_data: List of (concentration, viability) tuples

    Returns:
        float or None: Estimated IC50 value, or None if cannot be determined
    """
    import numpy as np

    concentrations = np.array([point[0] for point in curve_data])
    viabilities = np.array([point[1] for point in curve_data])

    # Find where viability crosses 50%
    above_50 = viabilities >= 50
    below_50 = viabilities < 50

    if not any(above_50) or not any(below_50):
        return None

    # Find the transition point
    for i in range(len(viabilities) - 1):
        if viabilities[i] >= 50 and viabilities[i + 1] < 50:
            # Linear interpolation
            slope = (viabilities[i + 1] - viabilities[i]) / (concentrations[i + 1] - concentrations[i])
            if slope != 0:
                ic50 = concentrations[i] + (50 - viabilities[i]) / slope
                return float(ic50)

    return None
