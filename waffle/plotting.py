"""Plotly figure for the Substance / Focus / Actionability cube."""

from __future__ import annotations

from typing import Any


def build_cube_figure(substance: float, focus: float, actionability: float) -> Any:
    """Return a 3D scatter with a single point at (S, F, A)."""
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[substance],
                y=[focus],
                z=[actionability],
                mode="markers+text",
                text=["Your Text"],
                textposition="top right",
                marker=dict(size=6, color="crimson"),
                textfont=dict(color="black"),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            bgcolor="white",
            domain=dict(x=[0.02, 0.92], y=[0.02, 0.92]),
            xaxis=dict(
                range=[0, 1],
                backgroundcolor="white",
                gridcolor="#9a9a9a",
                gridwidth=2,
                zerolinecolor="#777777",
                zerolinewidth=2,
                title=dict(text="Meatiness Quotient", font=dict(color="black")),
                tickfont=dict(color="black"),
                showbackground=True,
            ),
            yaxis=dict(
                range=[0, 1],
                backgroundcolor="white",
                gridcolor="#9a9a9a",
                gridwidth=2,
                zerolinecolor="#777777",
                zerolinewidth=2,
                title=dict(text="Laser Aim", font=dict(color="black")),
                tickfont=dict(color="black"),
                showbackground=True,
            ),
            zaxis=dict(
                range=[0, 1],
                backgroundcolor="white",
                gridcolor="#9a9a9a",
                gridwidth=2,
                zerolinecolor="#777777",
                zerolinewidth=2,
                title=dict(text="Get‑Stuff‑Done Quotient", font=dict(color="black")),
                tickfont=dict(color="black"),
                showbackground=True,
            ),
        ),
        paper_bgcolor="white",
        font=dict(color="black"),
        height=520,
        margin=dict(l=90, r=90, t=70, b=90),
    )
    return fig
