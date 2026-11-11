"""Interactive HTML (plotly, optional dependency): load and price on a shared, zoomable time axis."""

from __future__ import annotations

from pathlib import Path

from .data import RunData
from .style import BLUE, CAP_LINE, PINK, WITH_CAP, WITHOUT_CAP


def write_html(d: RunData, path: str | Path) -> Path:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise SystemExit("--html needs plotly: pip install plotly (or pip install -e .[html])") from e
    l15, q = d.load_15, d.qh
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Station load, 15-min mean (kW)", "Price (EUR/MWh)"))
    fig.add_trace(go.Scattergl(x=l15.index, y=l15["load_uncapped_kw"].round(1), name="Load without cap",
                               line={"color": WITHOUT_CAP, "width": 1}), 1, 1)
    fig.add_trace(go.Scattergl(x=l15.index, y=l15["load_kw"].round(1), name="Load with cap",
                               line={"color": WITH_CAP, "width": 1}), 1, 1)
    fig.add_hline(y=d.cap_kw, line={"color": CAP_LINE, "dash": "dash", "width": 1}, row=1, col=1,
                  annotation_text=f"Grid cap {d.cap_kw:,.0f} kW")
    fig.add_trace(go.Scattergl(x=q.index, y=q["da_price"].round(2), name="DA price",
                               line={"color": BLUE, "width": 1, "shape": "hv"}), 2, 1)
    if d.has_id:
        fig.add_trace(go.Scattergl(x=q.index, y=q["id_price"].round(2), name="ID price",
                                   line={"color": PINK, "width": 1, "shape": "hv"}), 2, 1)
    tag = " (SYNTHETIC prices, not market data)" if d.synthetic else ""
    fig.update_layout(
        title=f"Run {d.run_id}: load (reference run {d.meta['reference_run']}) and prices{tag}",
        font={"family": "Arial", "size": 12}, template="plotly_white", hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"}, height=800)
    fig.update_yaxes(title_text="Grid load (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Price (EUR/MWh)", row=2, col=1)
    fig.update_xaxes(rangeselector={"buttons": [
        {"count": 1, "label": "1 day", "step": "day", "stepmode": "backward"},
        {"count": 7, "label": "1 week", "step": "day", "stepmode": "backward"},
        {"count": 1, "label": "1 month", "step": "month", "stepmode": "backward"},
        {"step": "all", "label": "Year"}]}, row=1, col=1)
    fig.update_xaxes(title_text="Local time", row=2, col=1)
    path = Path(path)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path
