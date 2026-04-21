import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


STANDARD_10_20 = {
    "AF3": [-30, 50, 30], "O2": [20, -90, 10],
    "F7": [-50, 30, 0], "P8": [60, -60, 0],
    "F3": [-40, 30, 40], "T8": [70, -20, 0],
    "FC5": [-60, 0, 30], "FC6": [60, 0, 30],
    "T7": [-70, -20, 0], "F4": [40, 30, 40],
    "P7": [-60, -60, 0], "F8": [50, 30, 0],
    "O1": [-20, -90, 10], "AF4": [30, 50, 30],
}


def raw_and_spike_figure(raw_signal, spike_signal, feature_name, sample_index):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 3.5),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot(raw_signal, color="#1E90FF", linewidth=1.0, label="Raw Input")
    ax1.set_ylabel("Signal Values")
    ax1.set_title(f"Feature {feature_name} (Trial {sample_index})")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="upper right")

    _, stemlines, baseline = ax2.stem(
        np.arange(len(spike_signal)),
        spike_signal,
        linefmt="#1E90FF",
        markerfmt=" ",
        basefmt="gray",
    )
    plt.setp(stemlines, "linewidth", 1.2)
    plt.setp(baseline, "linewidth", 0.1, "alpha", 0.1)

    ax2.set_ylabel("Spike State")
    ax2.set_xlabel("Time Steps")
    ax2.set_yticks([])
    ax2.set_title("Output Spikes")
    plt.tight_layout()
    return fig


def feature_layout_figure(feature_names, clean_view=True):
    plot_coords = []
    plot_names = []
    eeg_matches = 0

    for raw_name in feature_names:
        clean_name = str(raw_name).replace("*", "")
        if clean_name in STANDARD_10_20:
            plot_coords.append(STANDARD_10_20[clean_name])
            plot_names.append(raw_name)
            eeg_matches += 1
        else:
            plot_coords.append([0, 0, 0])
            plot_names.append(raw_name)

    plot_coords = np.array(plot_coords)
    use_eeg_template = eeg_matches > 0

    if not use_eeg_template:
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
        plot_coords = np.column_stack([
            70 * np.cos(angles),
            70 * np.sin(angles),
            np.zeros(len(feature_names)),
        ])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=plot_coords[:, 0],
        y=plot_coords[:, 1],
        z=plot_coords[:, 2],
        mode="markers+text",
        text=plot_names,
        textposition="top center",
        textfont=dict(family="Arial Black", size=12, color="white"),
        marker=dict(size=12, color="#FF4B4B", opacity=1.0, line=dict(width=2, color="white")),
        name="Features",
    ))

    if use_eeg_template:
        phi = np.linspace(0, 2 * np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        phi, theta = np.meshgrid(phi, theta)
        radius = 90
        x_sphere = radius * np.sin(theta) * np.cos(phi)
        y_sphere = radius * np.sin(theta) * np.sin(phi)
        z_sphere = radius * np.cos(theta) - 10
        fig.add_trace(go.Mesh3d(
            x=x_sphere.flatten(),
            y=y_sphere.flatten(),
            z=z_sphere.flatten(),
            color="gray",
            opacity=0.2,
            name="Head Model",
            alphahull=0,
        ))

    grid_status = not clean_view
    axis_range = [-100, 100] if clean_view else None
    fig.update_layout(
        title="Schematic Head Model Visualisation" if use_eeg_template else "Generic Feature Layout",
        scene=dict(
            xaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title="X" if grid_status else ""),
            yaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title="Y" if grid_status else ""),
            zaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title="Z" if grid_status else ""),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
        showlegend=False,
    )
    return fig
