import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tempfile
from surfplot import Plot

def plot_brain(surf, data, labels=None, layout="row", views=["lateral", "medial"], clim_q=None, 
               cmap="viridis", cbar=False, cbar_label=None, cbar_kws=None, label_kws=None,
               outline=False, ax=None):
    """
    Plot brain surface data on a given surface mesh.

    Parameters
    ----------
    surf : str
        Path to the surface file.
    data : array-like
        Data to be plotted on the surface. Can be 1D or 2D.
    labels : list of str, optional
        List of labels for each subplot, by default None.
    layout : str, optional
        Layout of the subplots, either "row" or "col", by default "row".
    views : list of str, optional
        List of views to display, by default ["lateral", "medial"].
    clim_q : tuple of float, optional
        Percentile values to set the color limits, by default None.
    cmap : str, optional
        Colormap to use for the data, by default "viridis".
    cbar : bool, optional
        Whether to display a colorbar, by default False.
    cbar_label : str, optional
        Label for the colorbar, by default None.
    cbar_kws : dict, optional
        Additional keyword arguments for the colorbar, by default None.
    label_kws : dict, optional
        Additional keyword arguments for the labels, by default None.
    outline : bool, optional
        Whether to outline the data, by default False. Useful for parcellations.
    ax : matplotlib.axes.Axes or list of Axes, optional
        Axis or list of axes to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure or None
        The resulting figure if a new one is created, otherwise None.
    """
    
    cbar_kws_ = dict(pad=0.01, fontsize=15, shrink=1, decimals=2) if cbar_kws is None else cbar_kws.copy()
    label_kws_ = dict(fontsize=10) if label_kws is None else label_kws.copy()
    
    data = np.squeeze(data)
    if np.ndim(data) == 1 or np.shape(data)[1] == 1:
        data = data.reshape(-1, 1)
    
    n_data = np.shape(data)[1]
    
    # Create the figure and axes
    if ax is None:
        if layout == "row":
            fig, axs = plt.subplots(1, n_data, figsize=(len(views) * n_data * 1.5, 2))
        elif layout == "col":
            fig, axs = plt.subplots(n_data, 1, figsize=(3, n_data * 1.25))
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        axs = [axs] if n_data == 1 else axs.flatten()
    else:
        if isinstance(ax, list):
            if len(ax) != n_data:
                raise ValueError("Number of provided axes must match the number of brains to plot.")
            axs = ax
        else:
            if n_data > 1:
                raise ValueError("Multiple brains require a list of axes.")
            axs = [ax]
    
    # To plot multiple brains we need to save each figure to a temporary file then load it into the axes
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, ax in enumerate(axs):
            # Use surfplot to plot the data
            p = Plot(surf_lh=surf, views=views, size=(500, 250), zoom=1.25)
            color_range = [np.nanpercentile(data[:, i], clim_q[0]), np.nanpercentile(data[:, i], clim_q[1])] if clim_q else None
            p.add_layer(data=data[:, i], cmap=cmap, cbar=cbar, color_range=color_range)
            if outline:
                p.add_layer(data[:, i], as_outline=True, cmap="gray", cbar=False, color_range=(1, 2))
            temp_file = f"{temp_dir}/figure_{i}.png"
            fig = p.build(cbar_kws=cbar_kws_)
            if cbar:
                fig.get_axes()[1].set_xlabel(cbar_label, fontsize=cbar_kws_["fontsize"], labelpad=5)
            plt.close(fig)
            # Save the surfplot figure
            fig.savefig(temp_file, bbox_inches='tight')
            # Load the figure into the axes
            ax.imshow(plt.imread(temp_file))
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            # Plot labels
            if labels is not None:
                if layout == "row":
                    ax.set_title(labels[i], pad=0, fontsize=label_kws_["fontsize"])
                elif layout == "col":
                    ax.set_ylabel(labels[i], labelpad=0, rotation=0, ha="right", fontsize=label_kws_["fontsize"])
    
    return fig if ax is None else None


def plot_heatmap(data, ax=None, center=None, cmap="viridis", cbar=False, square=True):
    if ax is None:
        ax = plt.gca()
    vmin = np.min(data)
    vmax = np.max(data)

    cmap = plt.get_cmap(cmap)
    if center is not None:
        # Compute a symmetric range around center
        vrange = max(vmax - center, center - vmin)
        norm = mpl.colors.Normalize(vmin=center - vrange, vmax=center + vrange)

        # Remap the colormap to ensure center=0 is white
        cmin, cmax = norm([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap = mpl.colors.ListedColormap(cmap(cc))

    # Plot heatmap with colorbar
    mesh = ax.pcolormesh(data, cmap=cmap, **{"vmin": vmin, "vmax": vmax})

    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()

    # Create a colorbar with the same height as the heatmap
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)  # Adjust size and padding
        cb = plt.colorbar(mesh, cax=cax)

    # Set frame around heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.set_xticks([])
    ax.set_yticks([])
    if square:
        ax.set_aspect("equal")

    return ax

