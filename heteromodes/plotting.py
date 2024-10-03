import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tempfile
from surfplot import Plot

def plot_brain(surf, data, labels=None, layout="row", views=["lateral", "medial"], clim_q=None, 
               cmap="viridis", cbar=False, cbar_label=None, cbar_kws=None, outline=False, dpi=100):
    """Plot multiple surfaces with associated data.

    Parameters
    ----------
    surf : object
        The brain surface object.
    data : numpy.ndarray (n_verts, n_data)
        The data to be plotted on the surfaces. Medial wall indices should be set to NaN.
    labels : list, optional
        The labels for each surface, by default None.
    layout : str, optional
        The layout of the subplots, either "row" or "col", by default "row".
    views : list, optional
        The views of the brain surfaces to be plotted, by default ["lateral", "medial"].
    clim_q : list, optional
        The percentiles for the color range of the data, by default [5, 95].

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the subplots.
    """

    # Set dpi
    mpl.rcParams['figure.dpi'] = dpi

    # Set default colorbar keyword arguments and update with user-specified values
    cbar_kws_default = dict(pad=0.01, fontsize=15, shrink=1, decimals=2)
    cbar_kws_default.update(cbar_kws or {})

    # Check if the data is 1D or 2D
    data = np.squeeze(data)
    if np.ndim(data) == 1 or np.shape(data)[1] == 1:
        data = data.reshape(-1, 1)
        fig = plt.figure(figsize=(len(views)*1.5, 2.5))
        axs = [plt.gca()]
    else:
        if layout == "row":
            fig, axs = plt.subplots(1, np.shape(data)[1], figsize=(len(views)*np.shape(data)[1]*1.5, 2))
        elif layout == "col":
            fig, axs = plt.subplots(np.shape(data)[1], 1, figsize=(3, np.shape(data)[1]*1.25))
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        axs = axs.flatten()

    # Create a temporary directory to save the figures
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, ax in enumerate(axs):

            # Plot brain surface
            p = Plot(surf_lh=surf, views=views, size=(500, 250), zoom=1.25)
            if clim_q is not None:
                color_range = [np.nanpercentile(data[:, i], clim_q[0]), np.nanpercentile(data[:, i], clim_q[1])]
            else:
                color_range = None
            p.add_layer(data=data[:, i], cmap=cmap, cbar=cbar, color_range=color_range)

            if outline:
                p.add_layer(data[:, i], as_outline=True, cmap="gray", cbar=False, color_range=(1, 2))

            # Save the plot as a temporary file
            temp_file = f"{temp_dir}/figure_{i}.png"
            fig = p.build(cbar_kws=cbar_kws_default)
            if cbar:
                # Plot cbar label underneath the cbar
                fig.get_axes()[1].set_xlabel(cbar_label, fontsize=cbar_kws_default["fontsize"], 
                                             labelpad=5)
            plt.close(fig)  # Close the figure to avoid automatically displaying it
            fig.savefig(temp_file, bbox_inches='tight', dpi=dpi)

            # Load the figure and plot it as a subplot
            ax.imshow(plt.imread(temp_file))
            # Remove axes and ticklabels
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Plot labels
            if labels is not None:
                if layout == "row":
                    ax.set_title(labels[i], pad=0)
                elif layout == "col":
                    ax.set_ylabel(labels[i], labelpad=0, rotation=0, ha="right")

    return fig
