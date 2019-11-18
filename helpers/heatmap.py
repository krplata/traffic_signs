import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
c = [10000, 1000, 100, 10, 1, 0.1, 0.01]

# rbf_circles = np.array([
#     [51.10, 95.01, 97.28, 87.41, 54.78, 54.78, 54.78],
#     [95.09, 97.29, 98.92, 97.29, 54.78, 54.78, 54.78],
#     [97.25, 98.33, 99.36, 99.48, 54.88, 54.80, 54.80],
#     [98.13, 98.79, 99.63, 99.52, 54.95, 54.80, 54.80],
#     [98.47, 99.26, 99.63, 99.52, 54.95, 54.80, 54.80],
#     [98.65, 99.53, 99.63, 99.52, 54.95, 54.80, 54.80],
#     [99.11, 99.52, 99.63, 99.52, 54.95, 54.80, 54.80]])

# rbf_triangles = np.array([
#     [50.43, 86.04, 96.80, 86.76, 52.62, 52.62, 52.62],
#     [86.12, 96.20, 98.48, 97.62, 52.62, 52.62, 52.62],
#     [96.05, 97.36, 99.17, 99.24, 52.76, 52.63, 52.63],
#     [97.01, 98.06, 99.55, 99.29, 52.84, 52.63, 52.63],
#     [97.27, 98.98, 99.55, 99.29, 52.84, 52.63, 52.63],
#     [97.82, 99.35, 99.55, 99.29, 52.84, 52.63, 52.63],
#     [98.68, 99.35, 99.55, 99.29, 52.84, 52.63, 52.63]
# ])

degree = [2, 3, 4, 5, 6, 7]
# poly_circles = np.array([
#     [51.10, 51.10, 51.10, 51.10, 51.10, 51.10],
#     [51.10, 51.10, 51.10, 51.10, 51.10, 51.10],
#     [93.73, 51.10, 51.10, 51.10, 51.10, 51.10],
#     [97.27, 51.12, 51.10, 51.10, 51.10, 51.10],
#     [98.87, 95.68, 51.10, 51.10, 51.10, 51.10],
#     [99.28, 98.62, 91.22, 51.10, 51.10, 51.10],
#     [99.55, 99.28, 97.38, 51.10, 51.10, 51.10]])


poly_triangles = np.array([
    [50.13, 50.13, 50.13, 50.13, 50.13, 50.13],
    [50.13, 50.13, 50.13, 50.13, 50.13, 50.13],
    [75.90, 50.13, 50.13, 50.13, 50.13, 50.13],
    [96.66, 50.13, 50.13, 50.13, 50.13, 50.13],
    [98.35, 87.60, 50.13, 50.13, 50.13, 50.13],
    [99.01, 98.10, 53.40, 50.13, 50.13, 50.13],
    [99.43, 98.98, 97.02, 50.13, 50.13, 50.13]
])

# sigmoid_circles = np.array([
#     [51.10, 94.35, 96.34, 55.14, 54.81, 54.78, 54.78],
#     [94.35, 96.87, 96.97, 55.31, 54.82, 54.78, 54.78],
#     [96.87, 97.82, 94.04, 69.43, 58.22, 54.78, 54.78],
#     [97.83, 98.37, 92.69, 69.90, 74.26, 54.78, 54.78],
#     [98.39, 98.41, 92.51, 69.89, 74.47, 54.78, 54.78],
#     [98.50, 97.97, 92.49, 69.89, 74.45, 53.82, 54.78],
#     [98.45, 97.10, 92.49, 69.89, 74.45, 52.98, 54.78]
# ])

# sigmoid_triangles = np.array([
#     [50.06, 66.37, 88.99, 52.97, 52.63, 52.62, 52.62],
#     [66.53, 93.87, 93.06, 53.10, 52.64, 52.62, 52.62],
#     [93.91, 96.79, 88.40, 63.97, 57.38, 52.62, 52.62],
#     [96.81, 97.13, 86.33, 63.79, 61.08, 52.62, 52.62],
#     [97.17, 97.12, 85.95, 63.78, 63.01, 51.77, 52.62],
#     [97.33, 96.38, 85.91, 63.77, 62.95, 51.74, 52.62],
#     [97.31, 94.96, 85.88, 63.77, 62.95, 49.78, 52.62]
# ])


fig, ax = plt.subplots()
ax.set_title("Skuteczność klasyfikatora dla wielomianowej funkcji jądrowej.")
im, cbar = heatmap(poly_triangles, c, degree, ax=ax,
                   cmap=cm.Reds, cbarlabel="Skuteczność [%]")
texts = annotate_heatmap(im)

fig.tight_layout()
plt.xlabel('Stopień wielomianu')
plt.ylabel('c')
plt.show()
