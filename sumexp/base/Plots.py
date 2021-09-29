import six
import seaborn
import numpy as np
import matplotlib.pyplot as plt


def lineplot(
        dataContainer,
        xitem,
        yitem,
        custom_operator_x=lambda x: x,
        custom_operator_y=lambda y: y,
        ci=None,
        ax=None,
        *args, **kwargs
        ):
    """line plot

    Parameters
    ----------
    dataContainer : Database or Dataset
    xitem : str
        item of x-axis
    yitem : str
        item of y-axis
    custom_operator_x : func
        xdata is converted to custome_operator(x)
    custom_operator_y : func
        ydata is converted to custome_operator(y)
    ci : int or “sd” or None
        Size of the confidence interval to draw when aggregating with an estimator.
        "sd" means to draw the standard deviation of the data.
        Setting to None will skip bootstrapping.
    ax : matplotlib.axes._subplots.AxesSubplot
    other arguments for matplotlib.plot e.g. linestyle, color, label, linewidth

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
    """
    if ax is None:
        fig, ax = plt.subplots()

    if ci is None:
        X = custom_operator_x( dataContainer[xitem] )
        Y = custom_operator_y( dataContainer[yitem] )
        line = ax.plot(X, Y, *args, **kwargs)
        return ax
    else:
        X, Y = list(), list()
        for x, ys in dataContainer.iterItems([xitem, yitem]):
            X += [x] * len(ys)
            Y += ys
        X = custom_operator_x(X)
        Y = custom_operator_y(Y)
        seaborn.lineplot(x=X, y=Y, ci=ci, ax=ax, *args, **kwargs)
        return ax


def scatterplot(
        dataContainer,
        xitem,
        yitem,
        custom_operator_x=lambda x: x,
        custom_operator_y=lambda y: y,
        ax=None,
        *args, **kwargs
        ):
    """scatter plot

    Parameters
    ----------
    dataContainer : Database or Dataset
    xitem : str
        item of x-axis
    yitem : str
        item of y-axis
    custom_operator_x : func
        xdata is converted to custome_operator(x)
    custom_operator_y : func
        ydata is converted to custome_operator(y)
    ax : matplotlib.axes._subplots.AxesSubplot

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
    """
    if ax is None:
        fig, ax = plt.subplots()

    X, Y = list(), list()
    for x, y in dataContainer.iterItems([xitem, yitem]):
        X.append(x)
        Y.append(y)
    X = custom_operator_x(X)
    Y = custom_operator_y(Y)
    paths = ax.scatter(X, Y, *args, **kwargs)
    return ax


def histplot(
        dataContainer,
        item,
        ax=None,
        *args, **kwargs
        ):
    """create histgram

    Parameters
    ----------
    dataContainer : Database or Dataset
    item : str
        item name
    ax : matplotlib.axes._subplots.AxesSubplot

    See Also
    --------
    https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.hist.html
    """
    if ax is None:
        fig, ax = plt.subplots()

    items = list(dataContainer.iterItems(item))
    ax.hist(items, *args, **kwargs)
    return ax


def render_mpl_table(
        data,
        col_width=3.0,
        row_height=0.625,
        font_size=14,
        header_color='#40466e',
        row_colors=['#f1f1f2', 'w'],
        Edge_color='w',
        bbox=[0, 0, 1, 1],
        header_columns=0,
        **kwargs
        ):
    """
    Parameters
    ----------
    data : Pandas.DataFrame
    """
    figsize = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    mpl_table = ax.table(
        cellText=data.to_numpy(),
        bbox=bbox,
        colLabels=data.columns,
        **kwargs
        )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(Edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return fig, ax

