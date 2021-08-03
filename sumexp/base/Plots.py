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
            Size of the confidence interval to draw when aggregating with an estimator. “sd” means to draw the standard deviation of the data. Setting to None will skip bootstrapping.
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



