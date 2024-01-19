import matplotlib.pyplot as plt


def plot_vs(
    plot_x,
    plot_f1,
    plot_f2,
    title: str,
    plot_f3 =None,
    plot_f4 = None,
    plot_x2=None,
    plot_x3=None,
    plot_x4=None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    f1label: str | None = None,
    f2label: str | None = None,
    f3label: str | None = None,
    f4label: str | None = None,
    f1style: str = "-b",
    f2style: str = "-r",
    f3style: str = "-g",
    f4style: str = "-m",
    xticks=None,
    yticks=None,
    show=True,
    filename: str | None = None,
):
    """Plots 2-4 functions on the same plot

    Parameters
    ----------
    - `plot_x` - x-axis values
    - `plot_f1` - y-axis values for the 1st function
    - `plot_f2` - y-axis values for the 2nd function
    - `title`  - Title of the plot
    - 'plot_f3' (optional) - y-axis values for the 3rd function (defaut: None)
    - 'plot_f4' (optional) - y-axis values for the 4th function (defaut: None)
    - `xlabel` - Label of the x-axis
    - `ylabel` - Label of the y-axis
    - `f1label` - Label of the 1st function
    - `f2label` - Label of the 2nd function
    - `f3label` - Label of the 3rd function
    - `f4label` - Label of the 4th function
    - `f1_style` (optional) - style of the 1st function, (default: "-b")
    - `f2_style` (optional) - style of the 2nd function, (default: "-r")
    - `f3_style` (optional) - style of the 3rd function, (default: "-g")
    - `f4_style` (optional) - style of the 4th function, (default: "-m")
    - `xticks` (optional) - list of ticks to display on x-axis, (default: None)
    - `yticks` (optional) - list of ticks to display on y-axis, (default: None)
    - `show` (optional) - whether to show the plot (useful to add more thins before showing), (default: True)
    - `filename` (optional) - path to the file to save the plot, (default: None)
    """
    plt.figure(figsize=(15, 11))
    if title:
        plt.title(title)
    plt.grid()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.plot(plot_x, plot_f1, f1style, label=f1label, linewidth=1)
    plt.plot(plot_x if plot_x2 is None else plot_x2, plot_f2, f2style, label=f2label, linewidth=1, alpha=1)
    if plot_f3 is not None:
        plt.plot(plot_x if plot_x3 is None else plot_x3, plot_f3, f3style, label=f3label, linewidth=1, alpha=1)
    if plot_f4 is not None:
        plt.plot(plot_x if plot_x4 is None else plot_x4, plot_f4, f4style, label=f4label, linewidth=1, alpha=1)
    plt.legend(prop={"size": 10})
    plt.legend(fontsize=10)
    plt.legend()
    plt.tight_layout(pad=3)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()


def plot_solo(
    plot_x,
    plot_f,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    flabel: str | None = None,
    fstyle: str = "-b",
    xticks=None,
    yticks=None,
    filename: str | None = None,
):
    """Plots a single function

    Parameters
    ----------
    - `plot_x` - x-axis values
    - `plot_f` - y-axis values (i.e. values of the function)
    - `title`  - Title of the plot
    - `xlabel` - Label of the x-axis
    - `ylabel` - Label of the y-axis
    - `flabel` - Label of the function
    - `fstyle` (optional) - style of the function, (default: "-b")
    - `xticks` (optional) - list of ticks to display on x-axis, (default: None)
    - `yticks` (optional) - list of ticks to display on y-axis, (default: None)
    - `filename` (optional) - path to the file to save the plot, (default: None)
    """

    # plt.tight_layout(pad=5)
    plt.figure(figsize=(15, 10))
    plt.title(title)
    plt.grid()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.plot(plot_x, plot_f, fstyle, label=flabel, linewidth=1)
    plt.tight_layout(pad=2)
    if filename:
        plt.savefig(filename)
    plt.show()
