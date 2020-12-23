#! /usr/bin/env python3
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fg
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.pyplot as plt
import sys
import os

# plt.style.use(["science", "no-latex"])
# plt.style.use(["science", "no-latex", "high-vis"])
# plt.rc("font", family="Envy Code R")
# plt.rc("font", family="Helvetica")
# plt.rc("font", family="GARA")
mpl.rcParams["hatch.linewidth"]=0.2
mpl.rcParams["hatch.color"]='k'
plt.rc("text", usetex=True)


fname="Times New Roman 400.ttf"
fpath = os.path.join(mpl.rcParams["datapath"], "/home/saliei/.local/share/fonts/{}".format(fname))
font = fg.FontProperties(fname=fpath)

title = "Poiseuille Benchmark - Tesla V100-SXM2-16GB"
def plot_clustered_stacked(dfall, labels=None, title=title,  H="x", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                linewidth=0,
                stacked=True,
                ax=axe,
                legend=False,
                grid=False,
                colormap="BrBG",
                **kwargs) 

        h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(2*i / n_col)) #edited part
                rect.set_width(1 / (1.1*float(n_df + 1)))

    # axe.set_xticks(np.arange(0, 2 * n_ind, 2))
    # axe.set_xticks(np.arange(0, n_ind))
    if( n_df > 1 ):
        axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 0.45 / (1.1*float(n_df + 1))) / 2.)
    axe.set_xticklabels(df.index, rotation = 0, fontsize=7)
    axe.set_title(title, fontproperties=font)
    axe.set_xlabel("Data Structures", fontproperties=font, fontsize=10)
    axe.set_ylabel("AVG-Call[s]", fontproperties=font, fontsize=10)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    # l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    l1 = axe.legend(h[:n_col], l[:n_col], loc="upper right", ncol=1, prop=font, frameon=False)
    if labels is not None:
        # l2 = plt.legend(n, labels, loc=[1.01, 0.1])
        l2 = plt.legend(n, labels, loc="best", prop=font)
    axe.add_artist(l1)
    zoom_axe = inset_axes(axe, "45%", "60%", "upper left", borderpad=3)
    df[:6].plot(kind="bar",
            linewidth=0,
            stacked=True,
            ax=zoom_axe,
            legend=False,
            grid=False,
            colormap="BrBG",
            **kwargs)
    zoom_axe.set_xlabel("")
    zoom_axe.set_ylabel("")
    zoom_axe.set_xticklabels(df[:6].index, fontsize=7)
    zoom_axe.ticklabel_format(style="sci", axis="y", scilimits=(0,0))


    return axe

def load_table(table):
    df = pd.read_table(table, delim_whitespace=True, na_values="-")
    df.set_index("Config/Function", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df


if __name__ == "__main__":
    # for table in sys.argv[1:]:
        # try:
            # df = load_df(table)
            # # dfs.append(df)
        # except IOError:
            # print("Could not open file: {}".format(table))
        # except:
            # print("Exception occured during opening file: {}".format(table))
    df1 = pd.read_table(sys.argv[1], delim_whitespace=True, na_values="-")
    df1.set_index("Config/Function", inplace=True)
    df1.dropna(axis=1, how="all", inplace=True)

    labels = ["2D"]
    dfs = [df1]

    if(len(sys.argv) > 2):
        df2 = pd.read_table(sys.argv[2], delim_whitespace=True, na_values="-")
        df2.set_index("Config/Function", inplace=True)
        df2.dropna(axis=1, how="all", inplace=True)
        # dfs.append(df2)
        # labels.append("3D")


    # axe = plot_clustered_stacked(dfs, labels)
    axe = plot_clustered_stacked(dfs, None)
    plt.show()

