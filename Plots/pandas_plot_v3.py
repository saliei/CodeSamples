#! /usr/bin/env python3
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.font_manager as fg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import sys
import os

# plt.style.use(["science", "no-latex"])
# plt.style.use(["science", "no-latex"])
# plt.rc("font", family="Envy Code R")
# plt.rc("font", family="Helvetica")
# plt.rc("font", family="GARA")
mpl.rcParams["hatch.linewidth"]=0.2
mpl.rcParams["hatch.color"]='k'
plt.rc("text", usetex=True)


fname="Times New Roman 400.ttf"
fname1="Envy Code R.ttf"
fpath = os.path.join(mpl.rcParams["datapath"], "/home/saliei/.local/share/fonts/{}".format(fname))
fpath1 = os.path.join(mpl.rcParams["datapath"], "/home/saliei/.local/share/fonts/{}".format(fname1))
font = fg.FontProperties(fname=fpath)
font1 = fg.FontProperties(fname=fpath1)

# title = "KNL 1MPI/256Threads"
title = "Poiseuille, Strong scaling 512 cubic - 4xTesla-V100-SXM2-16GB"

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('BrBG')
new_cmap = truncate_colormap(cmap, 0.1, 0.4)

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
                colormap=new_cmap,
                **kwargs) 

        h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(2*i / n_col)) #edited part
                rect.set_width(1 / (1.3*float(n_df + 1)))

    # axe.set_xticks(np.arange(0, 2 * n_ind, 2))
    axe.set_xticks(np.arange(0, n_ind) - 0.05)
    # axe.set_xticks((np.arange(0, 2 * n_ind, 2) - 1.00 / (1.5*float(n_df + 1))) / 2.)
    if( n_df > 1 ):
        axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 0.45 / (1.3*float(n_df + 1))) / 2.)
    axe.set_xticklabels(df.index, rotation = 0, fontsize=8)
    # axe.set_yticklabels(fontproperties=font1)
    axe.set_title(title, fontproperties=font)
    axe.set_xlabel("Number of Nodes", fontproperties=font)
    axe.set_ylabel("AVG-Call[s]", fontproperties=font)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    # l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    # l1 = axe.legend(h[:n_col], l[:n_col], loc="upper left", bbox_to_anchor=(0.1, 0.99) ,ncol=1, prop=font, frameon=False)
    l1 = axe.legend(h[:n_col], l[:n_col], loc="upper left" ,ncol=1, prop=font, frameon=False)
    if labels is not None:
        # l2 = plt.legend(n, labels, loc=[1.01, 0.1])
        l2 = plt.legend(n, labels, loc="upper center", prop=font)
        axe.add_artist(l1)
        return axe
    axe.set_ylim(0, 0.03)
    axe.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    # for label in axe.get_yticklabels():
        # label.set_fontproperties(font)
    # base = axe.patches[0].get_y()
    # print(ys)
    # print(len(axe.patches))
    # eff = [bar.get_y()/ base for bar in axe.patches]
    # print(eff)
    zoom_axe = inset_axes(axe, "55%", "35%", "upper right", borderpad=2)
    tot_per_node = df.sum(axis=1)
    # strong scaling efficiency
    eff = tot_per_node.iloc[0] / (df.index.values * tot_per_node) * 100
    # weak scaling efficienct
    # eff = tot_per_node.iloc[0] / tot_per_node
    eff = eff.to_numpy()
    idx = df.index.values
    print(eff)
    zoom_axe.plot(idx, eff,  marker="o", lw=1, ms=3, ls="-.")
    # eff.plot(kind="line",
            # linewidth=0,
            # stacked=True,
            # ax=zoom_axe,
            # legend=False,
            # grid=False,
            # colormap="tab20",
            # **kwargs)
    # zoom_axe.set_xlabel("", fontproperties=font, fontsize)
    # zoom_axe.set_ylabel("Efficiency", fontproperties=font, fontsize=10)
    zoom_axe.set_title("Efficiency", fontproperties=font, fontsize=10)
    # zoom_axe.set_yticklabels(eff.to_numpy(), fontproperties=font)
    # zoom_axe.set_xticklabels(df.index.values, fontsize=5, fontproperties=font)
    # zoom_axe.grid(True, linestyle='-.')
    # zoom_axe.tick_params(axis="both", which="major", fontproperties=fon)
    for e in eff:
        zoom_axe.axhline(e, ls="-.", lw=0.5, color="gray", alpha=0.3)
    for n in idx:
        zoom_axe.axvline(n, ls="-.", lw=0.5, color="gray", alpha=0.3)
    zoom_axe.set_xticks(idx)
    # zoom_axe.set_yticks(np.floor(eff))
    # zoom_axe.set_yticks([0.9, 1, 1.1])
    # zoom_axe.set_ylim(0.8, 1.05)
    # zoom_axe.set_xscale("log")
    eff = []
    for tick in zoom_axe.xaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in zoom_axe.yaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    
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

    labels = ["NON-OPT"]
    dfs = [df1]

    if(len(sys.argv) > 2):
        df2 = pd.read_table(sys.argv[2], delim_whitespace=True, na_values="-")
        df2.set_index("Config/Function", inplace=True)
        df2.dropna(axis=1, how="all", inplace=True)
        dfs.append(df2)
        labels.append("OPT")


    if(len(dfs) < 2):
        labels = None
    axe = plot_clustered_stacked(dfs, labels)
    # axe = plot_clustered_stacked(dfs, None)
    plt.show()

