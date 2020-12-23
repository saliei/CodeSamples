#! /usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.font_manager as fg
from mpl_toolkits.axes_grid.inset_locator import inset_axes

fname="Times New Roman 400.ttf"
mpl.rcParams["hatch.linewidth"]=0.2
mpl.rcParams["hatch.color"]='k'
plt.rc("text", usetex=True)
fpath = os.path.join(mpl.rcParams["datapath"], "/home/saliei/.local/share/fonts/{}".format(fname))
font = fg.FontProperties(fname=fpath)

plot_types={
        "knl": {"data": ["srctimes-KNL-1MPI-256THREADS.dat"],
                "title": r"Poiseuille benchmark - Intel\textsuperscript\textregistered Knights Landing(KNL), 1MPI/256Threads", 
                "xlabel": "Data Structures",
                "figsize": (6.4, 5),
                "l1loc": "upper right",
                "cmap": "BrBG",
                "cmap_range": (0.1, 0.9),
                "width": 1.3,
                "xlabelrotation": 45,
                "xlabelfontsize": 5
                } ,
        "data_structures": {"data": ["srctimes-256-AVGxCall-MARCONI100-1MPI-v100.dat"],
                            "title": r"Poiseuille Benchmark - Tesla-V100-SXM2-16GB",
                            "xlabel": "Data Structures",
                            "figsize": (8.2, 5.2),
                            "l1loc": "upper right",
                            "cmap": "BrBG",
                            "cmap_range": (0.1, 0.9),
                            "width": 1.3,
                            "xlabelrotation": 45,
                            "xlabelfontsize": 5
                            },
        "strong_poiseuille_512": {"data": ["srctimes-AVGxCall-poiseuille-strong-512.dat"],
                                  "title": r"Poiseuille, Strong scaling 512 cubic - 4xTesla-V100-SXM2-16GB",
                                  "xlabel": "Number of Nodes",
                                  "figsize": (7.1, 4.2),
                                  "l1loc": "upper left",
                                  "l1bbox": (0.15, 0.99),
                                  "cmap": "Greys",
                                  "cmap_range": (0.2, 0.5),
                                  "width": 1.6,
                                  "strong_eff": True,
                                  "effbox_size": ("45%", "40%")
                                  },
        "strong_poiseuille_256": {"data": ["srctimes-AVGxCall-poiseuille-strong-256.dat"],
                                  "title": r"Poiseuille, Strong scaling 256 cubic - 4xTesla-V100-SXM2-16GB",
                                  "xlabel": "Number of GPUs",
                                  "figsize": (7.1, 4.2),
                                  "l1loc": "upper left",
                                  "l1bbox": (0.15, 0.99),
                                  "cmap": "Greys",
                                  "cmap_range": (0.2, 0.5),
                                  "width": 1.6,
                                  "strong_eff": True,
                                  "effbox_size": ("45%", "40%")
                                  },
        "speedup_cpu_gpu": {"data": ["srctimes-poiseuille-move-collide-cpu-gpu-256.dat"],
                            "title": r"Average call time of best version of $\texttt{move\_collide\_fused}$ kernel for different architectures with 256 cubic lattice",
                            "xlabel": "Architectures",
                            "figsize": (11, 5),
                            "l1loc": "upper right",
                            "l1bbox": (0.87, 0.97),
                            "cmap": "Greys",
                            "cmap_range": (0.2, 0.5),
                            "width": 1.6
                            },
        "multi_component_opt_non_opt": {"data": ["srctimes-AVGxCall-multi-component-256-non-opt.dat", "srctimes-AVGxCall-multi-component-256-opt.dat"],
                                        "title": r"Multi-component 256 cubic, Optimizations - 4xTesla-V100-SXM2-16GB",
                                        "xlabel": "Number of Nodes",
                                        "labels": ["Non-Opt", "Opt"],
                                        "figsize": (7, 5),
                                        "l1loc": "upper right",
                                        "l2loc": "upper center",
                                        "cmap": "BrBG",
                                        "cmap_range": (0.1, 0.9),
                                        "width": 1.3,
                                        },
        "strong_mulit_component_256": {"data": ["srctimes-AVGxCall-multi-component-strong-256.dat"],
                                       "title": "Strong Scaling, Multi-component with 256 cubic - 4xTesla-V100-SXM2-16GB",
                                       "xlabel": "Number of Nodes",
                                       "figsize": (8.4, 4.7),
                                       "l1loc": "upper left",
                                       "l1bbox": (0.1, 0.99),
                                       "cmap": "BrBG",
                                       "cmap_range": (0.6, 0.9),
                                       "width": 1.1,
                                       "strong_eff": True,
                                       "effbox_size": ("47%", "50%")
                                        },
        "strong_mulit_component_512": {"data": ["srctimes-AVGxCall-multi-component-strong-512.dat"],
                                       "title": "Strong Scaling, Multi-component with 512 cubic - 4xTesla-V100-SXM2-16GB",
                                       "xlabel": "Number of Nodes",
                                       "figsize": (9, 4.7),
                                       "l1loc": "upper left",
                                       "l1bbox": (0.20, 0.99),
                                       "cmap": "BrBG",
                                       "cmap_range": (0.6, 0.9),
                                       "width": 1.5,
                                       "strong_eff": True,
                                       "effbox_size": ("50%", "50%")
                                        },
        "weak_multi_component": {"data": ["srctimes-AVGxCall-multi-component-weak.dat"],
                                 "title": "Weak Scaling, Mulit-Component - 4xTesla-V100-SXM2-16GB",
                                 "xlabel": "Number of Nodes",
                                 "figsize": (7, 5.5),
                                 "l1loc": "upper left",
                                 "cmap": "BrBG",
                                 "cmap_range": (0.6, 0.9),
                                 "width": 1.5,
                                 "weak_eff": True,
                                 "effbox_size": ("45%", "30%")
                                 },
        "m100_mn4": {"data": ["srctimes-AVGxCall-M100-MN4.dat"],
                     "title": "Multi-Component 512,1024 cubic M100 vs. MN4 - 4xTesla-V100-SXM2-16GB, 2x24-cores Intel Xeon 8160",
                     "xlabel": "Number of Nodes",
                     "labels": ["MN4", "M100"],
                     "figsize": (9, 6),
                     "l1loc": "upper left",
                     "l1ncol": 2,
                     "l2loc": "upper right",
                     "cmap": "BrBG",
                     "cmap_range": (0.6, 0.9),
                     "width": 1.1
                    },
        }


def load_table(tables, index):
    dfs = []
    for table in tables:
        df = pd.read_table(table, delim_whitespace=True, na_values="-")
        df.set_index(index, inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        dfs.append(df)
    return dfs

def colormap(cmap, length=(0, 1), n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=length[0], b=length[1]),
        cmap(np.linspace(length[0], length[1], n)))
    return new_cmap

def plot_clustered_stacked(all_df, plot_type, plot_props, cmap="BrBG"):
    """Plot a clusteres stacked graph.

    Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.

    Parameters:
        all_df(Pandas.DataFrame): a list of pandas.DataFrame objects with the exact same indices and columns.
        plot_type(str): type of the plot in plot_types dict.
        plot_props(dict): a dict of properties for each plot type.
        cmap(matplotlib.colormap): colormap to be used.

    Return:
        matplotlib.pyplot.axes: axis object of the figure.
    """

    fig, axe = plt.subplots(1, 1, figsize=plot_props["figsize"])
    n_df  = len(all_df)
    n_col = len(all_df[0].columns)
    n_ind = len(all_df[0].index)

    for df in all_df : 
        axe = df.plot(kind="bar",
                linewidth=0,
                stacked=True,
                ax=axe,
                legend=False,
                grid=False,
                colormap=cmap)


        h, l = axe.get_legend_handles_labels() 
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch("x" * int(2*i / n_col)) 
                rect.set_width(1 / (plot_props["width"]*float(n_df + 1)))

    if "xlabelfontsize" in plot_props:
        if "xlabelrotation" in plot_props:
            rot = plot_props["xlabelrotation"]
        else:
            rot = 0
        axe.set_xticklabels(df.index, rotation = rot, fontsize=plot_props["xlabelfontsize"])
    else:
        if "xlabelrotation" in plot_props:
            rot = plot_props["xlabelrotation"]
        else:
            rot = 0
        axe.set_xticklabels(df.index, rotation = rot)
    axe.set_xlabel(plot_props["xlabel"], fontproperties=font)
    axe.set_title(plot_props["title"], fontproperties=font)
    axe.set_ylabel("AVG-Call[s]", fontproperties=font)


    if "l1bbox" in plot_props:
        if "l1ncol" in plot_props:
            l1 = axe.legend(h[:n_col], l[:n_col], loc=plot_props["l1loc"], bbox_to_anchor=plot_props["l1bbox"] ,ncol=plot_props["l2ncol"], prop=font, frameon=False)
        else:
            l1 = axe.legend(h[:n_col], l[:n_col], loc=plot_props["l1loc"], bbox_to_anchor=plot_props["l1bbox"] ,ncol=1, prop=font, frameon=False)
    else:
        if "l1ncol" in plot_props:
            l1 = axe.legend(h[:n_col], l[:n_col], loc=plot_props["l1loc"], ncol=plot_props["l1ncol"], prop=font, frameon=False)
        else:
            l1 = axe.legend(h[:n_col], l[:n_col], loc=plot_props["l1loc"] ,ncol=1, prop=font, frameon=False)

    if "labels" in plot_props:
        n=[]
        for i in range(len(plot_props["labels"])):
            n.append(axe.bar(0, 0, color="gray", hatch="x" * i))
        l2 = plt.legend(n, plot_props["labels"], loc=plot_props["l2loc"], ncol=1, prop=font, frameon=False)
        axe.add_artist(l1)


    if "strong_eff" in plot_props or "weak_eff" in plot_props:
        effbox_size = plot_props["effbox_size"]
        zoom_axe = inset_axes(axe, effbox_size[0], effbox_size[1], "upper right", borderpad=2)
        tot_per_node = df.sum(axis=1)
        if plot_type == "strong_mulit_component_512":
            fac = 4
        else:
            fac = 1
        if "strong_eff" in plot_props and plot_props["strong_eff"] == True:
            eff = tot_per_node.iloc[0] / ((df.index.values / fac) * tot_per_node) * 100
        else:
            eff = tot_per_node.iloc[0] / tot_per_node
        eff = eff.to_numpy()
        idx = df.index.values
        zoom_axe.plot(idx, eff,  marker="o", lw=1, ms=3, ls="-.")
        zoom_axe.set_title("Efficiency", fontproperties=font, fontsize=10)
        for e in eff:
            zoom_axe.axhline(e, ls="-.", lw=0.5, color="gray", alpha=0.3)
        for n in idx:
            zoom_axe.axvline(n, ls="-.", lw=0.5, color="gray", alpha=0.3)
        zoom_axe.set_xticks(idx)

    axe.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    if( n_df > 1 ):
        axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 0.45 / (plot_props["width"]*float(n_df + 1))) / 2.)
    
    if plot_type == "strong_poiseuille_512" or plot_type == "strong_poiseuille_256":
        axe.set_xticks((np.arange(0, 2 * n_ind, 2) - 0.70 / (plot_props["width"]*float(n_df + 1))) / 2.)
    
    if plot_type == "strong_mulit_component_512":
        axe.set_xticks((np.arange(0, 2 * n_ind, 2) - 0.45 / (plot_props["width"]*float(n_df + 1))) / 2.)
    
    if plot_type == "strong_mulit_component_256":
        axe.set_xticks(np.arange(0, n_ind))

    if plot_type == "data_structures":
        zoom_axe = inset_axes(axe, "45%", "60%", "upper left", borderpad=3)
        df[:6].plot(kind="bar",
                linewidth=0,
                stacked=True,
                ax=zoom_axe,
                legend=False,
                grid=False,
                colormap=cmap)
                
        zoom_axe.set_xlabel("")
        zoom_axe.set_ylabel("")
        zoom_axe.set_xticklabels(df[:6].index, fontsize=7)
   
    if plot_type == "speedup_cpu_gpu":
        colors = ["darkslategray", "dimgray","slategray" , "darkgray", "silver", "lightgray", "lightgray"]
        labels = [r"NVIDIA\textsuperscript\textregistered Tesla V100-SXM2-16GB", r"NVIDIA\textsuperscript\textregistered Tesla P100-SXM2-16GB",
                r"Intel\textsuperscript\textregistered Xeon\textsuperscript\textregistered Phi 7250 (Knights Landing) - 1MPI/64Threads",
                r"Intel\textsuperscript\textregistered Xeon\textsuperscript\textregistered 8160(Skylake-Platnium) - 1MPI/24Threads",
                r"Intel\textsuperscript\textregistered Xeon\textsuperscript\textregistered 6130(Skylake-Gold) - 1MPI/16Threads",
                r"Intel\textsuperscript\textregistered Xeon\textsuperscript\textregistered E5-2697(Broadwell) - 1MPI/18Threads"]
        for pa in axe.patches:
            pa.set_hatch("x")
        for idx, bar in enumerate(axe.patches):
            bar.set_color(colors[idx])
            if idx < 6:
                bar.set_label(labels[idx])
        axe.legend(loc=plot_props["l1loc"], frameon=False, bbox_to_anchor=plot_props["l1bbox"])
        avg = df.iloc[:,0].to_numpy()
        zoom_axe = inset_axes(axe, "35%", "30%", "upper left", borderpad=2)
        speed = avg / avg[0]
        zoom_axe.plot(df.index.values, speed, marker="o", ms=3, lw=1, ls="-.")
        zoom_axe.set_title("Speedup", fontproperties=font, fontsize=10)
        speed = np.round(speed, 2)
        for i in range(len(df)):
            zoom_axe.text(i-0.2, speed[i]+0.9, speed[i])
        zoom_axe.set_ylim(0, 24) 
        zoom_axe.set_xlim(-0.25, 5.5)
        zoom_axe.set_yticklabels("")

    if plot_type == "weak_multi_component":
        axe.set_ylim(0, 0.03)
        axe.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        txt = [r"$256^3$", r"$512^3$", r"$1024^3$"]
        x = [-0.27, 0.77, 1.77]
        y = df.sum(axis=1).to_list()
        for idx, tx in enumerate(txt):
            axe.text(x[idx]+0.08, y[idx]+0.001, tx)

    if plot_type == "m100_mn4":
        axe.set_xticks([-0.1, 0.4, 1.9, 2.4])
        axe.set_ylim(0, 0.07)
        axe.text(0.20, 0.050, r"$512^3$")
        axe.text(2.20, 0.050, r"$1024^3$")

        for idx, bar in enumerate(axe.patches):
            if idx % 2 != 0:
                bar.set_hatch("x" * 3)
                bar.set_x(bar.get_x() - 0.5)



    return axe
    

if __name__ == "__main__":
  
    for plot in plot_types:
        plot_props = plot_types[plot]

        cmap = plt.get_cmap(plot_props["cmap"])
        cmap = colormap(cmap, length=plot_props["cmap_range"])

        if plot == "speedup_cpu_gpu":
            index = "Arch"
        else:
            index = "Config/Function"
        dfs = load_table(plot_props["data"], index=index)
        axe = plot_clustered_stacked(dfs, 
                                     plot_type=plot, 
                                     plot_props=plot_types[plot],
                                     cmap=cmap)
        pltname = plot + ".eps"
        path = "./plots/{}".format(pltname)
        if not os.path.isdir("./plots"):
            os.mkdir("./plots")
        plt.savefig(path, dpi=90)

