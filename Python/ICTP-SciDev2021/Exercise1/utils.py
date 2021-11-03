"""Module file for exercise 1 for the ICTP SciDev2021 workshop.
For the problem description refere to:
https://corbetta.phys.tue.nl/pages/ictp-l21-121.html

TODO:
    * Fix type hinting for numpy arrays and objects, and 
        for functions that may raise an exception.
"""
from typing import Union, Tuple, NoReturn
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime
from time import time
import numpy as np
import argparse

def print_info(data: Union[np.ndarray, np.memmap], loadtime: float) -> None:
    """Function to print some useful information about the data.

    Args:
        data (np.ndarray | np.memmap): The data to print information about.
            Should be a numpy 2D array of either numpy.ndarray or numpy.memmap 
            object type, depending on the data loading method.
        loadtime (float): Loading time of the data.

    """
    infos = [("data:\n", data), 
             ("data.shape:", data.shape),
             ("data.min:", np.min(data)),
             ("data.max:", np.max(data)),
             ("data.nbytes:", data.nbytes),
             ("data.time_to_load[s]:", loadtime)]
    for info in infos:
        print(info[0]+" {}".format(info[1]))


def load_data(filename: str, lazyload: bool = True, printinfo: bool = False) -> Union[np.ndarray, np.memmap]:
    """Function to load the data.

    Args:
        lazyload (bool, optional): Default to True. If True function will use 
            numpy's `memmap` method to load the data from disk to RAM on demand. 
            If False function will use numpy's `fromfile` method to load 
            the whole data to RAM.
        printinfo (bool, optional): Default to False. 
            If True will print summary information about the data.

    Returns:
        A 2D array of np.memmap object if lazyload is set, 
            otherwise a 2D array of np.ndarray type.
    """
    if lazyload:
        # data have to have 43200 pixels with 21600 lines
        start = time()
        data = np.memmap(filename, dtype="uint8", mode='r', shape=(21600,43200))
        end = time()
    else:
        start = time()
        data = np.fromfile(filename, dtype="uint8")
        end = time()
        data = data.reshape(21600, 43200)
    if printinfo:
        loadtime = np.round(end-start, 2)
        print_info(data, loadtime)

    return data


def plot_data(data: Union[np.ndarray, np.memmap], colormap: str = "default", detailed: bool = False):
    """Plot the data with the default or provided colormap.

    As the data is large and possibly won't fit in the RAM, 
    we only plot a sparse data or we can plot a detailed image 
    but for a limited range of the map.

    Args:
        data (np.ndarray | np.memmap): The data to plot.
            Should be a numpy 2D array of either numpy.ndarray or 
            numpy.memmap object type, depending on the data loading method.
        colormap (str): Colormap of the image. 
            Could be either of the 'default' or 'maryland' type.
        detailed (bool): If true will plot a limited but detailed image.
            Defaults to False.

    Returns:
        matplotlib.image.AxeImage: Matplotlibs AxeImage object.

    """
    if detailed:
        plot_data = data[3000:6000, 20000:23000]
    else:
        plot_data = data[::50, ::50]
    # matplotlib expects RGB numbers to be between 0 and 1
    maryland_colors = np.array([( 68,  79, 137),
                                (  1, 100,   0),
                                (  1, 130,   0),
                                (151, 191,  71),
                                (  2, 220,   0),
                                (  0, 255,   0),
                                (146, 174,  47),
                                (220, 206,   0),
                                (255, 173,   0),
                                (255, 251, 195),
                                (140,  72,   9),
                                (247, 165, 255),
                                (255, 199, 174),
                                (  0, 255, 255),]) / 255
    maryland_cmap = colors.ListedColormap(maryland_colors)
    if colormap == "default":
        image = plt.imshow(plot_data)
    elif colormap == "maryland":
        image = plt.imshow(plot_data, cmap=maryland_cmap)
    else:
        raise SyntaxError("Colormap should be either 'default' or 'maryland'!")
    plt.show()

    return image


def get_index(lat: float, lon:float) -> Tuple[float, float]:
    """Given latitude and longitude return the pixel index in the map matrix.
    
    Latitude ranges from -180 to +180 degrees and Longitude ranges from -90 
    to +90. Based on the coordinate convention of the map, this function 
    returns the index of the pixel of the coordinate.

    Args:
        lat (float): Latitude provided by the user input.
        lon (float): Longitude provided by the user input.

    Returns:
        tuple: First element is axis 0 and the second is the axis 1 
            of the pixel matrix.

    """
    check_args(lat, lon)
    lat_diff =  90 - lat
    lon_diff = 180 + lon
    map_res = 0.00833
    lat_index = np.int(np.ceil(lat_diff / map_res))
    lon_index = np.int(np.ceil(lon_diff / map_res))

    return (lat_index, lon_index)

def get_args() -> Tuple[float, float]:
    """Get arguments with argparse module.

    Note that this function will only catch positional arguments, 
    and not from STDIN.

    Returns:
        tuple: First element is the Latitude and the second is Longitude.
        
    """
    parser = argparse.ArgumentParser(description='Determine if the provided \
            coordinate is on land or on water.')
    parser.add_argument("lat",  type=float, nargs='?', default=None, \
            help="Latitude in degrees. Range in -180 to +180.")
    parser.add_argument("lon",  type=float, nargs='?', default=None, \
            help="Longitude in degrees. Range in -90 to +90.")
    args = parser.parse_args()
    lat = args.lat
    lon = args.lon

    return (lat, lon)

def check_args(lat: float, lon: float) -> NoReturn:
    """Checks if the user provided coordinates fit in the map.

    Note that, for now there is no support for over the range, 
    coordinates provided in degrees, if provided the program 
    will raise an exception.

    Raises:
        ValueError: If either of the coordinates is out of range.\

    """
    if lat < -180. or lat > +180.:
        raise ValueError("Latitude is out of range.\n\
                See usage: ./landcover --help")
    if lon < -90. or lon > +90.:
        raise ValueError("Longitude is out of range.\n\
                See usage: ./landcover --help")


def land_or_water(data, lat: float, lon: float) -> None:
    """Check if the provided coordinate is on water or on land.

    Args:
        lat (float): Latitude of the coordinate.
        lon (float): Longitude of the coordinate.

    """
    lat_index, lon_index = get_index(lat, lon)
    if data[lat_index][lon_index] == 0:
        print("Water")
    else:
        print("Land")
 

def load_earthquakes(earthquakefile: str) -> np.array: 
    """Loads earthquake file, containing coordinates of several earthquakes.

    Args:
        earthquakefile (str): Name of the file for earthquake coordinates.

    Returns:
        np.array: An array of tuples, first element is the time of the 
            earthquake, the second is the latitude and the third is the longitude.

    """
    str2date = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    data = np.genfromtxt(earthquakefile, dtype=None, comments='#', \
            delimiter=';', skip_header=True, converters={0:str2date}, \
            names=("date", "lat", "lon", "depth", "mag", "src"), \
            usecols=(0, 1, 2, 3, 4, 5), encoding="latin-1")
    date_lat_lon = np.array((data["date"], data["lat"], data["lon"])).T

    return date_lat_lon


