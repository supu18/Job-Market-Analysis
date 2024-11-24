# utils/imports.py
"""this file contains all the necessary imports for the project"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import time
import geopandas as gpd
from mpl_toolkits.basemap import Basemap
import os
from matplotlib.widgets import CheckButtons
import pycountry
import seaborn as sns
from pywaffle import Waffle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from matplotlib.patches import Patch
import squarify
from matplotlib import cm
import calmap
from scipy.cluster.hierarchy import dendrogram, linkage
import calendar
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
from calendar import monthrange, month_name, day_abbr
import numpy as np
import mplcursors
from matplotlib.widgets import RadioButtons
from geopy.geocoders import Nominatim
