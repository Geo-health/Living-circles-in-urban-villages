import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from concave_hull import concave_hull, concave_hull_indexes
# from scipy.spatial import Concavehull

# read csv
all_data = pd.read_csv('results.csv', header=0)

# group by ID1 
group_data = all_data.groupby("ID1")

# build a null GeoDataFrame
gdf = gpd.GeoDataFrame(columns=['ID1', 'geometry'])

# traverse each group
for name, group in group_data:
    lons, lats = [], []
    
    
    for _, row in group.iterrows():
        lons.append(row['lon1'])
        lats.append(row['lat1'])
    
    # calculate concave hull
    hull_points = concave_hull(list(zip(lons, lats)))
    if len(hull_points) < 4:
           continue
 
    
    # build a polygon object
    polygon = Polygon(hull_points)


    # add polygon object and ID1 to GeoDataFrame
    gdf = gdf.append({'ID1': name, 'geometry': polygon}, ignore_index=True)

# save GeoDataFrame as Shapefile
gdf.to_file('./livingcircle.shp')
