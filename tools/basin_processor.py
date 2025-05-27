# ======================================
# Basin Selection and Processing Functions
# ======================================

import requests
import geopandas as gpd
import folium
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import os
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.windows import from_bounds
from osgeo import gdal
import glob
import sys

# Configure GDAL environment
if sys.platform.startswith('win'):
    # For Windows
    gdal_data = os.path.join(os.path.dirname(sys.executable), 'Library', 'share', 'gdal')
    if os.path.exists(gdal_data):
        os.environ['GDAL_DATA'] = gdal_data
else:
    # For Linux/Mac
    gdal_data = os.path.join(os.path.dirname(sys.executable), 'share', 'gdal')
    if os.path.exists(gdal_data):
        os.environ['GDAL_DATA'] = gdal_data

def download_watershed_shp(latitude, longitude, output_path, level=5):
    """
    Download watershed boundary data for a given latitude and longitude.
    
    Parameters:
    -----------
    latitude : float
        Latitude of the point of interest
    longitude : float
        Longitude of the point of interest
    output_path : str
        Directory path where output files will be saved
    level : int, default=5
        WBD level to query
        HUC_LEVEL = {
            "huc2": 1,
            "huc4": 2,
            "huc6": 3,
            "huc8": 4,
            "huc10": 5,
            "huc12": 6
        }
    
    Returns:
    --------
    Basin_Area : float
        Area of the watershed in square kilometers
    """
    # Create output directory if it doesn't exist
    # If output_path is a file (e.g., ends with .shp), only create its parent directory
    dir_to_create = os.path.dirname(output_path) if output_path.lower().endswith('.shp') else output_path
    if dir_to_create:
        os.makedirs(dir_to_create, exist_ok=True)
    
    # Step 1: Query WBD API to find the watershed containing this point
    wbd_url = f"https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/{level}/query"
    params = {
        'geometry': f'{longitude},{latitude}', # Note: longitude comes first
        'geometryType': 'esriGeometryPoint',
        'inSR': '4326',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': '*',
        'f': 'geojson'
    }

    response = requests.get(wbd_url, params=params)
    data = response.json()

    # Step 2: Save as local GeoJSON and read with GeoPandas
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    # Set CRS to match the input data (WGS 84 / EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    gdf['shape_Area'] = gdf['shape_Area'] / 1000000
    print('Basin Area (km2):')
    Basin_Area = round(gdf['shape_Area'][0], 2)
    print(Basin_Area)

    # Rename columns to fit 10-character limit
    column_rename_dict = {
        'shape_Length': 'shp_length',
        'metasourceid': 'metasource',
        'sourcedatadesc': 'sourcedata',
        'sourceoriginator': 'sourceorig',
        'sourcefeatureid': 'sourcefeat',
        'referencegnis_ids': 'ref_gnis'
    }

    # Apply the renaming
    gdf = gdf.rename(columns=column_rename_dict)

    # Save to shapefile
    # shp_path = os.path.join(output_path, f'Basin_selected_{level}.shp')
    gdf.to_file(output_path)

    return Basin_Area


def plot_watershed_with_gauges(basin_shp_path, gauge_meta_path, figure_path):
    """
    Plot watershed with USGS gauge stations and save both interactive HTML and static PNG (300 dpi) to figure_path.

    Parameters:
    -----------
    basin_shp_path : str
        Path to the watershed shapefile
    gauge_meta_path : str
        Path to the USGS gauge metadata CSV file
    figure_path : str
        Directory path where output HTML and PNG files will be saved

    Returns:
    --------
    None
    """
    import os
    import pandas as pd

    # Ensure output directory exists
    os.makedirs(figure_path, exist_ok=True)

    # Load the watershed shapefile and reproject to Web Mercator for plotting
    gdf_web = gpd.read_file(basin_shp_path).to_crs(epsg=3857)

    # Calculate centroid in projected CRS, then convert to WGS84 for folium
    centroid = gdf_web.geometry.unary_union.centroid
    center_point = gpd.GeoDataFrame(geometry=[centroid], crs='EPSG:3857').to_crs(epsg=4326)
    center_lat = center_point.geometry.y[0]
    center_lng = center_point.geometry.x[0]

    # Create interactive folium map centered on the watershed
    m = folium.Map(location=[center_lat, center_lng], zoom_start=8)

    # Remove datetime columns if present (folium/GeoJSON can't handle them)
    gdf_web = gdf_web.drop(columns=gdf_web.select_dtypes(include=["datetime64[ns]"]).columns)

    # Add watershed boundary to the folium map
    folium.GeoJson(
        gdf_web.to_crs(epsg=4326),
        name='Watershed Boundary',
        style_function=lambda x: {'fillColor': 'yellow', 'color': 'red', 'weight': 2, 'fillOpacity': 0.5}
    ).add_to(m)

    # Load USGS gauge information
    gauge_info = pd.read_csv(gauge_meta_path)

    # Convert gauge locations to GeoDataFrame
    gauge_points = gpd.GeoDataFrame(
        gauge_info,
        geometry=gpd.points_from_xy(gauge_info.LNG_GAGE, gauge_info.LAT_GAGE),
        crs='EPSG:4326'
    )

    # Reproject gauge points to match watershed CRS (Web Mercator)
    gauge_points = gauge_points.to_crs(epsg=3857)

    # Spatial join to find gauges within the watershed
    gauges_in_basin = gpd.sjoin(gauge_points, gdf_web, how='inner', predicate='within')

    # Create a new matplotlib figure for the static map
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot watershed boundary
    gdf_web.plot(ax=ax, alpha=0.5, edgecolor='red', facecolor='yellow', linewidth=2)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Plot centroid as a red star
    point = Point(center_lng, center_lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs='EPSG:4326').to_crs(epsg=3857)
    point_gdf.plot(ax=ax, color='red', marker='*', markersize=100)

    # Set map extent to focus on the watershed
    ax.set_xlim(gdf_web.total_bounds[[0, 2]])
    ax.set_ylim(gdf_web.total_bounds[[1, 3]])

    # Remove axes for a clean map
    ax.set_axis_off()

    # File paths for output
    html_path = os.path.join(figure_path, 'basin_map_with_gauges.html')
    png_path = os.path.join(figure_path, 'basin_map_with_gauges.png')

    if len(gauges_in_basin) > 0:
        # Plot gauge locations on the static map
        gauges_in_basin.plot(
            ax=ax,
            color='blue',
            marker='^',
            markersize=100,
            label='USGS Gauges'
        )

        # Add station IDs as labels on the static map
        for idx, row in gauges_in_basin.iterrows():
            padded_staid = str(row['STAID']).zfill(8)
            ax.annotate(
                padded_staid,
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                color='blue',
                fontsize=10,
                fontweight='bold'
            )

        # Add gauge markers to the interactive map
        for idx, row in gauges_in_basin.iterrows():
            padded_staid = str(row['STAID']).zfill(8)
            folium.Marker(
                location=[row['LAT_GAGE'], row['LNG_GAGE']],
                popup=f"Station ID: {padded_staid}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

        plt.title('Watershed Boundary with USGS Gauges')
        ax.legend(loc='upper right')
        plt.tight_layout()

        # Save static PNG with 300 dpi
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save interactive HTML map
        m.save(html_path)

        # Print all station IDs and names
        print("\nUSGS Gauge Stations in the Watershed:")
        print("--------------------------------------")
        for idx, row in gauges_in_basin.iterrows():
            padded_staid = str(row['STAID']).zfill(8)
            print(f"Station ID: {padded_staid}, Name: {row['STANAME']}, Latitude: {row['LAT_GAGE']:.2f}, Longitude: {row['LNG_GAGE']:.2f}")

        print(f"Interactive map saved to {html_path}")
        print(f"Static PNG map saved to {png_path}")

    else:
        plt.title('Watershed Boundary (No USGS Gauges Found)')
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save the map even if no gauges found
        m.save(html_path)

        print("No USGS gauge stations found within the watershed boundary.")
        print(f"Interactive map saved to {html_path}")
        print(f"Static PNG map saved to {png_path}")

# ======================================
# HydroSHEDS Data Download and Processing
# ======================================

def download_hydrosheds_data(latitude, longitude, dest_folder="../BasicData"):
    """
    Download and process HydroSHEDS data based on coordinates.
    
    Parameters:
    -----------
    latitude : float
        Latitude of the gauge station
    longitude : float
        Longitude of the gauge station
    dest_folder : str, optional
        Destination folder for downloaded data (default: "../BasicData")
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    import os
    import requests
    from zipfile import ZipFile
    import glob
    import shutil
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")
    else:
        # Clear all files in the destination folder before downloading
        for file_path in glob.glob(os.path.join(dest_folder, '*')):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        # print(f"Cleared all existing files in {dest_folder}")
    
    # Determine the tile prefix based on coordinates
    def get_tile_prefix(lat, lon):
        # For latitude: N20 means 20N-30N
        lat_prefix = f"N{int(lat) // 10 * 10}"
        
        # For longitude: W100 means 100W-90W
        # We need absolute value and then round down to nearest 10
        lon_abs = abs(lon)
        # For longitude: W100 means 100W-90W
        # Need to round up to the nearest 10 for the correct tile
        lon_prefix = f"W{(int(lon_abs) // 10 + 1) * 10:03d}"
        
        return f"{lat_prefix}{lon_prefix}"
    
    # Get the tile prefix based on gauge coordinates
    tile_prefix = get_tile_prefix(latitude, longitude)
    # print(f"Determined tile prefix: {tile_prefix}")
    
    # Define the URLs for the three datasets
    urls = [
        f"https://data.hydrosheds.org/file/hydrosheds-v1-acc/na_acc_3s/{tile_prefix}_acc.zip",
        f"https://data.hydrosheds.org/file/hydrosheds-v1-con/na_con_3s/{tile_prefix.lower()}_con.zip",
        f"https://data.hydrosheds.org/file/hydrosheds-v1-dir/na_dir_3s/{tile_prefix.lower()}_dir.zip"
    ]
    
    # Download and process each file
    for url in urls:
        # print(f"Downloading from {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Extract the filename from the URL
            filename = url.split('/')[-1]
            filepath = os.path.join(dest_folder, filename)
            
            # Save the zip file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            # print(f"Downloaded {filename} to {dest_folder}")
            
            # Extract the contents
            with ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(dest_folder)
            # print(f"Extracted contents to {dest_folder}")
            
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            return False
    
    # Clean up by removing the zip files
    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_folder, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            # print(f"Removed zip file: {filepath}")
    
    # Rename the extracted files according to a more standardized naming convention
    # Find all extracted files in the destination folder
    extracted_files = [f for f in os.listdir(dest_folder) if f.endswith('.tif')]
    
    for file in extracted_files:
        # Determine the file type (acc, con, or dir)
        if '_acc' in file.lower():
            new_name = "facc.tif"
        elif '_con' in file.lower():
            new_name = "dem.tif"
        elif '_dir' in file.lower():
            new_name = "fdir.tif"
        else:
            continue  # Skip if not one of the expected file types
        
        # Create the full paths
        old_path = os.path.join(dest_folder, file)
        new_path = os.path.join(dest_folder, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        # print(f"Renamed: {file} → {new_name}")
    
    return True

def clip_tif_by_shapefile(tif_path, output_path, shp_path):
    """
    Clip a GeoTIFF file to the bounding box of a shapefile.
    
    Parameters:
    -----------
    tif_path : str
        Path to the input GeoTIFF file
    output_path : str
        Path where the clipped GeoTIFF will be saved
    shp_path : str
        Path to the shapefile
    

    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read the shapefile and get its bounding box
        gdf = gpd.read_file(shp_path)
        
        # Get the bounding box (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = gdf.total_bounds
        
        # Open the GeoTIFF file
        with rasterio.open(tif_path) as src:
            # Create a window from the bounding box
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            
            # Read the data within the window (all bands)
            data = src.read(window=window)
            
            # Calculate the new transform for the clipped raster
            out_transform = src.window_transform(window)
            
            # Copy the metadata and update with new dimensions and transform
            out_meta = src.meta.copy()
            out_meta.update({
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": out_transform,
                "dtype": 'float32',  # Ensure output is float32
                "compress": 'deflate'
            })
            
            # Convert data to float32 if it's not already
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Write the clipped data to a new GeoTIFF
            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(data)
                
        
    except Exception as e:
        tqdm.write(f"Error clipping {os.path.basename(tif_path)}: {str(e)[:100]}...")

def batch_clip_tifs_by_shapefile(input_dir='../BasicData', output_dir='../BasicData_Clip', shp_path='../shpFile/WBDHU12_CobbFort_sub2.shp'):
    """
    Batch process all TIF files in the input directory,
    clipping them to the bounding box of the specified shapefile.
    
    Parameters:
    -----------
    input_dir : str, optional
        Directory containing TIF files to clip
    output_dir : str, optional
        Directory to save clipped files
    shp_path : str, optional
        Path to the shapefile
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIF files in the input directory
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    # print(f"Clipping {len(tif_files)} TIF files to basin boundary defined in {os.path.basename(shp_path)}")
    # print(f"Input folder: {os.path.abspath(input_dir)}")
    # print(f"Output folder: {os.path.abspath(output_dir)}")
    # print(f"Shapefile: {os.path.abspath(shp_path)}")
    
    # Process each TIF file with a progress bar
    successful = 0
    
    with tqdm(total=len(tif_files), desc="Clipping TIF files") as pbar:
        for tif_file in tif_files:
            # Get the base filename
            base_name = os.path.basename(tif_file)
            if '_' in base_name:
                base_name = base_name.split('_', 1)[1]  # Split at first '_' and keep the second part
            # Add '_clip' suffix to the filename before the extension
            base_name_without_ext, ext = os.path.splitext(base_name)
            output_name = f"{base_name_without_ext}_clip{ext}"
            output_path = os.path.join(output_dir, output_name)
            
            # Clip the TIF file
            if clip_tif_by_shapefile(tif_file, output_path, shp_path):
                successful += 1
            
            pbar.update(1)
    
    print(f"Output files saved to {os.path.abspath(output_dir)}")

# Visualize the clipped data with basin boundary overlay
def visualize_clipped_data_with_basin(clip_data_folder, basin_shp_path, figure_path):
    """
    Visualize clipped raster data with basin boundary overlay and save the figure.

    Parameters:
    -----------
    clip_data_folder : str
        Path to the folder containing clipped raster files
    basin_shp_path : str
        Path to the basin shapefile
    figure_path : str
        Path to the folder where the output PNG will be saved
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from rasterio.plot import show
    import matplotlib.colors as colors
    import matplotlib.patches as mpatches
    import os

    # Read the basin shapefile
    basin_gdf = gpd.read_file(basin_shp_path)

    # Get all TIF files in the clipped data folder
    tif_files = glob.glob(os.path.join(clip_data_folder, '*.tif'))

    if not tif_files:
        print(f"No TIF files found in {clip_data_folder}")
        return False

    print(f"Found {len(tif_files)} TIF files in {clip_data_folder}")

    # Map file names to descriptive titles and custom colormaps
    file_info = {
        'facc_clip.tif': {
            'title': 'Flow Accumulation Map (FAM)',
            'cmap': 'Blues',
            'norm': colors.LogNorm()  # Logarithmic scale for flow accumulation
        },
        'dem_clip.tif': {
            'title': 'Digital Elevation Model (DEM)',
            'cmap': 'terrain',
            'norm': None  # Linear scale for elevation
        },
        'fdir_clip.tif': {
            'title': 'Drainage Direction Map (DDM)',
            'cmap': None,  # Will be set to custom colormap for directions
            'norm': None
        }
    }

    # Display up to 3 TIF files with basin boundary in a single row
    if len(tif_files) > 0:
        # Create a figure with subplots in a single row with reduced spacing
        fig, axes = plt.subplots(1, min(3, len(tif_files)), figsize=(15, 5),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

        # Reduce horizontal spacing between subplots
        plt.subplots_adjust(wspace=0)

        # If only one file, axes won't be an array
        if len(tif_files) == 1:
            axes = [axes]

        # Process each file and plot in the corresponding subplot
        for i, (tif_file, ax) in enumerate(zip(tif_files[:3], axes)):
            file_name = os.path.basename(tif_file)
            # print(f"Visualizing: {file_name}")

            with rasterio.open(tif_file) as src:
                data = src.read(1)
                transform = src.transform

                # Add geographic features
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.STATES, linestyle=':')

                # Get custom visualization settings for this file
                file_settings = file_info.get(file_name, {
                    'title': file_name,
                    'cmap': 'viridis',
                    'norm': None
                })

                # Special handling for flow direction map to use discrete colors
                if file_name == 'fdir_clip.tif':
                    # Get unique values in the direction data
                    unique_values = np.unique(data)
                    unique_values = unique_values[~np.isnan(unique_values)]

                    # Create a custom colormap for the unique direction values
                    n_values = len(unique_values)
                    colors_list = plt.cm.tab10(np.linspace(0, 1, n_values))

                    # Create a custom discrete colormap
                    cmap = colors.ListedColormap(colors_list)
                    bounds = np.concatenate([unique_values - 0.5, [unique_values[-1] + 0.5]])
                    norm = colors.BoundaryNorm(bounds, cmap.N)

                    # Show the raster with discrete colors
                    img = show(data, ax=ax, transform=transform, cmap=cmap, norm=norm)

                    # Create a legend for direction values
                    legend_patches = []
                    for j, val in enumerate(unique_values):
                        patch = mpatches.Patch(color=colors_list[j], label=f'Direction {int(val)}')
                        legend_patches.append(patch)

                    # Add the legend
                    ax.legend(handles=legend_patches, loc='lower right', fontsize='small')
                else:
                    # Show other raster data with standard colormap
                    img = show(data, ax=ax, transform=transform,
                              cmap=file_settings['cmap'], norm=file_settings['norm'])

                    # Fix colorbar issue by directly creating it from the image
                    if hasattr(img, 'get_images') and img.get_images():
                        # For matplotlib 3.5+
                        cbar = plt.colorbar(img.get_images()[0], ax=ax, shrink=0.7)
                    elif hasattr(img, 'images') and img.images:
                        # For older matplotlib versions
                        cbar = plt.colorbar(img.images[0], ax=ax, shrink=0.7)

                    # Set appropriate colorbar label
                    if file_name == 'facc_clip.tif':
                        cbar.set_label('Flow Accumulation (log scale)')
                    elif file_name == 'dem_clip.tif':
                        cbar.set_label('Elevation (m)')

                # Add basin boundary with black color
                basin_gdf.boundary.plot(ax=ax, color='black', linewidth=2)

                # Add title using the mapping
                ax.set_title(file_settings['title'], fontsize=10)

        plt.tight_layout()
        # Save the figure as basic_data.png in the specified figure_path, dpi=300
        os.makedirs(figure_path, exist_ok=True)
        out_png = os.path.join(figure_path, "basic_data.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved to {out_png}")

def visualize_flow_accumulation_with_gauges(basin_shp_path, gauge_meta_path, clip_data_folder, figure_path):
    """
    Visualize flow accumulation with USGS gauge stations overlay and save the figure.
    This visualization emphasizes high flow accumulation values with a colormap
    and shows gauge stations with their IDs as an overlay.
    
    Parameters:
    -----------
    basin_shp_path : str
        Path to the watershed shapefile
    gauge_meta_path : str
        Path to the USGS gauge metadata CSV file
    clip_data_folder : str
        Path to the folder containing clipped raster files
    figure_path : str
        Path to the folder where the output PNG will be saved
    """
    import os
    import pandas as pd
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Ensure output directory exists
    os.makedirs(figure_path, exist_ok=True)
    
    # Find the flow accumulation file
    facc_file = os.path.join(clip_data_folder, 'facc_clip.tif')
    if not os.path.exists(facc_file):
        print(f"Flow accumulation file not found: {facc_file}")
        return
    
    # Load the basin shapefile
    basin_gdf = gpd.read_file(basin_shp_path)
    
    # Load USGS gauge information
    gauge_info = pd.read_csv(gauge_meta_path)
    
    # Convert gauge locations to GeoDataFrame
    gauge_points = gpd.GeoDataFrame(
        gauge_info,
        geometry=gpd.points_from_xy(gauge_info.LNG_GAGE, gauge_info.LAT_GAGE),
        crs='EPSG:4326'
    )
    
    # Reproject gauge points to match basin CRS if needed
    if gauge_points.crs != basin_gdf.crs:
        gauge_points = gauge_points.to_crs(basin_gdf.crs)
    
    # Spatial join to find gauges within the watershed
    gauges_in_basin = gpd.sjoin(gauge_points, basin_gdf, how='inner', predicate='within')
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Open and read the flow accumulation data
    with rasterio.open(facc_file) as src:
        data = src.read(1)
        transform = src.transform
        
        # Create a logarithmic colormap to emphasize high values
        data_mask = data > 0  # Exclude zeros and negative values
        if np.any(data_mask):
            vmin = data[data_mask].min()
            vmax = data.max()
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            
            # Use imshow instead of rasterio's show for better control
            extent = [
                transform[2],
                transform[2] + transform[0] * data.shape[1],
                transform[5] + transform[4] * data.shape[0],
                transform[5]
            ]
            
            # Create the image with imshow
            im = ax.imshow(
                data, 
                cmap='Blues',
                norm=norm,
                alpha=0.8,
                extent=extent
            )
            
            # Add a colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label('Flow Accumulation (log scale)')
            
            # Plot basin boundary
            basin_gdf.boundary.plot(ax=ax, color='red', linewidth=2, alpha=0.7)
            
            # Add title
            plt.title('Flow Accumulation with USGS Gauge Stations', fontsize=14)
            
            if len(gauges_in_basin) > 0:
                # Plot gauge locations
                gauges_in_basin.plot(
                    ax=ax,
                    color='black',
                    marker='^',
                    markersize=100,
                    alpha=0.9,
                    label='USGS Gauges'
                )
                
                # Add station IDs as labels
                for idx, row in gauges_in_basin.iterrows():
                    padded_staid = str(row['STAID']).zfill(8)
                    ax.annotate(
                        padded_staid,
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(7, 7),
                        textcoords='offset points',
                        color='black',
                        fontweight='bold',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
                    )
                
                # Add legend
                ax.legend(loc='upper right')
                
                # print(f"Added {len(gauges_in_basin)} gauge stations to the visualization")
            else:
                print("No USGS gauge stations found within the watershed boundary")
            
            # Remove axes
            ax.set_axis_off()
            
            # Save the figure
            output_path = os.path.join(figure_path, 'facc_with_gauges.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Flow accumulation visualization with gauges saved to {output_path}")
        else:
            print("No valid flow accumulation data found (all values <= 0)")
            plt.close(fig)


def visualize_figures_basin(figure_path):
    """
    Visualize all figures in the given directory and save them as PNG files.

    Parameters:
    figure_path : str
        Path to the folder where the output PNG will be saved
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import os
    
    # Check if the required PNG files exist
    basin_map_path = os.path.join(figure_path, 'basin_map_with_gauges.png')
    facc_map_path = os.path.join(figure_path, 'facc_with_gauges.png')
    basic_data_path = os.path.join(figure_path, 'basic_data.png')
    
    # First, visualize basin_map_with_gauges.png and facc_with_gauges.png side by side
    if os.path.exists(basin_map_path) and os.path.exists(facc_map_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Load and display basin map
        basin_img = mpimg.imread(basin_map_path)
        ax1.imshow(basin_img)
        ax1.set_title('Basin Map with Gauges', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Load and display flow accumulation map
        facc_img = mpimg.imread(facc_map_path)
        ax2.imshow(facc_img)
        ax2.set_title('Flow Accumulation with Gauges', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        combined_output_path = os.path.join(figure_path, 'combined_maps.png')
        plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"Combined visualization saved to {combined_output_path}")
    else:
        print("Warning: basin_map_with_gauges.png or facc_with_gauges.png not found")
    
    # Then, visualize basic_data.png separately
    if os.path.exists(basic_data_path):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Load and display basic data
        basic_img = mpimg.imread(basic_data_path)
        ax.imshow(basic_img)
        ax.set_title('Basic Data Visualization', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        print(f"Basic data visualization displayed from {basic_data_path}")
    else:
        print("Warning: basic_data.png not found")




def basin_processor(args):
    # print("DEBUG inside basin_processor ->", type(args), vars(args))
    print("Downloading watershed shapefile for basin: ", args.basin_name)
    Basin_Area = download_watershed_shp(args.selected_point[0], args.selected_point[1], args.basin_shp_path, args.basin_level)
    plot_watershed_with_gauges(args.basin_shp_path, args.gauge_meta_path, args.figure_path)

    download_hydrosheds_data(args.selected_point[0], args.selected_point[1], args.basic_data_path)
    batch_clip_tifs_by_shapefile(args.basic_data_path, args.basic_data_clip_path, args.basin_shp_path)
    visualize_clipped_data_with_basin(args.basic_data_clip_path, args.basin_shp_path, args.figure_path)
    
    # Add the new flow accumulation visualization with gauges
    visualize_flow_accumulation_with_gauges(args.basin_shp_path, args.gauge_meta_path, args.basic_data_clip_path, args.figure_path)
    visualize_figures_basin(args.figure_path)
    return Basin_Area



