#!/usr/bin/env python3

# Muhammet Nergizci, 2023


###The usual python imports for the notebook
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from scipy.ndimage import median_filter



def geojson2txt(input_file, output_file, columns_to_extract):
    '''
    The code provide the geojson to txt file
    Nergizci September, 2023
    # Example usage:
    # Define the input GeoJSON file, output text file, and columns to extract
    input_file = 'desc_merged.geojson'
    output_file = 'desc_merged.txt'
    columns_to_extract = 'velocity'
    
    # Call the function to extract and save the specified columns
    geojson2txt(input_file, output_file, columns_to_extract)

    '''
    # Load the GeoPandas DataFrame from the input file
    gdf = gpd.read_file(input_file)

    # Extract longitude, latitude, and velocity into separate columns
    gdf['longitude'] = gdf['geometry'].x
    gdf['latitude'] = gdf['geometry'].y

    # You can now access 'longitude', 'latitude', and 'velocity' columns
    result_df = gdf[['longitude', 'latitude', columns_to_extract]]
    # Save the result to a text file
    result_df.to_csv(output_file, sep='\t', index=False, header=False)

def multipol2xyz(input_geojson_path, output_csv_path):
    '''
    # This function help to extract multipolygon geometry to x,y column!
    Nergizci September, 2023
    extract_coordinates_to_csv('asc_frame_frame.geojson', 'hello.csv')
    '''
    # GeoJSON verilerini yükleyin
    gdf = gpd.read_file(input_geojson_path)

    # Koordinatları saklamak için boş listeleri başlatın
    x_coords = []
    y_coords = []

    # Feature'ları döngüye alın ve koordinatları çıkarın
    for index, row in gdf.iterrows():
        geometry = row['geometry']
        if geometry.geom_type == 'MultiPolygon':
            for polygon in geometry.geoms:
                for point in polygon.exterior.coords:
                    x_coords.append(point[0])
                    y_coords.append(point[1])
        elif geometry.geom_type == 'Polygon':
            for point in geometry.exterior.coords:
                x_coords.append(point[0])
                y_coords.append(point[1])

    # Çıkarılan koordinatları içeren bir DataFrame oluşturun
    coordinate_df = pd.DataFrame({'x': x_coords, 'y': y_coords})

    # Koordinatları bir CSV dosyasına kaydedin
    coordinate_df.to_csv(output_csv_path, index=False)


### it is for GMT change ogr2ogr -f "GMT" fault.xyz tien_shan_fault.geojson


def extract_burst_overlaps(frame):
    '''
    The code help to extract burstoverlap geopandas dataframe in both swath base and frame base:
    example: gpd_overlaps, overlap_gdf1, overlap_gdf2, overlap_gdf3 = extract_burst_overlaps('021D_05266_252525') 
    '''
   
    # Read GeoJSON data
    data_temp = gpd.read_file(frame + '.geojson')

    # Change CRS to EPSG:4326
    data_temp = data_temp.to_crs(epsg=4326)

    # Extract subswath information
    if frame.startswith('00'):
        data_temp['swath'] = data_temp.burstID.str[4]
    elif frame.startswith('0'):
        data_temp['swath'] = data_temp.burstID.str[5]
    else:
        data_temp['swath'] = data_temp.burstID.str[6]

    # Divide frame into subswaths
    data_temp = data_temp.sort_values(by=['burstID']).reset_index(drop=True)
    sw1 = data_temp[data_temp.swath == '1']
    sw2 = data_temp[data_temp.swath == '2']
    sw3 = data_temp[data_temp.swath == '3']

    # Divide burst overlaps into odd and even numbers
    a1 = sw1.iloc[::2]
    b1 = sw1.iloc[1::2]
    a2 = sw2.iloc[::2]
    b2 = sw2.iloc[1::2]
    a3 = sw3.iloc[::2]
    b3 = sw3.iloc[1::2]

    # Find burst overlaps
    overlap_gdf1 = gpd.overlay(a1, b1, how='intersection')
    overlap_gdf2 = gpd.overlay(a2, b2, how='intersection')
    overlap_gdf3 = gpd.overlay(a3, b3, how='intersection')

    # Merge swath overlaps
    gpd_overlaps = gpd.GeoDataFrame(pd.concat([overlap_gdf1, overlap_gdf2, overlap_gdf3], ignore_index=True))

    return gpd_overlaps, overlap_gdf1, overlap_gdf2, overlap_gdf3

###filtering


def apply_median_filter(data_array, filter_window_size = 32):
    """
    Apply median filtering to an xarray.DataArray.

    Parameters:
        data_array (xarray.DataArray): The input data array.
        filter_window_size (int): The size of the median filter window.

    Returns:
        xarray.DataArray: The filtered data array.

    # Example usage
    # Define your xarray.DataArray and filter_window_size
    # filtered_array = apply_median_filter(data_array, filter_window_size)
    """
    # Convert the DataArray to a NumPy array
    data_array_np = data_array.values

    # Apply median filtering using scipy's median_filter function
    filtered_data_np = median_filter(data_array_np, size=filter_window_size)

    # Create a new DataArray with the filtered data
    filtered_data_xr = xr.DataArray(filtered_data_np, coords=data_array.coords, dims=data_array.dims)

    return filtered_data_xr

def medianfilter_array(arr, ws = 32):
    """use dask median filter on array
    works with both xarray and numpy array
    """
    chunksize = (ws*8, ws*8)
    if type(arr)==type(xr.DataArray()):
        inn = arr.values
    else:
        inn = arr
    arrb = da.from_array(inn, chunks=chunksize)
    arrfilt=ndfilters.median_filter(arrb, size=(ws,ws), mode='reflect').compute()
    if type(arr)==type(xr.DataArray()):
        out = arr.copy()
        out.values = arrfilt
    else:
        out = arrfilt
    return out

def open_geotiff(path, fill_value=0):
    '''
    This code help open geotiff with gdal and remove nan to zero!
    '''
    try:
        bovl = gdal.Open(path, gdal.GA_ReadOnly)
        if bovl is None:
            raise Exception("Failed to open the GeoTIFF file.")

        band = bovl.GetRasterBand(1)
        bovl_data = band.ReadAsArray()

        # Replace NaN values with the specified fill_value
        bovl_data[np.isnan(bovl_data)] = fill_value

        return bovl_data
    except Exception as e:
        print("Error:", e)
        return None

from osgeo import gdal

# def export_to_tiff(output_filename, data_array, geotransform):
#     """
#     Export a NumPy array to a GeoTIFF file.
    
#     Parameters:
#     - output_filename: String, the name of the output GeoTIFF file.
#     - data_array: NumPy array containing the data to be exported.
#     - geotransform: List, the geotransform parameters [upper_left_x, x_pixel_size, x_rotation, upper_left_y, y_rotation, y_pixel_size].
#     # # Example usage
#     # output_filename = '2pai_correction_azboi.tif'
#     # corrected_az_of_bovl_new = corrected_az_of_bovl_new  # Assuming aa is defined in your code
#     # geotransform = [37.1415, 0.001, 0, 37.4131666, 0, -0.001]
#     # export_to_tiff(output_filename, corrected_az_of_bovl_new, geotransform)

#     Returns:
#     None
#     """
#     driver = gdal.GetDriverByName("GTiff")
    
#     # Get the dimensions of the data array
#     row, col = data_array.shape
    
#     # Create GeoTIFF
#     outdata = driver.Create(output_filename, col, row, 1, gdal.GDT_Float32)
#     outdata.SetGeoTransform(geotransform)
    
#     # Write data to the raster band
#     outdata.GetRasterBand(1).WriteArray(data_array)
    
#     # Flush the cache to disk
#     outdata.FlushCache()
#     outdata.FlushCache() 


def export_to_tiff(output_filename, data_array, reference_tif_path):
    """
    Export a NumPy array to a GeoTIFF file using a reference TIFF for geospatial properties.
    
    Parameters:
    - output_filename: String, the name of the output GeoTIFF file.
    - data_array: NumPy array containing the data to be exported.
    - reference_tif_path: String, the file path of the reference GeoTIFF.
        
    Returns:
    None

    # Example usage:
    # output_filename = 'exported_data.tif'
    # data_array = your_numpy_array_here  # NumPy array you want to export
    # reference_tif_path = 'path/to/reference.tif'
    # export_to_tiff_with_ref(output_filename, data_array, reference_tif_path)
    """
    # Open the reference TIFF to read its spatial properties
    ref_ds = gdal.Open(reference_tif_path)
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()

    driver = gdal.GetDriverByName("GTiff")
    
    # Get the dimensions of the data array
    row, col = data_array.shape
    
    # Create the output GeoTIFF
    outdata = driver.Create(output_filename, col, row, 1, gdal.GDT_Float32)
    
    # Set the geotransform and projection from the reference TIFF
    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(projection)
    
    # Write data to the raster band
    outdata.GetRasterBand(1).WriteArray(data_array)
    
    # Flush the cache to disk to write changes
    outdata.FlushCache()
    
    # Cleanup
    ref_ds = None
    outdata = None



# def gradient_nr(data, deramp=True):
#     """Calculates gradient of continuous data (not tested for phase)

#     Args:
#         xar (np.ndarray): A NumPy array, e.g. ifg['unw']
#         deramp (bool): If True, it will remove the overall ramp

#     Returns:
#         np.ndarray
        
#         gradient=calculate_gradient(azof,deramp=False)
#         plt.figure(figsize=(10,10))
#         plt.imshow(gradient, cmap='viridis', vmax=0.5)
#         plt.colorbar()    
#     """
#     gradis = data.copy()  # Assuming xar is already a NumPy array
#     vgrad = np.gradient(gradis)  # Use NumPy's gradient function
#     gradis = np.sqrt(vgrad[0]**2 + vgrad[1]**2)
#     if deramp:
#         gradis = deramp_unw(gradis)  # You should implement the deramp_unw function for NumPy arrays
#     return gradis

'''
example of histogram!
# Replace 0.00000000 with NaN
data=cc_boi
data[data == 0.00000000] = np.nan

# Remove NaN values and calculate statistics
data_no_nan = data[~np.isnan(data)]
median = np.nanmedian(data_no_nan)
mean = np.nanmean(data_no_nan)
minimum = np.nanmin(data_no_nan)
maximum = np.nanmax(data_no_nan)
std = np.nanstd(data_no_nan)
variance = np.nanvar(data_no_nan)

# Create and display a histogram
plt.hist(data_no_nan.ravel(), bins=50, color='b', alpha=0.7)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of rubber sheeted BOVL')

# Add lines for median and mean
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')

plt.legend()  # Show the legend

plt.show()

# Close the dataset
dataset = None

# Print mean and std
print(f"Mean: {mean}")
print(f"Standard Deviation (std): {std}")
'''













######################ISCE3 






##functionss
#Utility function to load data
def loadData(infile, band=1):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    #Data array
    data = ds.GetRasterBand(band).ReadAsArray()
    #Map extent
    trans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    extent = [trans[0], trans[0] + xsize * trans[1],
            trans[3] + ysize*trans[5], trans[3]]
    
    ds = None
    return data, extent

def asf_unzip(output_dir: str, file_path: str):
    """
    Takes an output directory path and a file path to a zipped archive.
    If file is a valid zip, it extracts all to the output directory.
    """
    ext = os.path.splitext(file_path)[1]
    assert type(output_dir) == str, 'Error: output_dir must be a string'
    assert type(file_path) == str, 'Error: file_path must be a string'
    assert ext == '.zip', 'Error: file_path must be the path of a zip'

    if path_exists(output_dir):
        if path_exists(file_path):
            print(f"Extracting: {file_path}")
            try:
                zipfile.ZipFile(file_path).extractall(output_dir)
            except zipfile.BadZipFile:
                print(f"Zipfile Error.")
            return


def multiLook(infile, outfile, fmt='GTiff', xlooks=None, ylooks=None, noData=None, method='average'):
    '''
    infile - Input file to multilook
    outfile - Output file to multilook
    fmt - Output format
    xlooks - Number of looks in x/range direction
    ylooks - Number of looks in y/azimuth direction
    '''
    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    #Input file dimensions
    xSize = ds.RasterXSize
    ySize = ds.RasterYSize

    #Output file dimensions
    outXSize = xSize//xlooks
    outYSize = ySize//ylooks

    ##Set up options for translation
    gdalTranslateOpts = gdal.TranslateOptions(format=fmt, 
                                              width=outXSize, height=outYSize,
                                             srcWin=[0,0,outXSize*xlooks, outYSize*ylooks],
                                             noData=noData, resampleAlg=method)

    #Call gdal_translate
    gdal.Translate(outfile, ds, options=gdalTranslateOpts)       
    ds = None
    



def multiLookCpx(infile, outfile, fmt='GTiff', xlooks=None, ylooks=None, noData=None, method='average'):
    '''
    infile - Input file to multilook
    outfile - Output file to multilook
    fmt - Output format
    xlooks - Number of looks in x/range direction
    ylooks - Number of looks in y/azimuth direction
    
    
    input cpx file
        |
    2 band real virtual
        |
    2 band real multilooked virtual
        |
    1 band complex virtual
        |
    output cpx file
        
    '''
    sourcexml = '''    <SimpleSource>
      <SourceFilename>{0}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>'''.format(infile)
    
    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    #Input file dimensions
    xSize = ds.RasterXSize
    ySize = ds.RasterYSize

    #Output file dimensions
    outXSize = xSize//xlooks
    outYSize = ySize//ylooks

    #Temporary filenames
    inmemfile = '/vsimem/cpxlooks.2band.vrt'
    inmemfile2 = '/vsimem/cpxlooks.multilooks.2band.vrt'
    inmemfile3 = '/vsimem/cpxlooks.combine.vrt'
    
    ##This is where we convert it to real bands and multilook
    #Create driver
    driver = gdal.GetDriverByName('VRT')
    rivrtds = driver.Create(inmemfile,xSize, ySize, 0)
    
    #Create realband
    options = ['subClass=VRTDerivedRasterBand',
               'PixelFunctionType=real',
               'SourceTransferType=CFloat32']
    rivrtds.AddBand(gdal.GDT_Float32, options)
    rivrtds.GetRasterBand(1).SetMetadata({'source_0' : sourcexml}, 'vrt_sources')
    
    #Create imagband
    options = ['subClass=VRTDerivedRasterBand',
               'PixelFunctionType=imag',
               'SourceTransferType=CFloat32']
    rivrtds.AddBand(gdal.GDT_Float32, options)
    rivrtds.GetRasterBand(2).SetMetadata({'source_0' : sourcexml}, 'vrt_sources')
    
    ##Add projection information
    rivrtds.SetProjection(ds.GetProjection())
    ds = None
    

    ##Set up options for translation
    gdalTranslateOpts = gdal.TranslateOptions(format='VRT', 
                                              width=outXSize, height=outYSize,
                                             srcWin=[0,0,outXSize*xlooks, outYSize*ylooks],
                                             noData=noData, resampleAlg=method)

    #Apply multilooking on real and imag channels
    mlvrtds = gdal.Translate(inmemfile2, rivrtds, options=gdalTranslateOpts)
    rivrtds = None
    mlvrtds = None
        
    #Write from memory to VRT using pixel functions
    mlvrtds = gdal.OpenShared(inmemfile2)
    cpxvrtds = driver.Create(inmemfile3, outXSize, outYSize, 0)
    cpxvrtds.SetProjection(mlvrtds.GetProjection())
    cpxvrtds.SetGeoTransform(mlvrtds.GetGeoTransform())


    options = ['subClass=VRTDerivedRasterBand',
               'pixelFunctionType=complex',
               'sourceTransferType=CFloat32']
    xmltmpl = '''    <SimpleSource>
      <SourceFilename>{0}</SourceFilename>
      <SourceBand>{1}</SourceBand>
    </SimpleSource>'''
    
    md = {'source_0': xmltmpl.format(inmemfile2, 1),
          'source_1': xmltmpl.format(inmemfile2, 2)}

    cpxvrtds.AddBand(gdal.GDT_CFloat32, options)
    cpxvrtds.GetRasterBand(1).SetMetadata(md, 'vrt_sources')
    mlvrtds = None
        
        
    ###Now create copy to format needed
    driver = gdal.GetDriverByName(fmt)
    outds = driver.CreateCopy(outfile, cpxvrtds)
    cpxvrtds = None
    
    outds = None
    gdal.Unlink(inmemfile)
    gdal.Unlink(inmemfile2)
    gdal.Unlink(inmemfile3)

'''
#Oversample greenland dem to 500m
#!gdal_translate -of GTiff -tr 100 100 -r cubicspline Greenland_500m.tif green_try.tif

#Greenland DEM downsampled to 60, 110
#!gdal_translate -of GTiff -outsize 60 110 -r nearest NETCDF:"Greenland1km.nc":topg Greenland_subsample.tif

#multilooking
#!gdal_translate -of GTiff -outsize 10% 10% NETCDF:"Greenland1km.nc":topg Greenland_10perc.tif
'''