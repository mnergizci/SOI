#!/usr/bin/env python3
'''
Created on 10/02/2024
@author: M Nergizci, some functions adapted from M Lazecky.  
'''
import numpy as np
import py_gamma as pg
import subprocess
import shutil
import time
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from matplotlib.path import Path
from lics_unwrap import *
#import framecare as fc
from scipy import ndimage 
from geocube.api.core import make_geocube
import geopandas as gpd
#import daz_lib_licsar as dl
#import LiCSAR_misc as misc
import sys
import rasterio
from rasterio.merge import merge
import os
#import LiCSquery as lq
from scipy.constants import speed_of_light
from scipy.signal import convolve2d, medfilt
from scipy.ndimage import sobel
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
import dask.array as da
from dask_image import ndfilters
from scipy.ndimage import generic_filter
import scipy.ndimage.filters as ndfilters

pi=np.pi
##############################################
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

def gradient_nr(data, deramp=True):
    """Calculates gradient of continuous data (not tested for phase)

    Args:
        xar (np.ndarray): A NumPy array, e.g. ifg['unw']
        deramp (bool): If True, it will remove the overall ramp

    Returns:
        np.ndarray
        
        gradient=calculate_gradient(azof,deramp=False)
        plt.figure(figsize=(10,10))
        plt.imshow(gradient, cmap='viridis', vmax=0.5)
        plt.colorbar()    
    """
    gradis = data.copy()  # Assuming xar is already a NumPy array
    vgrad = np.gradient(gradis)  # Use NumPy's gradient function
    gradis = np.sqrt(vgrad[0]**2 + vgrad[1]**2)
    if deramp:
        gradis = deramp_unw_np(gradis)  # You should implement the deramp_unw function for NumPy arrays
    return gradis

def deramp_unw_np(array):
    """Deramps unwrapped interferogram represented as a NumPy array."""
    # Assuming _detrend_2d_ufunc can work directly with NumPy arrays.
    # Replace np.nan with 0s for compatibility with the detrending function.
    array_nonan = np.nan_to_num(array)
    
    # Apply the detrending function.
    # This is a simplified call assuming _detrend_2d_ufunc is compatible with NumPy arrays.
    detrended_array = _detrend_2d_ufunc(array_nonan)
    
    return detrended_array

# Example usage:
# array = np.random.rand(100, 100)  # Example NumPy array
# detrended_array = deramp_unw_np(array)
    
def _detrend_2d_ufunc(arr):
    assert arr.ndim == 2, "Input array must be 2D."
    rows, cols = arr.shape
    
    # Prepare the design matrix for a plane: 1, x, and y terms.
    col0 = np.ones(rows * cols)
    col1 = np.repeat(np.arange(rows), cols) + 1
    col2 = np.tile(np.arange(cols), rows) + 1
    G = np.vstack([col0, col1, col2]).T
    
    # Reshape the 2D array into a vector for the observed data.
    d_obs = arr.ravel()
    
    # Use numpy.linalg.lstsq for solving the linear least squares problem.
    # It is more numerically stable than manually inverting matrices.
    m_est, _, _, _ = np.linalg.lstsq(G, d_obs, rcond=None)
    
    # Compute the estimated (fitted) data points and reshape to the original 2D array shape.
    d_est = G.dot(m_est)
    linear_fit = d_est.reshape(arr.shape)
    
    # Return the detrended array.
    return arr - linear_fit



def adf_flt(phase_data, kernel_size=5, alpha=0.6, median_kernel_size=3):
    """
    Enhanced adaptive filter inspired by Goldstein's approach for phase images, incorporating a median filter
    on the residuals between the original and initially filtered phase data.
    
    Args:
        phase_data (numpy.ndarray): 2D array of phase data.
        kernel_size (int): Size of the Gaussian kernel for the initial smoothing.
        alpha (float): Factor to control the adaptiveness of the initial filter.
        median_kernel_size (int): Size of the kernel for the median filter applied to residuals.
        
    Returns:
        numpy.ndarray: Final, adjusted filtered phase data.
    """
    # Step 0: change zero to NaN before filtering...
    phase_data = np.where(phase_data == np.nan, 0, phase_data)
    
    # Step 1: Convert phase to complex representation
    complex_data = np.exp(1j * phase_data)

    # Step 2: Generate a Gaussian kernel for initial smoothing
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(kernel_size / alpha))
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.sum()

    # Step 3: Apply initial filtering
    filtered_complex = convolve2d(complex_data, kernel, mode='same', boundary='wrap')
    filtered_phase = np.angle(filtered_complex)

    # Step 4: Calculate and smooth residuals with median filter
    residuals = phase_data - filtered_phase
    smoothed_residuals = medfilt(residuals, median_kernel_size)

    # Step 5: Adjust the filtered phase with smoothed residuals
    adjusted_filtered_phase = filtered_phase + smoothed_residuals

    #preserve the edge
    edge_control=phase_data-adjusted_filtered_phase
    condition = (edge_control == phase_data)
    adjusted_filtered_phase = np.where(condition, phase_data, adjusted_filtered_phase)
    adjusted_filtered_phase = np.where(phase_data == 0, np.nan, adjusted_filtered_phase)
    return adjusted_filtered_phase.astype(np.float32) #np.angle(np.exp(1j * adjusted_filtered_phase))  # Ensure the phase is wrapped properly


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

###median filtering on residuals, it meanings take time more.
from scipy.ndimage import median_filter
import dask.array as da
import xarray as xr
from dask_image.ndfilters import median_filter
# Make sure you have dask_image installed, or install it using pip install dask_image

def medianfilt_res(arr, ws=96):
    """
    Use Dask median filter on array and apply residual filtering.
    Works with both xarray and numpy array.
    
    Args:
        arr: Input array, either a NumPy array or an xarray DataArray.
        ws (int): Window size for the median filter.
        
    Returns:
        The filtered array with residual adjustment, same type as input.
    """
    print(ws)
    chunksize = (ws*8, ws*8)
    
    # Check if input is xarray DataArray and extract values if so
    is_xarray = isinstance(arr, xr.DataArray)
    if is_xarray:
        inn = arr.values
    else:
        inn = arr
    
    # Convert to Dask array for chunked processing
    arrb = da.from_array(inn, chunks=chunksize)
    
    # Apply median filter using dask_image.ndfilters
    arrfilt = median_filter(arrb, size=(ws, ws), mode='reflect').compute()
    
    # Calculate residuals
    residuals = inn - arrfilt
    
    # Smooth residuals using Dask (note: convert back to Dask array with the same chunksize)
    residuals_dask = da.from_array(residuals, chunks=chunksize)
    smoothed_residuals = median_filter(residuals_dask, size=(ws, ws), mode='reflect').compute()
    
    # Adjust the filtered array with smoothed residuals
    adjusted_arrfilt = arrfilt + smoothed_residuals
    
    # Wrap output in xarray DataArray if input was xarray
    if is_xarray:
        out = arr.copy()
        out.values = adjusted_arrfilt
    else:
        out = adjusted_arrfilt
    
    return out


def median_filter_phase(phase_data, median_kernel_size=7):
    """
    Apply median filtering to phase data.
    
    Args:
        phase_data (numpy.ndarray): 2D array of phase data.
        median_kernel_size (int): Size of the kernel for the median filter.
        
    Returns:
        numpy.ndarray: Median-filtered phase data.
    """
    # Replace NaNs with zeros (if needed)
    #phase_data_nonan = np.where(np.isnan(phase_data), 0, phase_data)
    
    # Apply median filtering
    filtered_phase = medfilt2d(phase_data, kernel_size=median_kernel_size)

    # Apply residual filtering
    residuals = phase_data - filtered_phase
    smoothed_residuals = medfilt2d(residuals, median_kernel_size)

    # Step 5: Adjust the filtered phase with smoothed residuals
    adjusted_filtered_phase = filtered_phase + smoothed_residuals
    
    # Restore NaNs where original data was NaN (if needed)
    #filtered_phase = np.where(np.isnan(phase_data), np.nan, filtered_phase)
    
    return adjusted_filtered_phase.astype(np.float32)


def gaussian_filter_phase(phase_data, sigma=1.0, median_kernel_size=3):
    """
    Apply Gaussian filtering to phase data.
    
    Args:
        phase_data (numpy.ndarray): 2D array of phase data.
        sigma (float): Standard deviation for Gaussian kernel.
        
    Returns:
        numpy.ndarray: Gaussian-filtered phase data.
    """
    # Replace NaNs with zeros (if needed)
    #phase_data_nonan = np.where(np.isnan(phase_data), 0, phase_data)
    
    # Apply Gaussian filtering
    filtered_phase = gaussian_filter(phase_data, sigma=sigma)
    
    # Apply residual filtering
    residuals = phase_data - filtered_phase
    smoothed_residuals = medfilt2d(residuals, median_kernel_size)

    # Step 5: Adjust the filtered phase with smoothed residuals
    adjusted_filtered_phase = filtered_phase + smoothed_residuals
    
    # Restore NaNs where original data was NaN (if needed)
    #filtered_phase = np.where(np.isnan(phase_data), np.nan, filtered_phase)
    
    return adjusted_filtered_phase.astype(np.float32)
  

#################
# def apply_mask_only(data, polygons, reverse_imag=False):
#     """
#     Applies a mask generated for polygons to the data, setting masked areas to np.nan.
#     For a specific condition (e.g., bwr_filtered), reverses the sign of the imaginary part.
    
#     Parameters:
#     - data: The numpy array to be masked. Can be complex.
#     - polygons: A list of shapely Polygon objects used to generate the mask.
#     - reverse_imag: Boolean flag to indicate if the imaginary part's sign should be reversed.
    
#     Returns:
#     - A numpy array with the same type as 'data' where areas covered by the polygons are
#       unchanged, and areas not covered are set to np.nan. If reverse_imag is True, the
#       imaginary part's sign is reversed for the masked areas.
#     """
#     dtype = np.complex64 if np.iscomplexobj(data) else np.float32
#     masked_data = np.full(data.shape, np.nan, dtype=dtype)
    
#     # Generate mask for polygons
#     mask = generate_mask_for_polygons_optimized(data.shape, polygons)
    
#     if reverse_imag:
#         # Apply the mask and reverse the sign of the imaginary part within the masked area
#         imag_part_reversed = np.conj(data[mask])  # Reverse sign of imag part
#         masked_data[mask] = np.real(data[mask]) + 1j * np.imag(imag_part_reversed)
#     else:
#         # Apply the mask without reversing the sign of the imaginary part
#         masked_data[mask] = data[mask]
    
#     return masked_data



def apply_mask_only(data, polygons, reverse_imag=False, phase_factor=-0.53):
    """
    Applies a mask generated for polygons to the data, setting masked areas to np.nan.
    For a specific condition (e.g., bwr_filtered), multiplies the phase by a factor if reverse_imag is True.
    
    Parameters:
    - data: The numpy array to be masked. Can be complex.
    - polygons: A list of shapely Polygon objects used to generate the mask.
    - reverse_imag: Boolean flag to indicate if the phase should be multiplied by the factor.
    - phase_factor: The factor by which to multiply the phase of the data in the masked areas.
    
    Returns:
    - A numpy array with the same type as 'data' where areas covered by the polygons are
      unchanged, and areas not covered are set to np.nan. If reverse_imag is True,
      the phase of the data is multiplied by the given factor in the masked areas.
    """
    dtype = np.complex64 if np.iscomplexobj(data) else np.float32
    masked_data = np.full(data.shape, np.nan, dtype=dtype)
    
    # Generate mask for polygons
    mask = generate_mask_for_polygons_optimized(data.shape, polygons)
    
    if reverse_imag:
        # Extract magnitude and phase for complex data within the mask
        magnitude = np.abs(data[mask])
        phase = np.angle(data[mask])
        
        # Multiply the phase by the given factor
        modified_phase = phase / phase_factor
        
        # Convert back to complex form using the modified phase
        modified_data = magnitude * np.exp(1j * modified_phase)
        
        masked_data[mask] = modified_data
    else:
        # Apply the mask without modifying the phase
        masked_data[mask] = data[mask]
    
    return masked_data




def pha2cpx(pha, cpx, az_line, width):
    """
    Modifies the real part of a new complex array generated from phase data ('pha')
    by replacing it with the real part of an existing complex array ('cpx').
    
    Parameters:
    - pha: A numpy array containing phase data.
    - cpx: A complex numpy array from which the real part is extracted and reshaped.
    - az_line: The number of azimuth lines for reshaping 'cpx'.
    - width: The width (number of range pixels) for reshaping 'cpx'.
    
    Returns:
    - A complex numpy array generated from 'pha' with its real part replaced by
      the real part from 'cpx', after reshaping 'cpx' to match 'az_line' and 'width'.
    """
    # Reshape 'cpx' to match the expected dimensions and generate a new complex array from 'pha'
    #cpx_reshaped = cpx.reshape(az_line, width)
    cpx_reshaped = cpx.copy()
    new_cpx_from_pha = np.exp(1j * pha)
    new_cpx_from_pha=new_cpx_from_pha.flatten()
    # Ensure 'new_cpx_from_pha' is compatible with the reshaped 'cpx' dimensions
    if new_cpx_from_pha.shape != cpx_reshaped.shape:
        raise ValueError("Shape mismatch between phase-generated complex array and existing complex array")
    
    # Extract the real part of reshaped 'cpx' and the imaginary part of 'new_cpx_from_pha'
    real_part_old_cpx = np.real(cpx_reshaped)
    imaginary_part_new_cpx_from_pha = np.imag(new_cpx_from_pha)
    
    # Combine the extracted real part with the imaginary part of the new complex array
    modified_cpx = real_part_old_cpx + 1j * imaginary_part_new_cpx_from_pha
    modified_cpx=modified_cpx.reshape(az_line, width)
    return modified_cpx



def create_tab_file(epoch, frame_dir, frame, type='RSLC'):
    # Adjust the file name based on the type
    if type == 'RSLC':
        tab_file = os.path.join(frame_dir, 'tab', epoch + 'R_tab')
    elif type == 'SLC':
        tab_file = os.path.join(frame_dir, 'tab', epoch + '_tab')
    else:
        raise ValueError("Unsupported type specified. Choose 'RSLC' or 'SLC'.")

    # Ensure the tab directory exists
    os.makedirs(os.path.dirname(tab_file), exist_ok=True)

    # Check if the tab file does not exist to avoid overwriting
    if not os.path.exists(tab_file):
        if type == 'RSLC':
            inp = os.path.join('RSLC', epoch, epoch)  # Corrected to match the type in the path
            cmd = f"createSLCtab_frame {inp} rslc {frame}"
        elif type == 'SLC':
            inp = os.path.join('SLC', epoch, epoch)  # Corrected to match the type in the path
            cmd = f"createSLCtab_frame {inp} slc {frame}"
        
        # Execute the command and write its output to the tab file
        with open(tab_file, 'w') as file:
            subprocess.run(cmd, shell=True, stdout=file, check=True)

    return tab_file

def framepath_tab(tab_array, frame_directory):
    """Prepend frame directory path to each entry in the tab array."""
    updated_tab_array = []
    for row in tab_array:
        updated_row = [os.path.join(frame_directory, item) for item in row]
        updated_tab_array.append(updated_row)
    return updated_tab_array


def rasterize_polygon_optimized(polygon, shape):
    # Convert polygon points to a Path object
    path = Path(polygon.exterior.coords)

    # Generate a grid of points across the array
    y, x = np.mgrid[:shape[0], :shape[1]]
    points = np.vstack((x.flatten(), y.flatten())).T

    # Use the Path object to test which points are inside the polygon
    mask = path.contains_points(points).reshape(shape)

    return mask

def s1_azfm(r, t0, azp):
  """azfr = s1_azfm(r, t0, azp)
  
  Calculate azimuth FM rate given slant range, reference slant-range delay and the azimuth FM rate polynomial for ScanSAR data
  
  **Arguments:**
  
  * r:    slant range (meters)
  * t0:   reference slant range time for the polynomial (center swath delay in s)
  * azp:  polynomial coefficients
  
  **Output:**
  
  * the function returns the azimuth FM rate"""

  tsr = 2.0 * r / speed_of_light;
  dt = tsr - t0;
  azfr = azp[0] + dt * (azp[1] + dt*(azp[2] + dt*(azp[3] + dt*azp[4])));
  return azfr;

def generate_mask_for_polygons_optimized(shape, polygons):
    # Initialize masks with the same shape as diff_double_mask_pha, filled with False
    mask = np.zeros(shape, dtype=bool)
    y_indices, x_indices = np.indices(shape)

    for polygon in polygons:
        minx, miny, maxx, maxy = polygon.bounds
        # Convert bounds to indices; you might need to adjust this depending on the coordinate system
        minx_idx, miny_idx = int(minx), int(miny)
        maxx_idx, maxy_idx = int(maxx) + 1, int(maxy) + 1
        # Ensure indices are within the array bounds
        minx_idx, miny_idx = max(minx_idx, 0), max(miny_idx, 0)
        maxx_idx, maxy_idx = min(maxx_idx, shape[1]), min(maxy_idx, shape[0])

        # Create a sub-mask for the bounding box area
        sub_mask = np.zeros_like(mask, dtype=bool)
        sub_mask[miny_idx:maxy_idx, minx_idx:maxx_idx] = True

        # Refine the sub-mask by checking points within the bounding box against the polygon
        sub_y, sub_x = np.ogrid[miny_idx:maxy_idx, minx_idx:maxx_idx]
        for x, y in zip(sub_x.flatten(), sub_y.flatten()):
            if polygon.contains(Point(x, y)):
                mask[y, x] = True
            else:
                sub_mask[y, x] = False

        # Combine the sub-mask with the overall mask
        mask |= sub_mask

    return mask

def get_param_gamma(param, parfile, floatt = True, pos = 0):
    a = grep1line(param,parfile).split()[1+pos]
    if floatt:
        a = float(a)
    return a


def get_dfDC(path_to_slcdir, f0=5405000500, burst_interval = 2.758277, returnka = True, returnperswath = False, returnscalefactor=True):
    #f0 = get_param_gamma('radar_frequency', parfile)
    #burst_interval = get_param_gamma('burst_interval', topsparfile)
    epoch = os.path.basename(path_to_slcdir)
    frame = path_to_slcdir.split('/')[-3]
    
    if len(frame)!=17:
        frame=path_to_slcdir.split('/')[-4]
    parfile = os.path.join(path_to_slcdir, epoch+'.rslc.par')
    #parfile = glob.glob(path_to_slcdir+'/????????.slc.par')[0]
    #topsparfiles = glob.glob(path_to_slcdir+'/????????.IW?.slc.TOPS_par')
    #iwparfiles = glob.glob(path_to_slcdir+'/????????.IW?.slc.par')
    #
    
    lam = speed_of_light / f0
    dfDC = []
    kas = []
    ctr_range = []
    far_range = []
    near_range = []
    scalefactor= []
    afmrate_srdly = []
    afmrate_ply= []
    kr_list = []
    numbursts = [ int(frame.split('_')[2][:2]), int(frame.split('_')[2][2:4]), int(frame.split('_')[2][4:6])]
    azps_list = []
    az_line_time_list = []
    #krs = []
    #print('This is a proper solution but applied to primary SLC image. originally it is applied by GAMMA on the RSLC...')
    #for n in range(len(topsparfiles)):
    for n in [1,2,3]:
        topsparfile = os.path.join(path_to_slcdir, epoch+'.IW'+str(n)+'.rslc.TOPS_par')
        iwparfile = os.path.join(path_to_slcdir, epoch+'.IW'+str(n)+'.rslc.par')
        if (not os.path.exists(iwparfile)) or (not os.path.exists(topsparfile)):
            dfDC.append(np.nan)
            kas.append(np.nan)
            numbursts[n-1] = np.nan
        else:
            #topsparfile = topsparfiles[n]
            #iwparfile = iwparfiles[n]
            az_steering_rate = get_param_gamma('az_steering_rate', topsparfile) # az_steering_rate is the antenna beam steering rate
            #az_ster_rate.append(az_steering_rate)
            r1 = get_param_gamma('center_range_slc', iwparfile)
            #get the satellite velocity
            midNstate = int(get_param_gamma('number_of_state_vectors', iwparfile)/2)+1
            sv = 'state_vector_velocity_' + str(midNstate)
            velc1 = get_param_gamma(sv, iwparfile, pos=0)
            velc2 = get_param_gamma(sv, iwparfile, pos=1)
            velc3 = get_param_gamma(sv, iwparfile, pos=2)
            vsat = np.sqrt(velc1**2 + velc2**2 + velc3**2)
            midNstate=1
            # now some calculations
            afmrate_srdelay = get_param_gamma('az_fmrate_srdelay_'+ str(midNstate), topsparfile)
            afmrate_srdly.append(afmrate_srdelay) 
            afmrate_poly = []
            afmrate_poly.append(get_param_gamma('az_fmrate_polynomial_' + str(midNstate), topsparfile, pos = 0))
            afmrate_poly.append(get_param_gamma('az_fmrate_polynomial_' + str(midNstate), topsparfile, pos = 1))
            afmrate_poly.append(get_param_gamma('az_fmrate_polynomial_' + str(midNstate), topsparfile, pos = 2))
            try:
                afmrate_poly.append(get_param_gamma('az_fmrate_polynomial_' + str(midNstate), topsparfile, pos = 3))
            except:
                afmrate_poly.append(0)
            try:
                afmrate_poly.append(get_param_gamma('az_fmrate_polynomial_' + str(midNstate), topsparfile, pos = 4))
            except:
                afmrate_poly.append(0)
            afmrate_ply.append(afmrate_poly)
            ka = s1_azfm(r1, afmrate_srdelay, afmrate_poly) #unit: Hz/s == 1/s^2
            kr = -2.0 * vsat * az_steering_rate*(pi / 180.0) / lam
            kr_list.append(kr)
            if (kr != 0.0):
                #kt = ka * kr/(kr - ka)
                # but maybe should be kt = (kr*ka)/(ka-kr) # see https://iopscience.iop.org/article/10.1088/1755-1315/57/1/012019/pdf  --- and remotesensing-12-01189-v2, and Fattahi et al...
                # ok, gamma reads kr to be ka... corrected
                kt = kr * ka/(ka - kr)
            else:
                kt = -ka
            #finally calculate dfDC:
            #burst_interval = get_param_gamma('burst_interval', topsparfile)
            kas.append(ka)
            #krs.append(kr)
            dfDC.append(kt*burst_interval) #burst_interval is time within the burst... we can also just calculate.. see Grandin: eq 15: hal.archives-ouvertes.fr/hal-01621519/document
            #ok, that's the thing - burst_interval is actually t(n+1) - t(n) - see remotesensing-12-01189-v2
            #so it should be kt * -burst_interval, that is why GAMMA has the -kt J ... ok, good to realise this

            ####calculating scalefactor (Nergizci)
            azps=np.float64(get_param_gamma('azimuth_pixel_spacing', iwparfile))
            if not azps_list:
                azps_list.append(azps)
            az_line_time=np.float64(get_param_gamma('azimuth_line_time', iwparfile))
            if not az_line_time_list:
                az_line_time_list.append(az_line_time)
            
            dfdc=kt*burst_interval
            sf=(azps)/(dfdc*az_line_time*2*np.pi)
            scalefactor.append(sf)
            
            ###calculating ssd (Nergizci)
            ctr_range_temp=get_param_gamma('center_range_slc', iwparfile)
            far_range_temp=get_param_gamma('far_range_slc', iwparfile)
            near_range_temp=get_param_gamma('near_range_slc', iwparfile)
            ctr_range.append(ctr_range_temp)
            far_range.append(far_range_temp)
            near_range.append(near_range_temp)
    
    #print(scalefactor)
    #for sw1 and sw2
    
    r_so=(far_range[0] + near_range[1])/2
    ka_sb_list=[]
    kt_sb_list=[]
    t_bd_list=[]
    for n in range(2):
        topsparfile = os.path.join(path_to_slcdir, epoch+'.IW'+str(n+1)+'.rslc.TOPS_par')
        iwparfile = os.path.join(path_to_slcdir, epoch+'.IW'+str(n+1)+'.rslc.par')
        ka_sb=s1_azfm(r_so, afmrate_srdly[n], afmrate_ply[n])
        ka_sb_list.append(ka_sb)
        if (kr_list[n] != 0.0):
            kt_sb = ka_sb_list[n]* kr_list[n]/(ka_sb_list[n] - kr_list[n])
        else:
            kt_sb = -ka_sb_list[n]
        kt_sb_list.append(kt_sb)

        t_bd_list_temp = []
        #get burst delta times
        for i in range(numbursts[n]):
            t_bd=get_param_gamma(f'burst_start_time_{i+1}', topsparfile)
            t_bd_list_temp.append(t_bd)

        t_bd_list.append(t_bd_list_temp)

    # Directly calculate dt_f for each corresponding pair across the two lists
    dt_f_list = [t_bd_list[1][i] - t_bd_list[0][i] for i in range(len(t_bd_list[0]))]
    
    dt_b_list = []
    for dt_f in dt_f_list:
        if dt_f < 0:
            dt_b = burst_interval + dt_f
        else:
            dt_b = dt_f - burst_interval
        dt_b_list.append(dt_b)
    
    # Now calculate dfDC_f and dfDC_b using the time differences
    dfDC_f_list = [(kt_sb_list[0]+kt_sb_list[1])*dt_f/2.0 for dt_f in dt_f_list]
    dfDC_b_list = [(kt_sb_list[0]+kt_sb_list[1])*dt_b/2.0 for dt_b in dt_b_list]

    # ####calculating scalefactor
    
    if len(azps_list) > 0 and len(az_line_time_list) > 0:
        azps = azps_list[0]
        az_line_time = az_line_time_list[0]

        # Calculate sff and sfb for each dfDC_f and dfDC_b value
        sff1 = [azps / (df * -1*az_line_time * 2 * np.pi) for df in dfDC_f_list]
        sfb1 = [azps / (db * -1*az_line_time * 2 * np.pi) for db in dfDC_b_list]
    
    else:
        print("azps_list or az_line_time_list is empty, cannot calculate sff and sfb.")

    ####################################################
    #for subswath 2-3
    r_so=(far_range[1] + near_range[2])/2
    ka_sb_list=[]
    kt_sb_list=[]
    t_bd_list=[]
    for n in range(1,3,1):
        topsparfile = os.path.join(path_to_slcdir, epoch+'.IW'+str(n+1)+'.rslc.TOPS_par')
        iwparfile = os.path.join(path_to_slcdir, epoch+'.IW'+str(n+1)+'.rslc.par')
        ka_sb=s1_azfm(r_so, afmrate_srdly[n], afmrate_ply[n])
        ka_sb_list.append(ka_sb)
        if (kr_list[n] != 0.0):
            kt_sb = ka_sb_list[n-1]* kr_list[n]/(ka_sb_list[n-1] - kr_list[n])
        else:
            kt_sb = -ka_sb_list[n-1]
        kt_sb_list.append(kt_sb)

        t_bd_list_temp = []
        #get burst delta times
        for i in range(numbursts[n]):
            t_bd=get_param_gamma(f'burst_start_time_{i+1}', topsparfile)
            t_bd_list_temp.append(t_bd)

        t_bd_list.append(t_bd_list_temp)


    # Directly calculate dt_f for each corresponding pair across the two lists
    dt_f_list = [t_bd_list[1][i] - t_bd_list[0][i] for i in range(len(t_bd_list[0]))]

    dt_b_list = []
    for dt_f in dt_f_list:
        if dt_f < 0:
            dt_b = burst_interval + dt_f
        else:
            dt_b = dt_f - burst_interval
        dt_b_list.append(dt_b)
    
    # Now calculate dfDC_f and dfDC_b using the time differences
    dfDC_f_list = [(kt_sb_list[0]+kt_sb_list[1])*dt_f/2.0 for dt_f in dt_f_list]
    dfDC_b_list = [(kt_sb_list[0]+kt_sb_list[1])*dt_b/2.0 for dt_b in dt_b_list]

    # ####calculating scalefactor
    
    if len(azps_list) > 0 and len(az_line_time_list) > 0:
        azps = azps_list[0]
        az_line_time = az_line_time_list[0]

        # Calculate sff and sfb for each dfDC_f and dfDC_b value
        sff2 = [azps / (df *1* az_line_time * 2 * np.pi) for df in dfDC_f_list]
        sfb2 = [azps / (db *1*az_line_time * 2 * np.pi) for db in dfDC_b_list]
    
    else:
        print("azps_list or az_line_time_list is empty, cannot calculate sff and sfb.")
    
    # Create a list with two columns, where each row is [sff1[i], sff2[i]]
    final_sff_temp = [[sff1[i], sff2[i]] for i in range(len(sff1))]
    # Reshape sff from 25x2 to a 50x1 list
    final_sff = [row[0] for row in final_sff_temp] + [row[1] for row in final_sff_temp]
    
    final_sfb_temp = [[sfb1[i], sfb2[i]] for i in range(len(sfb1))]
    # Reshape sff from 25x2 to a 50x1 list
    final_sfb = [row[0] for row in final_sfb_temp[1:]] + [row[1] for row in final_sfb_temp[1:]]
    if not returnperswath:
        numbursts = np.array(numbursts)
        dfDC = np.nansum(numbursts*np.array(dfDC)) / np.sum(numbursts)
        ka = np.nansum(numbursts*np.array(kas)) / np.sum(numbursts)
    #kr = np.mean(krs)
    if returnka:
        return dfDC, ka #, kr
    if returnscalefactor:
        return scalefactor, final_sff, final_sfb
    else:
        return dfDC

