import os
import re
import numpy
import subprocess
import os
import sys
import re
import time
import requests
import dateutil
import datetime as dt
import numpy as np
import warnings
import py_gamma as pg
from shapely.geometry import Polygon
from shapely.geometry import Point
#from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib import pyplot as plt
from modules_sw_mn import * #functions saved here
from func_mnergizci import *
#import matplotlib.path as path
import LiCSBAS_io_lib as io_lib
# import LiCSBAS_loop_lib as loop_lib
# import LiCSBAS_tools_lib as tools_lib
import pickle

if len(sys.argv) < 4:
    print('Please provide frame, pair and kernel size for median filtering of azimuth information: i.e python auto_cor_script.py 021D_05266_252525 20230129_20230210 96')
    sys.exit(1)

BLUE = '\033[94m'
ORANGE= '\033[38;5;208m'
RED = '\033[91m' 
GREEN = '\033[92m' 
ENDC = '\033[0m' 

# ANSI code to end formatting

##variables
frame=sys.argv[1]
pair=sys.argv[2]
kernel=np.int64(sys.argv[3])
batchdir=os.environ['BATCH_CACHE_DIR']
prime, second = pair.split('_')
framedir=os.path.join(batchdir, frame)
tr= int(frame[:3])

SLC_dir=os.path.join(framedir, 'SLC')
IFGdir=os.path.join(framedir, 'IFG')
RSLC_dir=os.path.join(framedir, 'RSLC')

###tiff example for exporting geotiff
GEOC_dir=os.path.join(framedir, 'GEOC')
geo_tif= os.path.join(GEOC_dir, pair, pair + '.geo.bovldiff.tif')


a=os.listdir(SLC_dir)
for i in a:
    if i.startswith('2') and len(i) == 8:
        ref_epoc=i
mli_par=os.path.join(SLC_dir, ref_epoc, ref_epoc +'.slc.mli.par')
width=int(io_lib.get_param_par(mli_par, 'range_samples'))
length=int(io_lib.get_param_par(mli_par, 'azimuth_lines'))
print(width, length)
bovl=np.fromfile(os.path.join(IFGdir, pair, 'ddiff_pha_adf'), np.float32).reshape(length, width).byteswap()
bovl_coh=np.fromfile(os.path.join(IFGdir, pair, 'ddiff_coh_adf'), np.float32).reshape(length, width).byteswap()
azi=np.fromfile(os.path.join(IFGdir, pair, pair+'.azi'), np.float32).reshape(length, width).byteswap()
###I can make this function?? already in subswth_bovl_mn


# number of subswaths
nsubswaths = 3

# number of looks
rlks = 20
azlks = 4

# parameter file of the mosaic
SLC_par = pg.ParFile(os.path.join(RSLC_dir,prime, prime + '.rslc.par'))

# # TOPS parameter file of each subswath
TOPS_par = []
TOPS_par.append(pg.ParFile(os.path.join(RSLC_dir,prime,prime +'.IW1.rslc.TOPS_par')))
TOPS_par.append(pg.ParFile(os.path.join(RSLC_dir,prime,prime +'.IW2.rslc.TOPS_par')))
TOPS_par.append(pg.ParFile(os.path.join(RSLC_dir,prime,prime +'.IW3.rslc.TOPS_par')))

# # read SLC parameters
r0 = SLC_par.get_value('near_range_slc', index = 0, dtype = float)        # near range distance
rps = SLC_par.get_value('range_pixel_spacing', index = 0, dtype = float)  # range pixel spacing
t0 = SLC_par.get_value('start_time', index = 0, dtype = float)            # time of first azimuth line
tazi = SLC_par.get_value('azimuth_line_time', index = 0, dtype = float)   # time between each azimuth line


# get number of bursts per subswath
# max_nbursts = 0
# nbursts = []
# for sw in range(nsubswaths):
#   nbursts.append(TOPS_par[sw].get_value('number_of_bursts', index = 0, dtype = int))
#   if nbursts[sw] > max_nbursts:
#     max_nbursts = nbursts[sw]
max_nbursts = 0
min_nbursts = float('inf')  # Initialize min_nbursts to infinity
nbursts = []

# Find the max and min number of bursts
for sw in range(nsubswaths):
    nb = TOPS_par[sw].get_value('number_of_bursts', index=0, dtype=int)  # Mocked function call
    nbursts.append(nb)
    if nb > max_nbursts:
        max_nbursts = nb
    if nb < min_nbursts:
        min_nbursts = nb

# Update nbursts to minimum burst size across all subswaths
nbursts = [min_nbursts] * nsubswaths



# initialize first and last range and azimuth pixel of each burst (SLC mosaic)
rpix0 = np.zeros((nsubswaths, min_nbursts))
rpix2 = np.zeros((nsubswaths, min_nbursts))
azpix0 = np.zeros((nsubswaths, min_nbursts))
azpix2 = np.zeros((nsubswaths, min_nbursts))

for sw in range(nsubswaths):
  for b in range(nbursts[sw]):
    # read burst window (modified burst window as in previous e-mail)
    ext_burst_win = TOPS_par[sw].get_value('ext_burst_win_%d' %(b+1))
    burst_win = TOPS_par[sw].get_value('burst_win_%d' %(b+1))
    #calculate pixel coordinates of bursts in mosaicked image
    rpix0[sw, b] = round((float(burst_win[0]) - r0) / rps)
    rpix2[sw, b] = round((float(burst_win[1]) - r0) / rps)
    azpix0[sw, b] = round((float(burst_win[2]) - t0) / tazi)
    azpix2[sw, b] = round((float(burst_win[3]) - t0) / tazi)


# first and last range and azimuth pixel of each burst (MLI mosaic / interferogram geometry)
rpix_ml0 = rpix0 / rlks
rpix_ml2 = (rpix2 + 1) / rlks - 1
azpix_ml0 = azpix0 / azlks
azpix_ml2 = (azpix2 + 1) / azlks - 1


#############find the overlap polygons in the radar coordinates:
# calculate intersection of bursts (subswath intersection)

bursts = []
overlaps_sw = {}  # Use a dictionary to store overlaps for each sw

for sw in range(0, 3):
    p_inter_sw = []
    offset = 30  # to create overlaps between bursts
    overlaps = []  # This will store overlaps for the current subswath

    for b0 in range(nbursts[sw] - 1):  # Ensure there's a next burst to compare with by stopping one early
        # Calculate the corners for the current burst polygon
        rg_az1 = [
            [rpix_ml0[sw, b0], azpix_ml0[sw, b0] - offset],
            [rpix_ml2[sw, b0], azpix_ml0[sw, b0] - offset],
            [rpix_ml2[sw, b0], azpix_ml2[sw, b0] + offset],
            [rpix_ml0[sw, b0], azpix_ml2[sw, b0] + offset]
        ]
        p0 = Polygon(rg_az1)
        bursts.append(p0)
    
        # Calculate the corners for the next burst polygon
        rg_az2 = [
            [rpix_ml0[sw, b0 + 1], azpix_ml0[sw, b0 + 1] - offset],
            [rpix_ml2[sw, b0 + 1], azpix_ml0[sw, b0 + 1] - offset],
            [rpix_ml2[sw, b0 + 1], azpix_ml2[sw, b0 + 1] + offset],
            [rpix_ml0[sw, b0 + 1], azpix_ml2[sw, b0 + 1] + offset]
        ]
        p1 = Polygon(rg_az2)
    
        # Calculate the overlap between the current burst and the next one
        overlap = p0.intersection(p1)
        overlaps.append(overlap)  # Add the overlap to the list for the current subswath

    overlaps_sw[sw] = overlaps  # Store the overlaps for the current subswath in the dictionary

##get dfdc values

path_to_slcdir=os.path.join(RSLC_dir,prime)
sfbo, sff, sfb=get_dfDC(path_to_slcdir, f0=5405000500, burst_interval=2.758277, returnka=False, returnperswath=False, returnscalefactor=True)

####let's m2rad for azi:

print('median filtering starting')

###filtering_azi

azi64_path=os.path.join(IFGdir, pair, pair + f'_azi{kernel}')

if os.path.exists(azi64_path):
    print(f'{kernel} median filtered azi exists, so skip to filtering!')
    azi64=np.fromfile(azi64_path, np.float32).reshape(length, width).byteswap()
else:
    print(f'{kernel} median filter not exits, filtering continue. It can take time...')
    azi64=medianfilt_res(azi, ws=kernel)
    azi64.byteswap().tofile(azi64_path)

print('median_filter is done!')

###geocoding progress
###variables for geocoding
lt_fine_suffix='lt_fine'
geo_dir= os.path.join(framedir, 'geo')
if os.path.exists(geo_dir) and os.path.isdir(geo_dir):
  for file in os.listdir(geo_dir):
    if file.endswith(lt_fine_suffix):
      lt_fine_file=os.path.join(geo_dir, file) 
        
  EQA_path=os.path.join(geo_dir, 'EQA.dem_par')
  widthgeo=int(io_lib.get_param_par(EQA_path, 'width'))
  print(f' widthgeo; {widthgeo}')
else:
  print(f'geo folder doesnt exists. Please check your {framedir}')
geoc_file=os.path.join(f'azi{kernel}.geo')
exec_str=['geocode_back', azi64_path, str(width), lt_fine_file, geoc_file, str(widthgeo), '0', '0', '0' ]
try:
  subprocess.run(exec_str, check=True)
  print(f"Command executed successfully: {' '.join(exec_str)}")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while executing the command: {e}")
    
geoc_tif=os.path.join(IFGdir, pair, pair + f'_azi{kernel}.geo.tif')
exec_str=['data2geotiff', EQA_path, geoc_file,'2', geoc_tif, '0.0' ]
try:
  subprocess.run(exec_str, check=True)
  print(f"Command executed successfully: {' '.join(exec_str)}")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while executing the command: {e}")





def optimize_azi_scaling(azi, overlaps_sw, sfbo):
    azi_adjusted = np.full(azi.shape, np.nan, dtype=azi.dtype)  # Başlangıçta tüm değerleri NaN ile doldur
    
    for sw, polygons in overlaps_sw.items():
        for polygon in polygons:
            minx, miny, maxx, maxy = polygon.bounds  # Poligonun sınırlarını al
            
            # Sınırları matrisin boyutlarına sığacak şekilde ayarla
            min_row, max_row = int(max(miny, 0)), int(min(maxy, azi.shape[0] - 1))
            min_col, max_col = int(max(minx, 0)), int(min(maxx, azi.shape[1] - 1))
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if polygon.contains(Point(col, row)):  # Nokta poligon içinde mi kontrol et
                        azi_adjusted[row, col] = azi[row, col] * sfbo[sw]  # sfbo ile çarp

    return azi_adjusted

azi_rd_file=(os.path.join(IFGdir, pair, pair + f'_azi{kernel}_rad'))
if not os.path.exists(azi_rd_file):
    print('Azimuth offset values changing meter to radian.')
    sfbo_m2r = [1/x for x in sfbo]
    azi_rd = optimize_azi_scaling(azi64, overlaps_sw, sfbo_m2r)
    # Assuming you have a function to save azi_rd to a file, which might look like this:
    azi_rd.astype(np.float32).byteswap().tofile(azi_rd_file)
else:
    print(f'{azi_rd_file} already exists.')
    # When loading, specify dtype and reshape
    azi_rd = np.fromfile(azi_rd_file, dtype=np.float32).reshape(length, width).byteswap()
####so we have azi bovl and sovl in radian base. We can make mathematical calculation easily rigth now.

print('meter2radian done! Wrapping calculating is just started!')
##coh masking
###coherence thresholding 0.7
threshold=0.8
mask_cc=(bovl_coh >threshold).astype(int)

###to get rid of edge problem
bovl_nan = np.where(bovl == 0, np.nan, bovl)
masknp = ~np.isnan(bovl_nan)
mask_cc=mask_cc*masknp
######


bovlrad_cc=bovl*mask_cc
azi_rd_cc=azi_rd*mask_cc
azi_rd_cc[azi_rd_cc==0]=np.nan

diff=bovlrad_cc-azi_rd_cc
diff=np.nan_to_num(diff)
diff_wrap= np.mod(diff + np.pi, 2*np.pi) - np.pi
azbovlrad=azi_rd_cc+diff_wrap
# azbovlrad[azbovlrad==0]=np.nan

print('Unwrapped bovl values changing rad2m')

azbovlmeter_file = os.path.join(IFGdir, pair, pair + f'_azibovl{kernel}_meter')
if not os.path.exists(azbovlmeter_file):
    azbovlmeter_temp = optimize_azi_scaling(azbovlrad, overlaps_sw, sfbo)
    print('rad2m is done!')
    azbovlmeter = np.nan_to_num(azbovlmeter_temp, nan=0)
    azbovlmeter = azbovlmeter.astype(np.float32)
    azbovlmeter.byteswap().tofile(azbovlmeter_file)  # Corrected method chain for saving
else:
    print(f'{azbovlmeter_file} already exists')
    azbovlmeter = np.fromfile(azbovlmeter_file, dtype=np.float32).reshape(length, width).byteswap()

# print('printing')
# plt.figure(figsize=(20, 20))
# plt.imshow(azbovlmeter, cmap='bwr')
# plt.colorbar()
# plt.savefig('azbovlmeter.png')

# print('printing')
# plt.figure(figsize=(20, 20))
# plt.imshow(azbovlrad, cmap='bwr', vmin=-10, vmax=10)
# plt.colorbar()
# plt.savefig('azbovlrad.png')

###save geotiff.
#output_bovl=os.path.join(GEOC_dir, pair, pair, + '.geo.bovl_unw.tif')


print('Geocoding starting')

###variables for geocoding
lt_fine_suffix='lt_fine'
geo_dir= os.path.join(framedir, 'geo')
if os.path.exists(geo_dir) and os.path.isdir(geo_dir):
  for file in os.listdir(geo_dir):
    if file.endswith(lt_fine_suffix):
      lt_fine_file=os.path.join(geo_dir, file) 
        
  EQA_path=os.path.join(geo_dir, 'EQA.dem_par')
  widthgeo=int(io_lib.get_param_par(EQA_path, 'width'))
  print(f' widthgeo; {widthgeo}')
else:
  print(f'geo folder doesnt exists. Please check your {framedir}')


###geocoding progress
geoc_file=os.path.join(IFGdir, pair, pair + f'_azibovl{kernel}_meter.geo')
exec_str=['geocode_back', azbovlmeter_file, str(width), lt_fine_file, geoc_file, str(widthgeo), '0', '0', '0' ]
try:
  subprocess.run(exec_str, check=True)
  print(f"Command executed successfully: {' '.join(exec_str)}")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while executing the command: {e}")
    
geoc_tif=os.path.join(IFGdir, pair, pair + f'_azibovl{kernel}_meter.geo.tif')
exec_str=['data2geotiff', EQA_path, geoc_file,'2', geoc_tif, '0.0' ]
try:
  subprocess.run(exec_str, check=True)
  print(f"Command executed successfully: {' '.join(exec_str)}")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while executing the command: {e}")
#################################
#################################





print(BLUE + 'unwrapped BOI is done! Lets go for SOI!' + ENDC)

####I need to change for 0.6 in subswath_bovl_mn.
temp_file=os.path.join(framedir, 'temp_data')

sovl=np.fromfile(os.path.join(temp_file, pair + '_diff_double_mask_pha'), np.float32).reshape(length, width).byteswap()
bwrs=os.path.join(temp_file,frame +'_bwr.pkl')
fwrs=os.path.join(temp_file,frame +'_fwr.pkl')

if not os.path.exists(bwrs) and not os.path.exists(fwrs):
    print(RED + 'The bwr and fwr polygons not in the folder. Please run: subswath_bovl_mn.py in the $batchdir' + ENDC)

else:
    with open(bwrs, 'rb') as f:
        bwr = pickle.load(f)
    
    with open(fwrs, 'rb') as f:
        fwr = pickle.load(f)

####

def sovl_azi_scaling(azi, bwr, sfb):
    azi_adjusted = np.full(azi.shape, np.nan, dtype=azi.dtype)
    
    for polygon, scale in zip(bwr, sfb):
        minx, miny, maxx, maxy = polygon.bounds
        expansion = 0 # Adjust as necessary
        minx -= expansion
        miny -= expansion
        maxx += expansion
        maxy += expansion
        min_row, max_row = int(max(miny, 0)), int(min(maxy, azi.shape[0] - 1))
        min_col, max_col = int(max(minx, 0)), int(min(maxx, azi.shape[1] - 1))
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if polygon.contains(Point(col, row)):
                    azi_adjusted[row, col] = azi[row, col] * scale
    
    return azi_adjusted

####calculation for SOI forward.

sff_m2r=[(1/x)/(-0.53) for x in sff]
#sff_m2r=[(1/x) for x in sff]
azi_sovlf_rad=sovl_azi_scaling(azi64, fwr, sff_m2r)
azi_sovlf_rad[azi_sovlf_rad==0]=np.nan
##masking sovl in forwards to compare easily,,,,
mask_sf = ~np.isnan(azi_sovlf_rad)
sovl_f=sovl*mask_sf
##same step for unwrapping
diff_sff=sovl_f-azi_sovlf_rad
diff_sff=np.nan_to_num(diff_sff)
diff_sff_wrap= np.mod(diff_sff + np.pi, 2*np.pi) - np.pi
azisfrad=azi_sovlf_rad+diff_sff_wrap
print('Unwrapped bovl values changing rad2m')
sff_r2m=[1/x for x in sff_m2r]
azisfmeter=sovl_azi_scaling(azisfrad, fwr, sff_r2m)
print('rad2m is done!')




####################
#sfb_m2r=[((1/x)*(-0.53)) for x in sfb]
sfb_m2r=[((1/x)) for x in sfb]
azi_sovlb_rad=sovl_azi_scaling(azi64, bwr, sfb_m2r)
azi_sovlb_rad[azi_sovlb_rad==0]=np.nan
mask_sb = ~np.isnan(azi_sovlb_rad)
sovl_b=sovl*mask_sb
# ######playing with here
# azi_sovlf_rad_nonan=np.nan_to_num(azi_sovlf_rad, nan=0)
# azi_sovlb_rad_nonan=np.nan_to_num(azi_sovlb_rad, nan=0)
# azi_sovlb_rad_nonan+azi_sovlf_rad_nonan
# ######playing with here

#same step for unwrapping
diff_sfb=sovl_b-azi_sovlb_rad
diff_sfb=np.nan_to_num(diff_sfb)
# diff_sfb_wrap=np.mod(diff_sfb + np.pi, 2*np.pi)-np.pi
# azisbrad=azi_sovlb_rad+(diff_sfb_wrap)
diff_sfb_wrap = np.mod(diff_sfb + np.pi, 2 * np.pi)-np.pi
# Recalculate diff_sfb_wrap_r for the entire array
diff_sfb_wrap_r = np.mod(diff_sfb + np.pi, 2*np.pi)
# Use np.where to selectively apply your condition
# This will check the condition for each element in diff_sfb_wrap
# If the condition is true, it uses diff_sfb_wrap_r for that element; otherwise, it uses diff_sfb_wrap
azisbrad = azi_sovlb_rad + np.where(np.abs(diff_sfb_wrap) > np.float32(10), diff_sfb_wrap_r, diff_sfb_wrap)
print('Unwrapped bovl values changing rad2m')
sfb_r2m=[1/x for x in sfb_m2r]
azisbmeter=sovl_azi_scaling(azisbrad, bwr, sfb_r2m)
print('rad2m is done!')

azisbmeter_nonan = np.nan_to_num(azisbmeter, nan=0.0)
azisfmeter_nonan = np.nan_to_num(azisfmeter, nan=0.0)

aziSOImeter=azisbmeter_nonan+azisfmeter_nonan
aziSOImeter=aziSOImeter.astype(np.float32)
aziSOImeter_file=os.path.join(IFGdir, pair, pair +'_aziSOImeter')
aziSOImeter.byteswap().tofile(aziSOImeter_file)

###geocoding progress
geoc_file=os.path.join('aziSOI_meter.geo')
exec_str=['geocode_back', aziSOImeter_file, str(width), lt_fine_file, geoc_file, str(widthgeo), '0', '0', '0' ]
try:
  subprocess.run(exec_str, check=True)
  print(f"Command executed successfully: {' '.join(exec_str)}")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while executing the command: {e}")
    
geoc_tif=os.path.join(IFGdir, pair, pair + f'_aziSOI_meter.geo.tif')
exec_str=['data2geotiff', EQA_path, geoc_file,'2', geoc_tif, '0.0' ]
try:
  subprocess.run(exec_str, check=True)
  print(f"Command executed successfully: {' '.join(exec_str)}")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while executing the command: {e}")
#################################
#################################



    

######calculate gradient
##this was for automatic correction but not efficient. MAybe next time.
# print('gradient_calculation')
# azi_filt_cc=azbovlrad*mask_cc
# azi_filt_cc=np.where(azi_filt_cc==0,np.nan, azi_filt_cc)

# bovl_filt_cc=bovl*mask_cc
# bovl_filt_cc=np.where(bovl_filt_cc==0,np.nan, bovl_filt_cc)

# # Calculate the gradient magnitude (with deramping turned off for demonstration)
# bovl_gradient = gradient_nr(bovl_filt_cc, deramp=True)
# bool_gradientbovl = (bovl_gradient > 3).astype(int)

# azi_gradient = gradient_nr(azi_filt_cc, deramp=True)
# bool_gradientazi = (azi_gradient > 3).astype(int)

# bool_gradientbovl = np.nan_to_num(bool_gradientbovl, nan=0)
# bool_gradientazi = np.nan_to_num(bool_gradientazi, nan=0)

# bool_gradientazi=bool_gradientazi.astype(np.float32)
# bool_gradientbovl=bool_gradientbovl.astype(np.float32)

# ##saving
# bool_gradbovlfile='bool_gradbovl'
# bool_gradientbovl.byteswap().tofile(bool_gradbovlfile)

# bool_gradazifile='bool_gradazi'
# bool_gradientazi.byteswap().tofile(bool_gradazifile)


# for suffix in ['azi', 'bovl']:
#     bool_file=f'bool_grad{suffix}file'
#     geoc_file=f'bool_grad{suffix}.geo'
#     print(bool_file, geoc_file)
#     exec_str=['geocode_back', bool_file, str(width), lt_fine_file, geoc_file, str(widthgeo), '0', '0', '0' ]
#     try:
#       subprocess.run(exec_str, check=True)
#       print(f"Command executed successfully: {' '.join(exec_str)}")
#     except subprocess.CalledProcessError as e:
#       print(f"An error occurred while executing the command: {e}")
        
#     geoc_tif=f'bool_grad{suffix}.geo.tif'
#     exec_str=['data2geotiff', EQA_path, geoc_file,'2', geoc_tif, '0.0' ]
#     try:
#       subprocess.run(exec_str, check=True)
#       print(f"Command executed successfully: {' '.join(exec_str)}")
#     except subprocess.CalledProcessError as e:
#       print(f"An error occurred while executing the command: {e}")





######till here burst is unwrapped with aid of Azimuth offset.
######Let's start same process for SOI's





##################
# from matplotlib.patches import Polygon as MplPolygon

# def plot_image_and_polygons(image, polygons_dict):
#     """
#     Plots an image with polygons overlaid.
    
#     :param image: 2D numpy array (e.g., azimuth data).
#     :param polygons_dict: Dictionary with subswath numbers as keys and lists of Shapely polygons as values.
#     """
#     fig, ax = plt.subplots(figsize=(20, 20))
#     # Plot the image
#     ax.imshow(image, cmap='bwr', vmin=-10, vmax=10)
    
#     # Iterate through the dictionary to plot polygons for each subswath
#     for sw, polygons in polygons_dict.items():
#         for polygon in polygons:
#             # Convert the polygon points to a format suitable for matplotlib
#             mpl_poly = MplPolygon(list(polygon.exterior.coords), edgecolor='red', facecolor='none', linewidth=2)
#             ax.add_patch(mpl_poly)
    
# #     plt.savefig('selamlar.png')

# # Assuming 'azi' is your azimuth data array and 'overlaps_sw' contains your polygons
# plot_image_and_polygons(azi_rd, overlaps_sw)
###################