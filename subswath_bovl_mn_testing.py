#!/usr/bin/env python3
'''
Created on 10/02/2024

@author: M Nergizci, C Magnard. 
'''
import numpy as np
import os
import py_gamma as pg
import sys
import subprocess
import shutil
import time
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from matplotlib.path import Path
from modules_sw_mn_testing import * #functions saved here
from lics_unwrap import *
import pickle
import LiCSAR_misc as misc

if len(sys.argv) < 3:
    print('Please provide frame and pair information: i.e python bovls_rad2mm.py 021D_05266_252525 20230129_20230210')
    sys.exit(1)

BLUE = '\033[94m'
ORANGE= '\033[38;5;208m'
ENDC = '\033[0m'  # ANSI code to end formatting


##variables
frame=sys.argv[1]
pair=sys.argv[2]
batchdir=os.environ['BATCH_CACHE_DIR']
prime, second = pair.split('_')
framedir=os.path.join(batchdir, frame)
tr= int(frame[:3])
metafile = os.path.join(os.environ['LiCSAR_public'], str(tr), frame, 'metadata', 'metadata.txt')
master = misc.grep1line('master=',metafile).split('=')[1]
master_slcdir = os.path.join(framedir, 'SLC', master)

#tab_files
SLC1_tab_name =  create_tab_file(prime, framedir, frame)
RSLC2_tab_name=  create_tab_file(second, framedir, frame, type='RSLC')
master_tab_name= create_tab_file(master, framedir, frame, type='SLC')

#off
off_par=os.path.join(framedir, 'IFG', pair, pair+'.off')
sim_unw=os.path.join(framedir, 'IFG', pair, pair+'.sim_unw')


##output_dir
temp_file=os.path.join(framedir, 'temp_data')
if not os.path.exists(temp_file):
    os.makedirs(temp_file)
SLC1_tab_mod1_name = os.path.join(temp_file, prime+'_mod1.SLC_tab')
SLC1_tab_mod2_name = os.path.join(temp_file, prime+'_mod2.SLC_tab')
RSLC2_tab_mod1_name = os.path.join(temp_file, second+'_mod1.SLC_tab')
RSLC2_tab_mod2_name = os.path.join(temp_file, second+'_mod2.SLC_tab')
SLC1_mod1_name = os.path.join(temp_file, prime+'_mod1.slc')
SLC1_mod2_name = os.path.join(temp_file, prime+'_mod2.slc')
RSLC2_mod1_name = os.path.join(temp_file, second+'_mod1.slc')
RSLC2_mod2_name = os.path.join(temp_file, second+'_mod2.slc')
mli1_mod1_name = os.path.join(temp_file, prime+'_mod1.mli')
mli1_mod2_name = os.path.join(temp_file, prime+'_mod2.mli')
rmli2_mod1_name = os.path.join(temp_file, second+'_mod1.mli')
rmli2_mod2_name = os.path.join(temp_file, second+'_mod2.mli')
diff_mod1_name = os.path.join(temp_file, pair+'_mod1.diff')
diff_mod2_name = os.path.join(temp_file, pair+'_mod2.diff')
diff_mod1_mask_name = os.path.join(temp_file, pair+'_mod1_mask.diff')
diff_mod2_mask_name = os.path.join(temp_file, pair+'_mod2_mask.diff')


# number of looks
rlks = 20
azlks = 4

# read SLC_tab files
SLC1_tab_temp = pg.read_tab(SLC1_tab_name)
RSLC2_tab_temp = pg.read_tab(RSLC2_tab_name)
master_tab_temp = pg.read_tab(master_tab_name)
SLC1_tab = np.array(framepath_tab(SLC1_tab_temp, framedir))
RSLC2_tab = np.array(framepath_tab(RSLC2_tab_temp, framedir))
master_tab=np.array(framepath_tab(master_tab_temp, framedir))


nrows = SLC1_tab.shape[0]
ncols = SLC1_tab.shape[1]


# read image sizes
nr = []
naz = []

for i in range(nrows):
  SLC_par = pg.ParFile(SLC1_tab[i][1])
  nr.append(SLC_par.get_value('range_samples', dtype = int, index = 0))
  naz.append(SLC_par.get_value('azimuth_lines', dtype = int, index = 0))

##prepare data with empty subswaths
#create an array and store it as a binary file
for i in range(nrows):
    # Create the file name using f-string
    file_name = f'empty.iw{1+i}.slc'
    
    # Create the array
    if not os.path.exists(os.path.join(temp_file, file_name)):
      pg.create_array(file_name, nr[i], naz[i], 5, 0.0, 0.0) #output width nlines dtpes val val_im
    
    # Move the created file
      shutil.move(file_name, temp_file)
      

#for i in np.array((1, 2, 3)):
#    filepath= os.path.join(temp_file, f'empty.iw{i}.slc')
#    data=np.fromfile(filepath, dtype=np.complex64)
#    print(np.shape(data))
#sys.exit()

SLC1_tab_mod1 = SLC1_tab.copy()
SLC1_tab_mod2 = SLC1_tab.copy()
RSLC2_tab_mod1 = RSLC2_tab.copy()
RSLC2_tab_mod2 = RSLC2_tab.copy()

##For both reference and secondary image, 2 SLC mosaics are generated: mod2 use subswaths 1 and 3, mod1 uses subswath 2. 
SLC1_tab_mod1[1][0] =os.path.join(temp_file, 'empty.iw2.slc')
SLC1_tab_mod2[0][0] =os.path.join(temp_file,'empty.iw1.slc')
SLC1_tab_mod2[2][0] =os.path.join(temp_file, 'empty.iw3.slc')
RSLC2_tab_mod1[1][0] =os.path.join(temp_file, 'empty.iw2.slc')
RSLC2_tab_mod2[0][0] =os.path.join(temp_file, 'empty.iw1.slc')
RSLC2_tab_mod2[2][0] =os.path.join(temp_file, 'empty.iw3.slc')


##save the new tabs to temp_file: mod1 is IW2, mod2 is IW1 and IW3

pg.write_tab(SLC1_tab_mod1, SLC1_tab_mod1_name)
pg.write_tab(SLC1_tab_mod2, SLC1_tab_mod2_name)
pg.write_tab(RSLC2_tab_mod1, RSLC2_tab_mod1_name)
pg.write_tab(RSLC2_tab_mod2, RSLC2_tab_mod2_name)


#replace burst_win range paramaters by ext_burst_win range paramaters:

for i in range(nrows):
  TOPS_par_master=pg.ParFile(master_tab[i][2])
  TOPS_par=pg.ParFile(SLC1_tab[i][2])
  nburst=TOPS_par.get_value('number_of_bursts', dtype= int, index=0)
   
  for b in range(nburst):
    ext_burst_win = TOPS_par_master.get_value(f'ext_burst_win_{b+1}')
    #print('ext_burst_win:', ext_burst_win)
    burst_win=TOPS_par.get_value(f'burst_win_{b+1}')
    #print('burst_win:', burst_win)
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[0], index = 0)
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[1], index = 1)
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[4], index = 4)
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[5], index = 5)
    
  
  TOPS_par.write_par(SLC1_tab[i][2])
  pg.update_par(SLC1_tab[i][2], SLC1_tab[i][2])


###same progress for RSLC
  TOPS_par = pg.ParFile(RSLC2_tab[i][2])
  nburst = TOPS_par.get_value('number_of_bursts', dtype = int, index = 0)

  for b in range(nburst):
    ext_burst_win = TOPS_par_master.get_value(f'ext_burst_win_{b+1}')
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[0], index = 0)
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[1], index = 1)
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[4], index = 4)
    TOPS_par.set_value(f'burst_win_{b+1}', ext_burst_win[5], index = 5)

  TOPS_par.write_par(RSLC2_tab[i][2])
  pg.update_par(RSLC2_tab[i][2], RSLC2_tab[i][2])

##Please check until here, burst window don't grap well!!!####
##mosaic modified subswaths

print(BLUE + 'mosaicking_stars!!' + ENDC)
time.sleep(5)



# Check conditions and run pg.SLC_mosaic_ScanSAR if appropriate
if not (os.path.exists(SLC1_mod1_name) and os.path.exists(SLC1_mod1_name + '.par')):
    pg.SLC_mosaic_ScanSAR(SLC1_tab_mod1_name, SLC1_mod1_name, SLC1_mod1_name + '.par', rlks, azlks, 0)

if not (os.path.exists(SLC1_mod2_name) and os.path.exists(SLC1_mod2_name + '.par')):
    pg.SLC_mosaic_ScanSAR(SLC1_tab_mod2_name, SLC1_mod2_name, SLC1_mod2_name + '.par', rlks, azlks, 0)

if not (os.path.exists(RSLC2_mod1_name) and os.path.exists(RSLC2_mod1_name + '.par')):
    pg.SLC_mosaic_ScanSAR(RSLC2_tab_mod1_name, RSLC2_mod1_name, RSLC2_mod1_name + '.par', rlks, azlks, 0, SLC1_tab_name)

if not (os.path.exists(RSLC2_mod2_name) and os.path.exists(RSLC2_mod2_name + '.par')):
    pg.SLC_mosaic_ScanSAR(RSLC2_tab_mod2_name, RSLC2_mod2_name, RSLC2_mod2_name + '.par', rlks, azlks, 0, SLC1_tab_name)


print(BLUE + 'multilooking!!' + ENDC)
time.sleep(5)

##multilook
if not (os.path.exists(mli1_mod1_name) and os.path.exists(mli1_mod1_name + '.par')):
    pg.multi_look(SLC1_mod1_name, SLC1_mod1_name + '.par', mli1_mod1_name, mli1_mod1_name + '.par', rlks, azlks)

if not (os.path.exists(mli1_mod2_name) and os.path.exists(mli1_mod2_name + '.par')):
    pg.multi_look(SLC1_mod2_name, SLC1_mod2_name + '.par', mli1_mod2_name, mli1_mod2_name + '.par', rlks, azlks)

if not (os.path.exists(rmli2_mod1_name) and os.path.exists(rmli2_mod1_name + '.par')):
    pg.multi_look(RSLC2_mod1_name, RSLC2_mod1_name + '.par', rmli2_mod1_name, rmli2_mod1_name + '.par', rlks, azlks)

if not (os.path.exists(rmli2_mod2_name) and os.path.exists(rmli2_mod2_name + '.par')):
    pg.multi_look(RSLC2_mod2_name, RSLC2_mod2_name + '.par', rmli2_mod2_name, rmli2_mod2_name + '.par', rlks, azlks)

mli_mosaic_par = pg.ParFile(mli1_mod1_name + '.par')
mli_mosaic_nr = mli_mosaic_par.get_value('range_samples', dtype = int, index = 0)


print(BLUE + 'multilooking finished!!!!' + ENDC)
time.sleep(5)

#std_mosaic_par = pg.ParFile(SLC1_mod1_name+ '.par')
#std_mosaic_nr = std_mosaic_par.get_value('range_samples', dtype = int, index = 0)
#print(mli_mosaic_nr, std_mosaic_nr)

### calculate a raster image from data with power-law scaling using a specified colormap

pg.raspwr(mli1_mod1_name, mli_mosaic_nr)
pg.raspwr(mli1_mod2_name, mli_mosaic_nr)
pg.raspwr(rmli2_mod1_name, mli_mosaic_nr)
pg.raspwr(rmli2_mod2_name, mli_mosaic_nr)


# calculation of differential interferograms
if not (os.path.exists(diff_mod1_name) and os.path.exists(diff_mod2_name)):
  pg.SLC_diff_intf(SLC1_mod1_name, RSLC2_mod1_name, SLC1_mod1_name + '.par', RSLC2_mod1_name + '.par', off_par, sim_unw, diff_mod1_name, rlks, azlks, 1, 0, 0.2, 1, 1)
  pg.SLC_diff_intf(SLC1_mod2_name, RSLC2_mod2_name, SLC1_mod2_name + '.par', RSLC2_mod2_name + '.par', off_par, sim_unw, diff_mod2_name, rlks, azlks, 1, 0, 0.2, 1, 1)

  pg.rasmph_pwr(diff_mod1_name, mli1_mod1_name, mli_mosaic_nr)
  pg.rasmph_pwr(diff_mod2_name, mli1_mod2_name, mli_mosaic_nr)

# mask data
pg.mask_data(diff_mod1_name, mli_mosaic_nr, diff_mod1_mask_name, diff_mod2_name + '.bmp', 1)
pg.mask_data(diff_mod2_name, mli_mosaic_nr, diff_mod2_mask_name, diff_mod1_name + '.bmp', 1)

pg.rasmph_pwr(diff_mod1_mask_name, mli1_mod1_name, mli_mosaic_nr)
pg.rasmph_pwr(diff_mod2_mask_name, mli1_mod2_name, mli_mosaic_nr)

# visualize differential phase of subswath overlap areas
#pg.dis2ras(diff_mod1_mask_name + '.bmp', diff_mod2_mask_name + '.bmp')
#pg.dis2mph(diff_mod1_mask_name, diff_mod2_mask_name, mli_mosaic_nr, mli_mosaic_nr)


####make the double differencing!!
print(BLUE + 'double difference interferogram are calculating!!!!' + ENDC)
print('printing type and shape of the inf data')

mli_par_path=os.path.join(mli1_mod1_name + '.par')

if os.path.exists(mli_par_path):
  with open(mli_par_path, 'r') as mli_par:
    for line in mli_par:
      if line.startswith('range_samples'):
        width=int(line.split(':')[-1].strip())
      elif line.startswith('azimuth_lines'):
        az_line=int(line.split(':')[-1].strip())
print(width, az_line)

diff_mod1=np.fromfile(diff_mod1_mask_name, np.complex64).byteswap()
diff_mod2=np.fromfile(diff_mod2_mask_name, np.complex64).byteswap()
double_diff= diff_mod2*np.conj(diff_mod1)
diff_double_mask_temp=os.path.join(temp_file, pair + '_diff_double_mask_temp')
double_diff.byteswap().tofile(diff_double_mask_temp)
diff_double_mask_pha_temp=os.path.join(temp_file, pair + '_diff_double_mask_pha_temp')

###original values
# ###double_difference_inf
# diff_mod1=np.fromfile(diff_mod1_mask_name, np.complex64).byteswap()
# diff_mod2=np.fromfile(diff_mod2_mask_name, np.complex64).byteswap()
# double_diff= diff_mod2*np.conj(diff_mod1)
# diff_double_mask=os.path.join(temp_file, pair + '_diff_double_mask')
# double_diff.byteswap().tofile(diff_double_mask)

##create_geotiff

exec_str= ['cpx_to_real', diff_double_mask_temp, diff_double_mask_pha_temp, str(width), '4']
try:
  subprocess.run(exec_str, check=True)
  print(f"Command executed successfully: {' '.join(exec_str)}")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while executing the command: {e}")
##original values



#########
print(BLUE + 'bwr and fwr polygons are calculating!!!!' + ENDC)

prime_RSLC=os.path.join(framedir, 'RSLC', prime)
SLC_par=pg.ParFile(os.path.join(prime_RSLC, prime + '.rslc.par'))

nsubswaths = 3
TOPS_par = []
for i in range(1,4):
    TOPS_file=os.path.join(prime_RSLC, prime + f'.IW{i}.rslc.TOPS_par')
    TOPS_par.append(pg.ParFile(TOPS_file))

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

# Find minimum burst number, then fix for all subswaths.
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
    burst_win = TOPS_par[sw].get_value('burst_win_%d' %(b+1))
    # calculate pixel coordinates of bursts in mosaicked image
    rpix0[sw, b] = round((float(burst_win[0]) - r0) / rps)
    rpix2[sw, b] = round((float(burst_win[1]) - r0) / rps)
    azpix0[sw, b] = round((float(burst_win[2]) - t0) / tazi)
    azpix2[sw, b] = round((float(burst_win[3]) - t0) / tazi)
    
# first and last range and azimuth pixel of each burst (MLI mosaic / interferogram geometry)
rpix_ml0 = rpix0 / rlks
rpix_ml2 = (rpix2 + 1) / rlks - 1
azpix_ml0 = azpix0 / azlks
azpix_ml2 = (azpix2 + 1) / azlks - 1

# calculate intersection of bursts (subswath intersection)
p_inter = []
for sw in range(nsubswaths - 1):
  p_inter_sw = []
  for b0 in range(nbursts[sw]):
    p_inter_b0 = []
    rg_az1 = [[rpix_ml0[sw, b0], azpix_ml0[sw, b0]], [rpix_ml2[sw, b0], azpix_ml0[sw, b0]], [rpix_ml2[sw, b0], azpix_ml2[sw, b0]], [rpix_ml0[sw, b0], azpix_ml2[sw, b0]]]
    p0 = Polygon(rg_az1)
    for b1 in range(nbursts[sw+1]):
      rg_az2 = [[rpix_ml0[sw+1, b1], azpix_ml0[sw+1, b1]], [rpix_ml2[sw+1, b1], azpix_ml0[sw+1, b1]], [rpix_ml2[sw+1, b1], azpix_ml2[sw+1, b1]], [rpix_ml0[sw+1, b1], azpix_ml2[sw+1, b1]]]
      p1 = Polygon(rg_az2)
      p_inter_b0.append(p0.intersection(p1))
    p_inter_sw.append(p_inter_b0)
  p_inter.append(p_inter_sw)

# show the intersection between burst 6 of iw1 and burst 5 of iw2 (MLI mosaic geometry)
# print(p_inter[0][5][4].bounds)
# print(p_inter[1][5][4].bounds)

bwr = []
fwr = []


### Define the backward and forward subswath overlap polygons
for sw in range(len(p_inter)):
    # Iterate over bursts in the current sub swath
    for b0 in range(len(p_inter[sw])):
        for b1 in range(len(p_inter[sw][b0])):
            #print(sw, b0, b1)
            # Corrected the logical expression to use logical OR `or` instead of bitwise OR `|`
            if (azpix_ml0[1][0] - azpix_ml0[0][0] > 0 and sw==0):
                #print(sw, b0, b1)
                if b0 == b1:
                    # Forward intersection
                    fwr.append(p_inter[sw][b0][b1])
                elif b0 + 1 == b1:
                    # Backward intersection
                    bwr.append(p_inter[sw][b1][b0])
                    
            elif (azpix_ml0[2][0] - azpix_ml0[1][0] > 0 and sw==1):
                if b0 == b1:
                    # Forward intersection
                    fwr.append(p_inter[sw][b0][b1])
                elif b0 + 1 == b1:
                    # Backward intersection
                    bwr.append(p_inter[sw][b1][b0])

            # Corrected chained comparison for clarity and correctness
            elif azpix_ml0[1][0] - azpix_ml0[0][0] < -300:
                if sw == 0 and b0 + 2 == b1:
                    #print(sw, b0, b1)
                    fwr.append(p_inter[sw][b0][b1])
                elif sw == 0 and b0 + 1 == b1:
                    # Backward intersection
                    bwr.append(p_inter[sw][b0][b1])
            elif -300 < azpix_ml0[1][0] - azpix_ml0[0][0] < 0:
                if sw == 0 and b0 + 1== b1:
                    #print(b0, b1)
                    fwr.append(p_inter[sw][b0][b1])
                elif sw == 0 and b0 == b1:
                    # Backward intersection
                    bwr.append(p_inter[sw][b0][b1])

            # Adjusted conditions for sw == 1
            elif azpix_ml0[2][0] - azpix_ml0[1][0] < -300:
                if sw == 1 and b0 + 2 == b1:
                    #print(b0, b1)
                    fwr.append(p_inter[sw][b0][b1])
                elif sw == 1 and b0 + 1 == b1:
                    # Backward intersection
                    bwr.append(p_inter[sw][b0][b1])
            elif -300 < azpix_ml0[2][0] - azpix_ml0[1][0] < 0:
                if sw == 1 and b0 + 1 == b1:
                    #print(b0, b1)
                    fwr.append(p_inter[sw][b0][b1])
                elif sw == 1 and b0 == b1:
                    # Backward intersection
                    bwr.append(p_inter[sw][b0][b1])

##saving bwr and fwr as pickle##
bwrs=os.path.join(temp_file,frame +'_bwr.pkl')
fwrs=os.path.join(temp_file,frame + '_fwr.pkl')
with open(bwrs, 'wb') as f:
    pickle.dump(bwr, f)

# Saving fwr to a file named 'fwr.pkl'
with open(fwrs, 'wb') as f:
    pickle.dump(fwr, f)



#########calculate the scaling factor and maskint to produce bwr and fwr ifgs
start_time=time.time()

metafile = os.path.join(os.environ['LiCSAR_public'], str(tr), frame, 'metadata', 'metadata.txt')
#primepoch = misc.grep1line('master=',metafile).split('=')[1]
path_to_slcdir = os.path.join(batchdir,frame, 'RSLC', prime)


#Find the index values of empty polygons for 'fwr' and 'bwr'
empty_polygons_indices_fwr = [i for i, polygon in enumerate(fwr) if polygon.is_empty]
empty_polygons_indices_bwr = [i for i, polygon in enumerate(bwr) if polygon.is_empty]

print(f'empty fwr polygons indices:{empty_polygons_indices_fwr}')
print(f'empty bwr polygons indices:{empty_polygons_indices_bwr}')

fwr_filtered = [polygon for i, polygon in enumerate(fwr) if i not in empty_polygons_indices_fwr]
bwr_filtered = [polygon for i, polygon in enumerate(bwr) if i not in empty_polygons_indices_bwr]

#######calculating scaling factors
path_to_slcdir = os.path.join(batchdir,frame, 'RSLC', prime)
sf_array=get_sf_array(path_to_slcdir, f0=5405000500, burst_interval=2.758277)

###reopen phase
dd_pha=np.fromfile(diff_double_mask_temp, np.complex64).byteswap().reshape(az_line, width)


#################
print(BLUE + 'adf_filtering!!!!' + ENDC)


def process_diff_data(dd_pha, polygons_filtered, sf_array, temp_file, pair, suffix, width, az_line):
    double_diff = scaling_before_adf(dd_pha, polygons_filtered, sf_array)

    # Replace NaN in real and imaginary parts with 0
    real_part_nonan = np.nan_to_num(double_diff.real)
    imaginary_part_nonan = np.nan_to_num(double_diff.imag)

    # Combine back into a complex array without NaN values
    double_diff_nonan = real_part_nonan + 1j * imaginary_part_nonan

    # Save the data
    diff_double_mask_path = os.path.join(temp_file, f"{pair}_diff_double_{suffix}")
    double_diff_nonan.astype(np.complex64).byteswap().tofile(diff_double_mask_path)

    # ADF filtering
    diff_double_mask_adf_path = os.path.join(temp_file, f"{pair}_{suffix}_diff_double_mask_adf")
    diff_double_mask_pha_path = os.path.join(temp_file, f"{pair}_{suffix}_diff_double_mask_pha")
    diff_double_mask_coh_path = os.path.join(temp_file, f"{pair}_{suffix}_diff_double_mask_coh")

    # Execute ADF
    exec_str = ['adf', diff_double_mask_path, diff_double_mask_adf_path, diff_double_mask_coh_path, str(width), '1', '-', '-', '-', '-', '-', '-']
    try:
        subprocess.run(exec_str, check=True)
        print(f"ADF executed successfully for {suffix}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing ADF for {suffix}: {e}")

    # Convert complex to real
    exec_str = ['cpx_to_real', diff_double_mask_adf_path, diff_double_mask_pha_path, str(width), '4']
    try:
        subprocess.run(exec_str, check=True)
        print(f"Conversion to real executed successfully for {suffix}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during conversion to real for {suffix}: {e}")

    # Load the phase data
    dd_mask_pha = np.fromfile(diff_double_mask_pha_path, dtype=np.float32).byteswap().reshape(az_line, width)
    print(f"Phase data loaded for {suffix}")

    return dd_mask_pha

# Dictionary to hold the phase data for both suffixes
phase_data = {}

# Loop over both fwr and bwr data
for suffix, polygons_filtered in [('fwr', fwr_filtered), ('bwr', bwr_filtered)]:
    # Process each and store the resulting phase data
    phase_data[suffix] = process_diff_data(dd_pha, polygons_filtered, sf_array, temp_file, pair, suffix, width, az_line)

# Set phase data to NaN where values are 0.00 for both fwr and bwr before merging
phase_data['fwr'][phase_data['fwr'] == 0.0] = np.nan
phase_data['bwr'][phase_data['bwr'] == 0.0] = np.nan
# Start with fwr data as the base
merged_phase_data = np.copy(phase_data['fwr'])

# Create a mask where bwr data is not NaN
mask_bwr = ~np.isnan(phase_data['bwr'])

# Use bwr data to fill in where it has information (not NaN), overriding fwr data
merged_phase_data[mask_bwr] = phase_data['bwr'][mask_bwr]

end_time=time.time()
execution_time = end_time - start_time
print(f"Scaling calculating finished. {execution_time} seconds to run.")


merged_bwr_fwr_soi_phase_m=os.path.join(temp_file, pair + '_merged_soi_phase_m')
merged_phase_data.byteswap().tofile(merged_bwr_fwr_soi_phase_m)

###geocoding progress
print(ORANGE + 'Geocoding starts!!' + ENDC)
time.sleep(5)

###variables for geocoding
lt_fine_suffix='lt_fine'
geo_dir= os.path.join(framedir, 'geo')
if os.path.exists(geo_dir) and os.path.isdir(geo_dir):
  for file in os.listdir(geo_dir):
    if file.endswith(lt_fine_suffix):
      lt_fine_file=os.path.join(geo_dir, file) 

  EQA_path=os.path.join(geo_dir, 'EQA.dem_par')
  EQA_par=pg.ParFile(EQA_path)
  widthgeo=EQA_par.get_value('width', dtype = int, index= 0)
  print(f'widthgeo value is {widthgeo}')

else:
  print(f'geo folder doesnt exists. Please check your {framedir}')


###geocoding progress
print(ORANGE + 'Geocoding....' + ENDC)
time.sleep(5)


for prefix in ['merged']:
    phase_data = os.path.join(str(temp_file), str(pair) + f"_{prefix}_soi_phase_m")
    geoc_file=os.path.join(str(temp_file), str(pair) + f'_{prefix}_soi_geo_phase_m')
    exec_str=['geocode_back', phase_data, str(width), lt_fine_file, geoc_file, str(widthgeo), '0', '0', '0' ]
    try:
      subprocess.run(exec_str, check=True)
      print(f"Command executed successfully: {' '.join(exec_str)}")
    except subprocess.CalledProcessError as e:
      print(f"An error occurred while executing the command: {e}")
        
    geoc_tif=os.path.join(str(temp_file), str(pair) + f'_{prefix}_soi_geo_phase_m.tif')
    exec_str=['data2geotiff', EQA_path, geoc_file,'2', geoc_tif, '0.0' ]
    try:
      subprocess.run(exec_str, check=True)
      print(f"Command executed successfully: {' '.join(exec_str)}")
    except subprocess.CalledProcessError as e:
      print(f"An error occurred while executing the command: {e}")
    
    print(BLUE + f'please run for quick preview: create_preview_wrapped {geoc_tif}'+ENDC)




'''
for i,x in enumerate(sfb):
    for a,y in enumerate(sff):
        if i==a:
            print(x/y)
for x, y in zip(sfb, sff):
    print(x / y)  # Again, assuming x and y are numbers and y is not zero

'''