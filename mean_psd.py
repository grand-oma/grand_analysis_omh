# system
import os
import glob
import argparse
import datetime

# scipy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# grandlib
import grand.io.root_trees as rt


###########################

'''

#########################################
# PLOTTING MEAN PSD OF GRAND DATA FILES #
#########################################

GRANDLIB BRANCH: `dev_io_root`

This script reads out a GRAND data file in ROOT format.
It computes the mean power spectrum density (PSD) of the traces
in the data file, where each detection unit (DU) is treated
separately. In particular, the PSD are correctly normalized, such
that they can be compared to other data sets/runs. The mean PSDs
are saved in npz files for each DU.

'''

##############################
# SET CUSTOM PLOT PARAMETERS #
##############################

plt.rcParams.update({'axes.labelsize': 30})
plt.rcParams.update({'xtick.labelsize': 25})
plt.rcParams.update({'ytick.labelsize': 25})
plt.rcParams.update({'axes.titlesize': 15})
plt.rcParams.update({'legend.fontsize': 20})
#plt.style.use('/pbs/home/p/pcorrea/tools/matplotlib_style_sans-serif.txt')


###########################
# DEFINE PARSER ARGUMENTS #
###########################

parser = argparse.ArgumentParser(description="Plot mean PSD of ADC traces of GRAND events\
                                              accross multiple files.")

parser.add_argument('--path',
                    dest='path_to_data_dir',
                    default='/sps/grand/data/gp13/may2023/20dB/ROOTfiles/',
                    type=str,
                    help='Specifies the path of the directory containing the\
                          ROOT files to analyse.')

parser.add_argument("--savefig",
                    dest="savefig",
                    action='store_true',
                    help="Save plots in `../plots/` directory.")
parser.set_defaults(savefig=False)

parser.add_argument("--no_show",
                    dest="show_plot",
                    action='store_true',
                    help="Show the plots.")
parser.set_defaults(no_show=False)

parser.add_argument("--file_spec",
                    dest="file_spec",
                    default='*.root',
                    type=str,
                    help="Specify which files to analyze.")

parser.add_argument('--empty_ch',
                    dest='empty_ch',
                    default=0,
                    type=int,
                    help='Specifies which of the 4 ADC channels is not branched.')

parser.add_argument("--period",
                    dest="period",
                    default='all',
                    type=str,
                    help="Perform the analysis for day time (6h-21h) with `day`\
                          or night time (21h-6h) with `night`\
                          or both combined with `all`.")


parse_args = parser.parse_args()

path_to_data_dir  = parse_args.path_to_data_dir
savefig           = parse_args.savefig
show_plot         = not parse_args.show_plot
file_spec         = parse_args.file_spec
empty_ch          = parse_args.empty_ch
period            = parse_args.period

if period not in ['all','day','night']:
    raise Exception('Provide a valid period to analyse the analysis.')


#####################################
# GET ROOT FILES AND DU INFORMATION #
#####################################

if not os.path.exists(path_to_data_dir):
    raise Exception('File not found:',path_to_data_dir)

root_files = sorted( glob.glob(path_to_data_dir+file_spec) )

# Initiate TADC tree of first file to get basic info
# This assumes all files are from the same data run
# and therefore have the same run configuration
# Also ensure that the first file is not empty
number_of_dus = 0
file_idx      = 0
for i, root_file in enumerate(root_files):
    tadc  = rt.TADC(root_file)
    tadc.get_entry(0)

    if len( tadc.get_list_of_all_used_dus() ) > number_of_dus:
        number_of_dus = len( tadc.get_list_of_all_used_dus() )
        file_idx      = i

tadc  = rt.TADC(root_files[file_idx])
tadc.get_entry(0)

# Get the sampling frequency of the traces
sample_freq = tadc.adc_sampling_frequency[0] # [MHz]

# Compute the FFT normalization
n_samples = len( tadc.trace_ch[0][0] )
fft_freq  = np.fft.rfftfreq(n_samples) * 5e2 # [MHz]
fft_norm  = 1. / n_samples**2 / np.diff(fft_freq)[0] # [MHz^-1]

# Sometimes all four channels are enabled, but one is
# not branched
# For GP13 it is always channel 0
mask_ch = np.array(tadc.adc_enabled_channels_ch[0],dtype=bool)
if np.all(mask_ch):
    mask_ch[empty_ch] = False

# Sometimes there seems to be a bug where there are less than
# 3 enabled ADC channels
if len( mask_ch[mask_ch] ) < 3:
    mask_ch = np.ones(4,dtype=bool)
    mask_ch[empty_ch] = False

# Get some useful parameters from the data directory
path_to_data_dir_split = path_to_data_dir.split('/')
array = path_to_data_dir_split[4]
month = path_to_data_dir_split[5]
mode  = path_to_data_dir_split[6]

# Get the UTC time zone of the arrays
if array == 'gp13':
    utc_zone = 8
elif array == 'nancay':
    msg = '\nWARNING: UTC+1 or UTC+2??'
    print(msg)
    utc_zone = 1 
else:
    msg = '\nWARNING: no time zone specified, assuming UTC+0'
    print(msg)
    utc_zone = 0

# Get the time limits for day and night
# The dates are irrelevant, only the time
begin_day = datetime.datetime(1,1,1,6,0,0)
end_day   = datetime.datetime(1,1,1,21,0,0)

# Define the npz directory and file name template 
# to store the computed PSDs
npz_dir           = '/sps/grand/pcorrea/{}/{}/{}/mean_psd/'.format(array,month,mode)
npz_file_template = 'mean_psd_{}_{}_{}_{}_du_{}.npz'.format(period,array,month,mode,'{}')

# Get the list of used DUs
du_list = tadc.get_list_of_all_used_dus()

msg = '\nFiles contain data from following DUs: {}'.format(du_list)
print(msg)


#########################################
# COMPUTE AND PLOT MEAN PSD FOR EACH DU #
#########################################

# Make dictionaries, that way you don't have to loop over all files
# each time you want to analyze a different DU
sum_fft_x    = {}
sum_fft_y    = {}
sum_fft_z    = {}
n_entries_du = {}

for du in du_list:
    sum_fft_x[du]    = 0
    sum_fft_y[du]    = 0
    sum_fft_z[du]    = 0
    n_entries_du[du] = 0

# Loop over all files in run
n_files = len(root_files)
for i, root_file in enumerate(root_files[:]):
    trawv = rt.TRawVoltage(root_file)

    n_entries_file = trawv.get_number_of_entries()

    msg = '\n{}/{}: Looping over {} events in {}'.format(i+1,n_files,n_entries_file,root_file)
    print(msg)

    # Define here for debugging purpose
    local_time = begin_day

    # Loop over all entries in file
    for entry in range(n_entries_file):
        trawv.get_entry(entry)

        # Use the DU as key for the FFT dictionaries
        # NOTE: assumes that there are no coincident events
        #       i.e. only one DU (and trace etc) per event
        du = trawv.du_id[0]

        # Get the GPS time and only keep event if in period
        # of day that is specified
        # Convert to local time
        # NOTE: for old GP13 DUs the GPS did not work correctly.
        if (month == 'may2023' and (not du == 1078)):
            gps_time = trawv.gps_time[0]
            local_time = datetime.datetime.fromtimestamp(gps_time)\
                       + datetime.timedelta(hours=utc_zone)

        if period == 'day'\
           and (local_time.time() < begin_day.time() or local_time.time() >= end_day.time()):
            continue
        if period == 'night'\
           and not (local_time.time() < begin_day.time() or local_time.time() >= end_day.time()):
            continue
    
        # Get the traces
        new_trace_ch = np.array(trawv.trace_ch[0],dtype=object)
        new_trace_ch = new_trace_ch[mask_ch]

        if len(new_trace_ch[0]) != n_samples\
        or len(new_trace_ch[1]) != n_samples\
        or len(new_trace_ch[2]) != n_samples:
            msg  = '\nWARNING for entry {}: {} samples in trace != (3,{})'.format(entry,new_trace_ch.shape,n_samples)
            msg += '\nSkipping...'
            print(msg)
            continue

        # Above a GPS temperature of 55ºC the ADC is unreliable
        # Mostly seen in GP13 data of March 2023
        # Sometimes this yields corrupted GPS data that is <10ºC
        # Keeping data between 10-55ºC keeps all the good data
        # Might become obsolete with furder upgrades to the prototypes
        if (trawv.gps_temp[0] > 55 or trawv.gps_temp[0] < 10)\
        and (array,month) == ('gp13','mar2023'):
            msg  = '\nWARNING for entry {}: gps_temp = {} not between 10 and 55 ºC'.format(entry,trawv.gps_temp[0])
            msg += '\nSkipping...'
            print(msg)
            continue

        n_entries_du[du] += 1

        trace_x = new_trace_ch[0]
        trace_y = new_trace_ch[1]
        trace_z = new_trace_ch[2]
        
        # Get the FFTs
        # Take the square for the PSD
        fft_x = np.abs( np.fft.rfft(trace_x) )**2 # [µV^2]
        fft_y = np.abs( np.fft.rfft(trace_y) )**2 # [µV^2]
        fft_z = np.abs( np.fft.rfft(trace_z) )**2 # [µV^2]
        
        # Sum up contribution
        sum_fft_x[du] += fft_x
        sum_fft_y[du] += fft_y
        sum_fft_z[du] += fft_z


for du in du_list:
    # Compute the normalized mean PSD
    mean_psd_x = fft_norm * sum_fft_x[du]/n_entries_du[du] * 1e-12 # [V^2 MHz^-1]
    mean_psd_y = fft_norm * sum_fft_y[du]/n_entries_du[du] * 1e-12 # [V^2 MHz^-1]
    mean_psd_z = fft_norm * sum_fft_z[du]/n_entries_du[du] * 1e-12 # [V^2 MHz^-1]

    # Save mean PSD in npz file
    npz_file = npz_file_template.format(du)
    np.savez(npz_dir+npz_file,
             du=du,
             n_entries=n_entries_du[du],
             fft_freq=fft_freq,
             mean_psd_x=mean_psd_x,
             mean_psd_y=mean_psd_y,
             mean_psd_z=mean_psd_z)
    msg = '\nSaved npz file: {}'.format(npz_dir+npz_file)
    print(msg)


    #####################
    # PLOT THE MEAN PSD #
    #####################

    if show_plot:
        # Create figure
        fig, ax = plt.subplots(figsize=(10,7))

        # Plot the PSDs for X=NS, Y=EW, Z=UP
        ax.plot(fft_freq,mean_psd_x,label='Channel X',color='b')
        ax.plot(fft_freq,mean_psd_y,label='Channel Y',color='m')
        ax.plot(fft_freq,mean_psd_z,label='Channel Z',color='r')

        # Set axis scales
        ax.set_yscale('log')

        # Set axis labels
        ax.set_xlabel(r'Frequency [MHz]')
        ax.set_ylabel(r'Mean PSD [$\mathrm{V^2~ MHz^{-1}}$]')

        # Set figure title
        #title = title_template.format(du,n_entries_du)
        title = 'Data: {} | DU: {} | Events: {}'.format(path_to_data_dir,du,n_entries_du[du])
        ax.set_title(title)

        # Set figure legend
        ax.legend(frameon=True)

        # Save figure
        if savefig:
            plot_dir  = '../plots/'
            #plot_file = plot_file_template.format(du,n_entries_du)
            plot_file = 'mean_psd_{}_{}_{}_du_{}.png'.format(array,month,mode,du)
            plt.savefig(plot_dir+plot_file,bbox_inches='tight',dpi=150)
            print('Saved figure at',plot_dir+plot_file)

if show_plot:
    plt.show()



