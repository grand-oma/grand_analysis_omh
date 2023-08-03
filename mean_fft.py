# system
import os
import glob
import argparse

# scipy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# grandlib
import grand.io.root_trees as rt


###########################

'''

##########################################
# PLOTTING MEAN FFTS OF GRAND DATA FILES #
##########################################

GRANDLIB BRANCH: `dev_io_root`

This script reads out a GRAND data file in ROOT format.
It computes the mean fast Fourier transform (FFT) of the traces
in the data file, where each detection unit (DU) is treated
separately. This allows us to take a quick look at FFTs,
particularly during on-site analyses.

'''

##############################
# SET CUSTOM PLOT PARAMETERS #
##############################

plt.rcParams.update({'axes.labelsize': 30})
plt.rcParams.update({'xtick.labelsize': 25})
plt.rcParams.update({'ytick.labelsize': 25})
plt.rcParams.update({'axes.titlesize': 20})
plt.rcParams.update({'legend.fontsize': 20})
#plt.style.use('/pbs/home/p/pcorrea/tools/matplotlib_style_sans-serif.txt')

###########################
# DEFINE PARSER ARGUMENTS #
###########################

parser = argparse.ArgumentParser(description="Plot mean FFT of ADC traces of GRAND events\
                                              in one file.")

parser.add_argument('--path',
                    dest='path_to_data_file',
                    default='/sps/grand/data/nancay/may2023/ROOTfiles/md000400_f0001.root',
                    type=str,
                    help='Specifies the path of the ROOT data file containing the\
                          traces to plot.')

parser.add_argument("--savefig",
                    dest="savefig",
                    action='store_true',
                    help="Save plots in `../plots/` directory.")
parser.set_defaults(savefig=False)

parser.add_argument("--do_run",
                    dest="do_run",
                    action='store_true',
                    help="Run script for whole run corresponding to given data file.")
parser.set_defaults(do_run=False)

parse_args = parser.parse_args()

path_to_data_file = parse_args.path_to_data_file
savefig           = parse_args.savefig
do_run            = parse_args.do_run


###################################
# READ DATA FILE AND OBTAIN TREES #
###################################

if not os.path.exists(path_to_data_file):
    raise Exception('File not found:',path_to_data_file)

data_file = path_to_data_file.split('/')[-1]


# Initiate TADC tree
df   = rt.DataFile(path_to_data_file)
tadc = df.tadc
tadc.get_entry(0)



# Check to plot for whole run or only specified file
# Adapt title and filenames for plot accordingly
if do_run:
    run = data_file.split('_')[0]
    print('Making mean FFT for whole run:',run)

    # Get all the files in the run
    data_files = sorted( glob.glob(path_to_data_file.replace(data_file,'') + run + '*.root') )

    title_template     = 'Run: {:}'.format(run) + ' | DU: {:} | Events: {:}'
    plot_file_template = 'mean_fft_' + run + '_du_{:}.png'

else:
    # Only use specified data file
    data_files = [path_to_data_file]

    title_template     = 'File: {:}'.format(data_file) + ' | DU: {:} | Events: {}'
    plot_file_template = 'mean_fft_' + data_file.replace('.root','') + '_du_{:}.png'


# Get the sampling frequency of the traces
sample_freq = tadc.adc_sampling_frequency[0] # [MHz]

# Depending on how you take data, not all ADC channels are the same.
# For GP13 data of March: X, Y, Z correspond to channels 1, 2, 3
# For Nancay data of May: X, Y, Z correspond to channels 0, 1, 2
channels = [1,2,3]
if tadc.trace_ch[0][3] == []:
    channels = [0,1,2]
channels = [0,1,2]

labels = ['Channel {:}'.format(channel) for channel in channels]


#########################################
# COMPUTE AND PLOT MEAN FFT FOR EACH DU #
#########################################

du_list   = tadc.get_list_of_all_used_dus()

print('File contains data from following DUs:',du_list)

# Loop over all DUs
for du in du_list:
    print('Making FFT plot for DU',du)

    sum_fft_0 = 0
    sum_fft_1 = 0
    sum_fft_2 = 0

    n_entries_du = 0

    # Loop over all data_files
    for file in data_files:
        df   = rt.DataFile(file)
        tadc = df.tadc
        tadc.get_entry(0)

        n_entries_file = tadc.get_number_of_entries()

        msg = 'Looping over {} events in {}'.format(n_entries_file,file)
        print(msg)

        # Loop over events in data file
        for entry in range(n_entries_file):
            tadc.get_entry(entry)

            # Only select events with the right DU
            # NOTE: assumes that there are no coincident events
            #       i.e. only one DU (and trace etc) per event
            if du != tadc.du_id[0]:
                continue

            n_entries_du += 1

            # Get the traces
            trace_0 = tadc.trace_ch[0][channels[0]]
            trace_1 = tadc.trace_ch[0][channels[1]]
            trace_2 = tadc.trace_ch[0][channels[2]]

            # Get the FFTs
            fft_0 = np.abs( np.fft.rfft(trace_0) )
            fft_1 = np.abs( np.fft.rfft(trace_1) )
            fft_2 = np.abs( np.fft.rfft(trace_2) )

            # Sum up contribution
            sum_fft_0 += fft_0
            sum_fft_1 += fft_1
            sum_fft_2 += fft_2


    # Compute the mean FFT
    # We ignore correct FFT/PSD normalization in this script
    mean_fft_0 = sum_fft_0/n_entries_du
    mean_fft_1 = sum_fft_1/n_entries_du
    mean_fft_2 = sum_fft_2/n_entries_du


    # Get the frequencies of the FFT (same for all channels)
    fft_freq    = np.fft.rfftfreq( len(trace_0) )*sample_freq # [MHz]


    #####################
    # PLOT THE MEAN FFT #
    #####################

    # Create figure
    fig, ax = plt.subplots(figsize=(10,7))

    # Plot the FFTs for X=NS, Y=EW, Z=UP
    ax.plot(fft_freq,mean_fft_0,label=labels[0],color='b')
    ax.plot(fft_freq,mean_fft_1,label=labels[1],color='m')
    ax.plot(fft_freq,mean_fft_2,label=labels[2],color='r')

    # Set axis scales
    ax.set_yscale('log')

    # Set axis labels
    ax.set_xlabel(r'Frequency [MHz]')
    ax.set_ylabel(r'Mean FFT [a.u.]')

    # Set figure title
    title = title_template.format(du,n_entries_du)
    ax.set_title(title)

    # Set figure legend
    ax.legend(frameon=True)

    # Save figure
    if savefig:
        plot_dir  = '../plots/'
        plot_file = plot_file_template.format(du,n_entries_du)
        plt.savefig(plot_dir+plot_file,bbox_inches='tight',dpi=150)
        print('Saved figure at',plot_dir+plot_file)


# Show all figures
plt.show()


'''
#################
# END OF SCRIPT #
#################
'''
