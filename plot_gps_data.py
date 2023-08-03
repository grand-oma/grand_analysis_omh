# system
import os
import argparse

# scipy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# grandlib
import grand.io.root_trees as rt


###########################

'''

#############################################
# PLOTTING GPS INFORMATION GRAND DATA FILES #
#############################################

GRANDLIB BRANCH: `dev_io_root`

This script reads out a GRAND data file in ROOT format.
It plots the GPS parameters of all events in a data file,
where each detection unit (DU) is treated separately. 
This allows us to take a quick look at the GPS data,
particularly during on-site analyses.

'''

##############################
# SET CUSTOM PLOT PARAMETERS #
##############################

plt.rcParams.update({'axes.labelsize': 30})
plt.rcParams.update({'xtick.labelsize': 26})
plt.rcParams.update({'ytick.labelsize': 26})
plt.rcParams.update({'axes.titlesize': 15})
plt.rcParams.update({'legend.fontsize': 26})


###########################
# DEFINE PARSER ARGUMENTS #
###########################

parser = argparse.ArgumentParser(description="Plot GPS parameters of ADC traces of GRAND events\
                                              in one file.")

parser.add_argument('--path',
                    dest='path_to_data_file',
                    default='/sps/grand/data/gp13/may2023/ROOTfiles/md000500_f0002.root',
                    type=str,
                    help='Specifies the path of the ROOT data file containing the\
                          traces to plot.')

parser.add_argument("--savefig",
                    dest="savefig",
                    action='store_true',
                    help="Save plots in `../plots/` directory.")
parser.set_defaults(savefig=False)

parse_args = parser.parse_args()

path_to_data_file = parse_args.path_to_data_file
savefig           = parse_args.savefig


###################################
# READ DATA FILE AND OBTAIN TREES #
###################################

if not os.path.exists(path_to_data_file):
    raise Exception('File not found:',path_to_data_file)

data_file = path_to_data_file.split('/')[-1]


# Initiate TRawVoltage tree (to have GPS data in human readable format)
df    = rt.DataFile(path_to_data_file)
trawv = df.trawvoltage
trawv.get_entry(0)


# Depending on how you take data, not all ADC channels are the same.
# For GP13 data of March: X, Y, Z correspond to channels 1, 2, 3
# For Nancay data of May: X, Y, Z correspond to channels 0, 1, 2
channels = [1,2,3]
if trawv.trace_ch[0][3] == []: 
    channels = [0,1,2]

labels = ['Channel {:}'.format(channel) for channel in channels]


#########################################
# COMPUTE AND PLOT GPS DATA FOR EACH DU #
#########################################

n_entries = trawv.get_number_of_entries()
du_list   = trawv.get_list_of_all_used_dus()

print('File contains data from following DUs:',du_list)

# Loop over all DUs
for du in du_list:
    print('Making GPS data plots for DU',du)

    gps_time = []
    gps_temp = []

    du_entries = []

    # Loop over events in data file
    for entry in range(n_entries):
        trawv.get_entry(entry)

        # Only select events with the right DU
        # NOTE: assumes that there are no coincident events
        #       i.e. only one DU (and trace etc) per event
        if du != trawv.du_id[0]:
            continue

        # Store the entry number of event at this DU
        du_entries.append(entry)
      
        # Get the GPS time
        gps_time.append(trawv.gps_time[0])

        # Get the GPS temperature
        gps_temp.append(trawv.gps_temp[0])
    

    #####################
    # PLOT THE GPS TIME #
    #####################

    # Create figure
    fig, ax = plt.subplots(figsize=(10,8))

    # Plot the GPS time
    ax.plot(du_entries,gps_time,color='b')

    # Set axis labels
    ax.set_xlabel(r'Entry',fontsize=20)
    ax.set_ylabel(r'GPS time [s]',fontsize=20)

    # Set figure title
    title = r'File: {:} | DU: {:}'
    title = title.format(data_file,du)
    ax.set_title(title)

    # Save figure
    if savefig:
        plot_dir  = '../plots/'
        plot_file = 'gps_time_' + data_file.replace('.root','') + '_du_{:}.png'.format(du)
        plt.savefig(plot_dir+plot_file,bbox_inches='tight',dpi=150)
        print('Saved figure at',plot_dir+plot_file)


    ############################
    # PLOT THE GPS TEMPERATURE #
    ############################

    # Create figure
    fig, ax = plt.subplots(figsize=(10,8))

    # Plot the GPS temperature
    ax.plot(du_entries,gps_temp,color='b')

    # Set axis labels
    ax.set_xlabel(r'Entry',fontsize=20)
    ax.set_ylabel(r'GPS temperature [${}^{\circ}$C]',fontsize=20)

    # Set figure title
    title = r'File: {:} | DU: {:}'
    title = title.format(data_file,du)
    ax.set_title(title)

    # Save figure
    if savefig:
        plot_dir  = '../plots/'
        plot_file = 'gps_temp_' + data_file.replace('.root','') + '_du_{:}.png'.format(du)
        plt.savefig(plot_dir+plot_file,bbox_inches='tight',dpi=150)
        print('Saved figure at',plot_dir+plot_file)


# Show all figures
plt.show()


'''
#################
# END OF SCRIPT #
#################
'''