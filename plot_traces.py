import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import grand.io.root_trees as rt

'''
## PLOTTING TRACES OF GRAND DATA FILES #
#######################################

GRANDLIB BRANCH: `dev_io_root`

This script reads out a GRAND data file in ROOT format,
to take a quick look at traces, particularly during on-site analyses.
'''

## Parser
#########

parser = argparse.ArgumentParser(description="Plot ADC traces of GRAND events.")

parser.add_argument('--path_to_data_file',
                    dest='path_to_data_file',
                    default='/sps/grand/data/nancay/may2023/ROOTfiles/md000500_f0002.root',
                    type=str,
                    help='Specifies the path of the ROOT data file containing the\
                          traces to plot.')
parser.add_argument('--time_sleep',
                    dest='time_sleep',
                    default=1,
                    type=float,
                    help='Specifies the sleep time between plotting traces of each entry. If 666, only the frame specified as --start_entry is displayed')
parser.add_argument('--start_entry',
                    dest='start_entry',
                    default=0,
                    type=int,
                    help='Specifies the entry to start at.')
parser.add_argument('--NC',
                    dest='n_channel',
                    default=5,
                    type=int,
                    help='Specifies the number of channels to be displayed. The NC first channels will be displayed.\
                    If not specified, the script will find channels with datas')
      
                    
options = parser.parse_args()


PathToDataFile   = options.path_to_data_file
TimeSleep        = options.time_sleep
StartEntry       = options.start_entry
Nchannels        = options.n_channel
ChannelNames     = ['X','Y','Z','meh']

## Read data file and obtain trees
##################################
if not os.path.exists(PathToDataFile):
    raise Exception('File not found:',PathToDataFile)
data_file = PathToDataFile.split('/')[-1]


## Initiate TADC tree and get first TADC entry
##############################################
df   = rt.DataFile(PathToDataFile)
tadc = df.tadc
tadc.get_entry(0)
print(tadc)


## If not entered by user, determines Nchannel the number of channels 
## containing datas. 
## Determines which channels are enabled into 'channels' list
## To do that, we use tadc.adc_enabled_channels_ch[0] that contains, 
## a boolean reflecting the enabling of each channel
#######################################################################

# Extract from tadc which channels are enabled
channels = [k for k in range(len(tadc.adc_enabled_channels_ch[0])) if tadc.adc_enabled_channels_ch[0][k]==True]
# Construct the list of active channels - either by user entry, either automatically
if Nchannels == 5:
    Nchannels = len(channels)
    if Nchannels == 1:
        print('\n There is 1 enabled channel : %s \n' %([ChannelNames[channels[0]]]) )
    else:
        print('\n There are %s enabled channels : %s \n' %(Nchannels,[ChannelNames[channels[k]] for k in range(Nchannels)]) )
else:
    channels = channels[:Nchannels]
    if Nchannels == 1:
        print('The first enabled channel is %s' %([ChannelNames[channels[0]]]) )
    else:
        print('The first %s enabled channels are %s' %(Nchannels,[ChannelNames[channels[k]] for k in range(Nchannels)]) )
    

## Plot the traces
##################
## Idea is to create figure before loop, and redraw at each iteration
#####################################################################

labels = ['Channel {:}'.format(channel) for channel in channels]
colors = ['b','m','r','c']
subplots = []
trace = []

plt.ion()
fig, ax = plt.subplots(Nchannels,1,sharex=True,figsize=(10,10))

##############3
# print(tadc)   #<- for debug instances
################

## subplots/plot definition (have to deal differently if 1 or more channels)
############################################################################
channels = [k for k in range(len(tadc.adc_enabled_channels_ch[0])) if tadc.adc_enabled_channels_ch[0][k]==True]
for i in range(Nchannels):
    # Formating 'trace' list length to match number of channels
    trace.append([])
    
    # Let's trick Python in case we only one channel to subscript the plot axes :
    # in case of a plot, ax is a scalar but in case of a subplot it's a list - so
    # let us make a list of that scalar if only 1 channel
    if Nchannels > 1:
        axs = ax           
    else:
        axs = [ax]

    # Now we can prepare our subplots or our 'subplot'
    splt, = axs[i].plot(tadc.trace_ch[0][channels[i]],label=labels[i],color=colors[i])
    subplots.append(splt)
    axs[i].legend(frameon=True)
            
# Set axis labels
axs[Nchannels-1].set_xlabel('Sample number',fontsize=20)
axs[0].set_ylabel('ADC counts',fontsize=20)
    
for entry in range(StartEntry,tadc.get_number_of_entries()):
        # Load the entry in the tree
        tadc.get_entry(entry)
        print(tadc)
                
        # Set figure title
        title = r'File: {:} | Entry: {:} | Event index: {:} | DU: {:}'
        title = title.format(data_file,entry,tadc.event_id[0],tadc.du_id[0])
        
        for i in range(Nchannels):
            # Get the traces
            trace[i] = tadc.trace_ch[0][channels[i]]
            # Plot the traces in X=NS, Y=EW, Z=UP  and Rescale y axe(s)
            subplots[i].set_ydata(trace[i])
            axs[i].set_ylim([np.min(trace[i]) - 10, np.max(trace[i]) + 10])
            axs[0].set_title(title)
            # Draw figure and flush events for next iterations
            fig.canvas.draw()
            fig.canvas.flush_events()
                
        print('Entry %s' %(entry))       
        if TimeSleep == 666:       # yeah ikr ... but allows some quiet time with our victim
            input('Press Enter\n')
            break
        else:
            # Sleep for some time to show the figure
            time.sleep(TimeSleep)
