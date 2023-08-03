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

##################################################
# CHECKING CONTENT OF TADC AND TRAWVOLTAGE TREES #
##################################################

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
plt.rcParams.update({'xtick.labelsize': 25})
plt.rcParams.update({'ytick.labelsize': 25})
plt.rcParams.update({'axes.titlesize': 20})
plt.rcParams.update({'legend.fontsize': 20})


###########################
# DEFINE PARSER ARGUMENTS #
###########################

parser = argparse.ArgumentParser(description="Plot GPS parameters of ADC traces of GRAND events\
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

parse_args = parser.parse_args()

path_to_data_file = parse_args.path_to_data_file
savefig           = parse_args.savefig


###################################
# READ DATA FILE AND OBTAIN TREES #
###################################

if not os.path.exists(path_to_data_file):
    raise Exception('File not found:',path_to_data_file)

data_file = path_to_data_file.split('/')[-1]


# Initiate TADC and TRawVoltage trees
df    = rt.DataFile(path_to_data_file)
# tadc  = df.tadc
# trawv = df.trawvoltage
tadc  = rt.TADC(path_to_data_file)
trawv = rt.TRawVoltage(path_to_data_file)
tadc.get_entry(0)
trawv.get_entry(0)
#trunv = rt.TRunVoltage(path_to_data_file)

du_list        = tadc.get_list_of_all_used_dus()
n_entries_file = tadc.get_number_of_entries()

msg  = '\nFile contains data from following DUs: {}'.format(du_list)
msg += '\nFile contains {} events'.format(n_entries_file)
print(msg)

for du in du_list:
    msg = '\n>>>>>>>>>> INVESTIGATING DU {} <<<<<<<<<<'.format(du)
    print(msg)

    # Loop over events in data file
    for entry in range(n_entries_file):
        tadc.get_entry(entry)


        # Only select events with the right DU
        # NOTE: assumes that there are no coincident events
        #       i.e. only one DU (and trace etc) per event
        if du != tadc.du_id[0]:
            continue

        last_du_entry = entry
        

        # trunv.adc_conversion
        # trunv.adc_enabled_channels
        # trunv.adc_input_channels
        # trunv.adc_sampling_frequency
        # trunv.adc_sampling_resolution
        # trunv.channel_properties_x
        # trunv.channel_properties_y
        # trunv.channel_properties_z
        # trunv.channel_trig_settings_x
        # trunv.channel_trig_settings_y
        # trunv.channel_trig_settings_z
        # trunv.creation_datetime
        # trunv.digi_ctrl
        # trunv.digi_prepost_trig_windows
        # trunv.file
        # trunv.file_name
        # trunv.firmware_version
        # trunv.gain
        # trunv.modification_history
        # trunv.run_number
        # trunv.trace_length
        # trunv.tree
        # trunv.tree_name
        # trunv.trigger_position
        # trunv.type
    
 
    # Check TADC run parameters
    tadc.get_entry(last_du_entry)

    # acceleration_x                      = tadc.acceleration_x
    # acceleration_y                      = tadc.acceleration_y
    # acceleration_z                      = tadc.acceleration_z

    adc_enabled_channels_ch             = tadc.adc_enabled_channels_ch
    adc_input_channels_ch               = tadc.adc_input_channels_ch
    adc_samples_count_ch                = tadc.adc_samples_count_ch

    # adc_samples_count_channel0          = tadc.adc_samples_count_channel0
    # adc_samples_count_channel1          = tadc.adc_samples_count_channel1
    # adc_samples_count_channel2          = tadc.adc_samples_count_channel2
    # adc_samples_count_channel3          = tadc.adc_samples_count_channel3
    # adc_samples_count_total             = tadc.adc_samples_count_total

    adc_sampling_frequency              = tadc.adc_sampling_frequency
    adc_sampling_resolution             = tadc.adc_sampling_resolution

    base_maximum_ch                     = tadc.base_maximum_ch
    base_minimum_ch                     = tadc.base_minimum_ch
    common_coincidence_time             = tadc.common_coincidence_time
    enable_1PPS                         = tadc.enable_1PPS
    enable_auto_reset_timeout           = tadc.enable_auto_reset_timeout
    enable_DAQ                          = tadc.enable_DAQ
    enable_filter_ch                    = tadc.enable_filter_ch
    enable_readout_ch                   = tadc.enable_readout_ch
    enable_trigger_10s                  = tadc.enable_trigger_10s
    enable_trigger_calibration          = tadc.enable_trigger_calibration
    enable_trigger_ch                   = tadc.enable_trigger_ch
    enable_trigger_ch0_ch1              = tadc.enable_trigger_ch0_ch1
    enable_trigger_ch2_ch3              = tadc.enable_trigger_ch2_ch3
    enable_trigger_external_test_pulse  = tadc.enable_trigger_external_test_pulse
    enable_trigger_notch0_ch1           = tadc.enable_trigger_notch0_ch1
    enable_trigger_redch0_ch1           = tadc.enable_trigger_redch0_ch1
    fire_single_test_pulse              = tadc.fire_single_test_pulse
    firmware_version                    = tadc.firmware_version
    force_firmware_reset                = tadc.force_firmware_reset
    gain_correction_ch                  = tadc.gain_correction_ch
    integration_time_ch                 = tadc.integration_time_ch
    ncmax_ch                            = tadc.ncmax_ch
    ncmin_ch                            = tadc.ncmin_ch
    noise_threshold_ch                  = tadc.noise_threshold_ch
    offset_correction_ch                = tadc.offset_correction_ch
    pre_coincidence_window_ch           = tadc.pre_coincidence_window_ch
    post_coincidence_window_ch          = tadc.post_coincidence_window_ch
    qmax_ch                             = tadc.qmax_ch
    qmin_ch                             = tadc.qmin_ch
    run_number                          = tadc.run_number
    selector_readout_ch                 = tadc.selector_readout_ch
    signal_threshold_ch                 = tadc.signal_threshold_ch
    tcmax_ch                            = tadc.tcmax_ch
    test_pulse_rate_divider             = tadc.test_pulse_rate_divider
    tper_ch                             = tadc.tper_ch
    tprev_ch                            = tadc.tprev_ch
    trigger_pattern_10s                 = tadc.trigger_pattern_10s
    trigger_pattern_calibration         = tadc.trigger_pattern_calibration
    trigger_pattern_ch                  = tadc.trigger_pattern_ch
    trigger_pattern_ch0_ch1             = tadc.trigger_pattern_ch0_ch1
    trigger_pattern_ch2_ch3             = tadc.trigger_pattern_ch2_ch3
    trigger_pattern_external_test_pulse = tadc.trigger_pattern_external_test_pulse
    trigger_pattern_notch0_ch1          = tadc.trigger_pattern_notch0_ch1
    trigger_pattern_redch0_ch1          = tadc.trigger_pattern_redch0_ch1
    trigger_position                    = tadc.trigger_position
    
    

    #trunv.di

    first_du                = tadc.first_du
    time_seconds            = tadc.time_seconds
    time_nanoseconds        = tadc.time_nanoseconds
    event_type              = tadc.event_type
    event_version           = tadc.event_version



    msg  = '\n************************************\n'
    msg += '*** CHECKING TADC RUN PARAMETERS ***\n'
    msg += '************************************\n\n'

    msg += 'RUN NUMBER\n\n'
    msg += 'run_number:                          {:}\n'.format(run_number)
    
    msg += '\n\nDIGITIZER MODE PARAMETERS\n'
    msg += '\n--Digitizer Control--\n'
    msg += 'enable_1PPS:                         {:}\n'.format(enable_1PPS)
    msg += 'enable_auto_reset_timeout:           {:}\n'.format(enable_auto_reset_timeout)
    msg += 'enable_DAQ:                          {:}\n'.format(enable_DAQ)
    msg += 'enable_filter_ch:                    {:}\n'.format(enable_filter_ch)
    msg += 'force_firmware_reset:                {:}\n'.format(force_firmware_reset)
    
    msg += '\n--Trigger Enable Mask--\n'
    msg += 'enable_trigger_10s:                  {:}\n'.format(enable_trigger_10s)
    msg += 'enable_trigger_calibration:          {:}\n'.format(enable_trigger_calibration)
    msg += 'enable_trigger_ch:                   {:}\n'.format(enable_trigger_ch)
    msg += 'enable_trigger_ch0_ch1:              {:}\n'.format(enable_trigger_ch0_ch1)
    msg += 'enable_trigger_ch2_ch3:              {:}\n'.format(enable_trigger_ch2_ch3)
    msg += 'enable_trigger_external_test_pulse:  {:}\n'.format(enable_trigger_external_test_pulse)
    msg += 'enable_trigger_notch0_ch1:           {:}\n'.format(enable_trigger_notch0_ch1)
    msg += 'enable_trigger_redch0_ch1:           {:}\n'.format(enable_trigger_redch0_ch1)

    msg += '\n--Test Pulse Rate Divider & Channel Readout Enable--\n'
    msg += 'enable_readout_ch:                   {:}\n'.format(enable_readout_ch)
    msg += 'test_pulse_rate_divider:             {:}\n'.format(test_pulse_rate_divider)
    
    msg += '\n--Input Selector for Readout Channel--\n'
    msg += 'selector_readout_ch:                 {:}\n'.format(selector_readout_ch)


    msg += '\n\nDIGITIZER TIME WINDOW PARAMETERS\n\n'
    msg += 'common_coincidence_time:             {:} samples of 2 ns\n'.format(common_coincidence_time)
    msg += 'pre_coincidence_window_ch:           {:} samples of 2 ns\n'.format(pre_coincidence_window_ch)
    msg += 'post_coincidence_window_ch:          {:} samples of 2 ns\n'.format(post_coincidence_window_ch)


    msg += '\n\nCHANNEL PROPERTY PARAMETERS\n\n'
    msg += 'gain_correction_ch:                  {:} ? LSB ?\n'.format(gain_correction_ch)
    msg += 'offset_correction_ch:                {:} ? LSB ?\n'.format(offset_correction_ch)
    msg += 'base_maximum_ch:                     {:} ? LSB ?\n'.format(base_maximum_ch)
    msg += 'base_minimum_ch:                     {:} ? LSB ?\n'.format(base_minimum_ch)
    msg += 'integration_time_ch:                 {:} ? ns ?\n'.format(integration_time_ch)


    msg += '\n\nCHANNEL TRIGGER PARAMETERS\n'
    msg += '\n--Trigger Thresholds--\n'
    msg += 'noise_threshold_ch:                  {:}\n'.format(noise_threshold_ch)
    msg += 'signal_threshold_ch:                 {:}\n'.format(signal_threshold_ch)

    msg += '\n\n--Trigger Logic--\n'
    msg += 'tper_ch:                             {:} >> [0,255] x 5 ns\n'.format(tper_ch)
    msg += 'tprev_ch:                            {:} >> [0,255] x 5 ns\n'.format(tprev_ch)
    msg += 'tcmax_ch:                            {:} >> [0,255] x 5 ns\n'.format(tcmax_ch)
    msg += 'ncmax_ch:                            {:} >> [0,255]\n'.format(ncmax_ch)
    msg += 'ncmin_ch:                            {:} >> [0,255]\n'.format(ncmin_ch)
    msg += 'qmax_ch:                             {:} ? LSB ?\n'.format(qmax_ch)
    msg += 'qmin_ch:                             {:} ? LSB ?\n'.format(qmin_ch)


    msg += '\n\nOTHER STUFF\n\n'


    msg += 'adc_enabled_channels_ch:             {:}\n'.format(adc_enabled_channels_ch)
    msg += 'adc_input_channels_ch:               {:}\n'.format(adc_input_channels_ch)
    #msg += 'adc_samples_count_channel0: {:}\n'.format(adc_samples_count_channel0)
    msg += 'adc_samples_count_ch:                {:}\n'.format(adc_samples_count_ch)
    msg += 'adc_sampling_frequency:              {:} MHz\n'.format(adc_sampling_frequency)
    msg += 'adc_sampling_resolution:             {:} LSB\n'.format(adc_sampling_resolution)
 

    
    msg += 'fire_single_test_pulse:              {:}\n'.format(fire_single_test_pulse)
    msg += 'firmware_version:                    {:}\n'.format(firmware_version)
    
    
    
    
    
    
    
    
    msg += 'trigger_position:                    {:}\n'.format(trigger_position)
    msg += 'trigger_pattern_10s:                 {:}\n'.format(trigger_pattern_10s)
    msg += 'trigger_pattern_calibration:         {:}\n'.format(trigger_pattern_calibration)
    msg += 'trigger_pattern_ch:                  {:}\n'.format(trigger_pattern_ch)
    msg += 'trigger_pattern_ch0_ch1:             {:}\n'.format(trigger_pattern_ch0_ch1)
    msg += 'trigger_pattern_ch2_ch3:             {:}\n'.format(trigger_pattern_ch2_ch3)
    msg += 'trigger_pattern_external_test_pulse: {:}\n'.format(trigger_pattern_external_test_pulse)
    msg += 'trigger_pattern_notch0_ch1:          {:}\n'.format(trigger_pattern_notch0_ch1)
    msg += 'trigger_pattern_redch0_ch1:          {:}\n'.format(trigger_pattern_redch0_ch1)
    print(msg)

# msg += 'first_du:                   {:}\n'.format(first_du)
# msg += 'time_seconds:               {:} s\n'.format(time_seconds)
# msg += 'time_nanoseconds:           {:} ns\n'.format(time_nanoseconds)
# msg += 'event_type:                 {:}\n'.format(event_type)
# msg += 'event_version:              {:}\n'.format(event_version)


#msg += 'event_version:      {:}\n'.format(tadc.)
#msg += 'du_count:           {:}\n'.format(tadc.du_count)


# GET EVENT BY EVENT PARAMETERS #

battery_level = []

# Loop over events in data file
for entry in range(n_entries_file):
    tadc.get_entry(entry)

    if tadc.run_number != run_number:
        msg  = 'WARNING: TADC `run_number` not same for all events in file\n'
        msg += 'Entry 0: {}'.format(run_number)
        msg += 'Entry {}: {}'.format(entry,tadc.run_number)
        print(msg)

    # if tadc.first_du != first_du:
    #     msg  = 'WARNING: TADC `first_du` not same for all events in file\n'
    #     msg += 'Entry 0: {}'.format(first_du)
    #     msg += 'Entry {}: {}'.format(entry,tadc.first_du)
    #     print(msg)          
    
    # Get the battery voltages
    battery_level.append(trawv.battery_level)

    # # Get the GPS time
    # gps_time.append(trawv.gps_time[0])

    # # Get the GPS temperature
    # gps_temp.append(trawv.gps_temp[0])


##########################
# PLOT THE BATTERY LEVEL #
##########################

# Create figure
fig, ax = plt.subplots(figsize=(10,8))

# Plot the GPS time
ax.plot(battery_level,color='b')

# Set axis labels
ax.set_xlabel(r'Entry',fontsize=20)
ax.set_ylabel(r'TADC battery level [V]',fontsize=20)

# Set figure title
title = r'File: {:}'
title = title.format(data_file)
ax.set_title(title)

# # Save figure
# if savefig:
#     plot_dir  = '../plots/'
#     plot_file = 'gps_time_' + data_file.replace('.root','') + '_du_{:}.png'.format(du)
#     plt.savefig(plot_dir+plot_file,bbox_inches='tight',dpi=150)
#         print('Saved figure at',plot_dir+plot_file)



plt.show()