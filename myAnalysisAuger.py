#ccenv root
import grand.dataio.root_trees as rt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os.path
import numpy as np
#from scipy import optimize
#from scipy import signal
from scipy.stats import norm
import ROOT
import datetime
import pandas as pd

runpath = '/home/olivier/GRAND/data/GP300/argentina/auger/GRANDfiles/'
outpath = '/home/olivier/GRAND/GP300/ana/auger/Aug2023/'

# Following parameters hardcoded for now. Eventually to be read from TRunVoltage and other TTrees.
kadc = 1.8/16384 # Conversion from LSB to V
fsamp = 500e6 # Hz
gaindb = 20 # Hardcoded for now
gainlin = pow(10,gaindb/20) # voltage gain
ib = 1024
t = np.linspace(0,ib/fsamp,ib)*1e6 #mus
nch = 3

def singleTraceProcess(s):
    '''
    Computes FFT, plot signal in time & frequency domains
    '''
    fft=np.abs(np.fft.rfft(s))
    freq=np.fft.rfftfreq(len(s))*fsamp/1e6
    if 0:
        plt.figure()
        plt.subplot(211)
        plt.plot(t, s,'-')
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (ADC)")
        plt.subplot(212)
        plt.plot(freq, fft,'-')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("FFT")
        plt.show()

    return freq, fft

def fillHists(uid,runid,subid):
    '''
    Loads ROOT trees extract relevant info and stores it into numpy arrays saved in npz format
    '''
    print("### Running fillHisto for DU",uid)
    #
    fn = runpath+"/td"+f'{runid:06}'+"_f"+f'{subid:04}'+".root"
    print(f"Reading file ",fn)

    if 0:
      # Print content of ROOT file
      tf = ROOT.TFile(fn)
      for key in tf.GetListOfKeys():
          t = tf.Get(key.GetName())
          try:
             t.Print()
          except:
             print(key+ "tree print() failed.")

    # Now load file
    try:
        df = rt.DataFile(fn)
    except:
        print("Could not find file",fn)
        return
    trun = df.trun
    #print(trun)
    trawv = df.trawvoltage
    tadc = df.tadc
    listevt=trawv.get_list_of_events()  # (evt number, run number)
    nevents = len(listevt)
    print("Run:",runid)
    print("Subrun:",subid)
    tadc.get_event(listevt[0][0],listevt[0][1]) # Evt nb  & run nb of first event
    #print(tadc.get_traces_length())  # Does not work
    ndus = len(trawv.get_list_of_all_used_dus())
    print(ndus,"DUs in run:",tadc.get_list_of_all_used_dus())
    if sum(np.isin(tadc.get_list_of_all_used_dus(),uid)) == 0:
        print('Unit',uid,'not in run',runid,'_',subid,'. Skipping it.')
        return
    # Following parameters to be read from TRunVOltage and other TTrees. To be done.
    #fsamp=tadc.adc_sampling_frequency[0]*1e6 #Hz #
    #print(tadc.adc_input_channels_ch)
    #channels = [k for k in range(len(tadc.adc_enabled_channels_ch[0])) if tadc.adc_enabled_channels_ch[0][k]==True]
    #nch = len(channels)
    # Assume same length for all channels
    #ib = tadc.adc_samples_count_ch[0][1] # Problem with channel 0?? = 192
    print("Traces lengths:",tadc.get_traces_lengths()) # Problem with this function?
    print("VGA gain = ",gainlin,"(",gaindb,"dB)")
    print("Sampling frequency (MHz):",fsamp/1e6)
    print("Trace length (samples):",ib)
    print("Nb of events in run:",nevents)
    print('######')
    print('######')
    #
    # INitialize histos
    trace=np.zeros((len(listevt),nch,ib),dtype='int')  # Array for traces
    sig=[] # Std dev for trace
    sgps=[] # DU second
    mfft = [] # FFT
    battery = [] # FEB voltage level
    temp = [] # FEB temperature
    nev = 0
    # Now looping on events
    for i, v in enumerate(trawv):
        #print("DU IDs in event",i,":",tadc.du_id)
        ind = np.argwhere( np.array(v.du_id) == int(uid))
        if len(ind) == 0:  # target ID not found in this event
          #print("Skipping event",listevt[i][0])
          continue
        ind = int(ind)  # UID index in trace matrix
        #print("DU",uid,"found at index",ind)

        for j in range(nch):
            if len(v.trace_ch[ind][j])==ib:  # Valid size
                # Store trace in dedicated array
                trace[i][j]=np.array(v.trace_ch[ind][j])/1e6/kadc  # Back to LSB
            else:  # Bad event size
                continue

        if nev/100 == int(nev/100):
            print("Event", i, "DU",v.du_id[ind])
            for j in range(nch):
                print("Std dev Ch",j,":",np.std(trace[i][j]))

        # Load arrays
        sgps.append(v.du_seconds[ind])
        sig.append(np.std(trace[i,:,:],axis=1))
        battery.append(v.battery_level[ind])
        temp.append(v.gps_temp[ind])

        # Get FFTs
        freqx, fftx = singleTraceProcess(trace[i][0])
        freqy, ffty = singleTraceProcess(trace[i][1])
        freqz, fftz = singleTraceProcess(trace[i][2])
        if len(mfft)==0:  # Initialize
            mfft = np.array([fftx, ffty, fftz])
        else: #
            mfft = mfft + [fftx, ffty, fftz]  # Stack FFTs
        nev += 1

    # Wrap up arrays
    mfft = np.array(mfft)
    mfft = mfft/nev # Average FFT values
    sgps = np.array(sgps)
    inds = np.argsort(sgps, axis=0)
    sgps = sgps[inds]
    sig = np.array(sig)
    sig = sig[inds]
    battery = np.array(battery)
    battery = battery[inds]
    temp = np.array(temp)
    temp = temp[inds]
    fnpz = outpath + "td"+f'{runid:06}'+"_f"+f'{subid:04}'+"_DU"+str(uid)+".npz"
    if len(sig)>0:
      np.savez(fnpz, sig=sig, time = sgps, freq = freqx, bat = battery, temp = temp, mfft = mfft, trace = trace)
    else:
      print("Empty histos! No event from DU",uid,"in file",fn,"?")

def plotHists(uid,fn):
   '''
   Loads npz files, extracts reduced info and plots it
   '''
   print("### Running plotHistos for DU",uid)
   fn = outpath + fn
   if os.path.isfile(fn) == False:
       print("No file",fn,". Aborting plotHistos.")
       return

   print("Loading file",fn)
   # First extract arrays
   a = np.load(fn)
   time = a['time']
   sig = a['sig']
   battery = a['bat']
   temp = a['temp']
   trace = a['trace']
   freq = a['freq']
   mfft = a['mfft']
   nev = np.shape(sig)[0]
   print("Nb of events for DU",uid,":",nev)
   validt = (time>0) & (time < 1700000000)  # Valid GPS time
   dt = []
   dth = []
   for i in range(nev):
     dt.append(datetime.datetime.fromtimestamp(time[i]))  # Date
     dth.append(datetime.datetime.fromtimestamp(time[i]).hour+datetime.datetime.fromtimestamp(time[i]).minute/60.)  # Hour of day
   dt = np.array(dt)
   dth = np.array(dth)
   dgps = np.diff(time)
   mfft = mfft*kadc # Now to volts
   mfft = mfft/gainlin # Back to voltage @ board input
   #
   # Now to power
   pfft = np.zeros(np.shape(mfft)) # Back to 1MHz step
   for j in range(3):
       pfft[j] = mfft[j]*mfft[j]/ib/ib  # Now normalized to size
   dnu = (freq[1]-freq[0])/1 # MHz/MHz unitless
   pfft = pfft/dnu

   # Now do plots
   plt.rcParams["date.autoformatter.minute"] = "%d - %H:%M"
   tit = fn + "- DU" + str(uid)
   labch = [tit + "- X", tit + "- Y",tit + "- Z"]
   colch = ["blue","orange","green"]

   ## Amplitude histos
   plt.figure(1)
   nbins = 50
   plt.figure(1)
   for j in range(nch):
       alldata = trace[:,j,:].flatten()
       print("Std dev Channel",j,":",np.std(alldata))
       plt.subplot(311+j)
       plt.hist(alldata,nbins,label=labch[j], lw=3)
       plt.yscale('log')
       plt.grid()
       #plt.legend(loc='best')
       if j==nch-1:
           plt.xlabel('All amplitudes (LSB)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "hist_"+str(runid)+"_DU"+str(uid)
   plt.savefig(outpath+f)

   ## FFT
   plt.figure(2)
   gal = np.loadtxt("galaxyrfftX18h.txt",delimiter=',')  # Expected level for Galactic signal
   sel = gal[:,0]<=250
   freq_gal = gal[sel,0]
   sig_galX18 = gal[sel,1]
   u = "(V$^2$/MHz)"
   for j in range(nch):
       indplot = 311+j
       plt.subplot(indplot)
       pfft[j,0:10] = pfft[j,10]
       #plt.subplot(311+i)
       plt.semilogy(freq,pfft[j],label=labch[j])
       plt.xlim(0,max(freq))
       plt.ylabel('FFT' + u + "(VGA corrected)")
       plt.grid()
       if j == 2:
         #plt.legend(loc="best")
         plt.xlabel('Frequency (MHz)')

   plt.subplot(311)
   #plt.semilogy(freq_gal,sig_galX18,label="Galaxy - X@18hLST (obs)")
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "FFT_"+str(runid)+"_DU"+str(uid)
   plt.savefig(outpath+f)
   print("Std dev from FFT @ DAQ level:", np.sqrt(2*np.sum(mfft,axis=1))*gainlin)

   # ## GPS plots
   # plt.figure(3)
   # plt.subplot(221)
   # plt.plot(time,'+-', label=fn)
   # #plt.plot(sgps,'+-', label='GPS second')
   # plt.xlabel('Index')
   # plt.ylabel('Unix time')
   # plt.legend(loc='best')
   # plt.subplot(222)
   # plt.plot(dt[validt],'+-', label=tit)
   # #plt.plot(dts[validt],'+-', label='GPS second (valid)')
   # plt.legend(loc='best')
   # plt.xlabel('Index')
   # plt.ylabel('Date (UTC)')
   # plt.subplot(223)
   # plt.hist(dgps,100)
   # plt.xlabel("$\Delta$ t GPS (s)")
   # plt.subplot(224)
   # plt.plot(tmn[validt],dt[validt],'+-', label=tit)
   # #plt.plot(tmn[validt],dts[validt],'+-', label='GPS second (valid)')
   # plt.legend(loc='best')
   # plt.xlabel('Run duration (min)')
   # plt.ylabel('Date (UTC)')
   # plt.suptitle(tit)
   # mng = plt.get_current_fig_manager()
   # mng.resize(*mng.window.maxsize())
   # f = "GPS_"+str(runid)+"_DU"+str(uid)
   # plt.savefig(f)

   ## Time variation plots
   plt.figure(4)
   plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%HH:%mm'))
   for j in range(nch):
      indplot = 311+j
      plt.subplot(indplot)
      plt.plot(dt[validt],sig[validt,j],'+',label=labch[j])
      #plt.legend(loc='best')
      plt.ylabel('Std dev (LSB)')
      plt.grid()
   plt.xlabel('Date (UTC)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "sig_"+str(runid)+"_DU"+str(uid)+"_oneday"
   plt.savefig(outpath+f)

   ## Time variation plots (folded over day)
   plt.figure(41)
   for j in range(nch):
      indplot = 311+j
      plt.subplot(indplot)
      plt.plot(dth[validt],sig[validt,j],'+',label=labch[j])
      #plt.legend(loc='best')
      plt.xlim([0,24])
      plt.axvspan(11.2, 21.2, facecolor='yellow', alpha=0.5)  # Sun up @ Malargue / Aug 21st
      plt.ylabel('Std dev (LSB)')
      plt.grid()
   plt.xlabel('Time of day (UTC)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "sig_"+str(runid)+"_DU"+str(uid)
   plt.savefig(outpath+f)

   # Temperature & Voltage plots
   plt.figure(5)
   plt.subplot(211)
   plt.plot(dt[validt],temp[validt],'+',label='DU'+ str(uid))
   #plt.legend(loc='best')
   plt.xlabel('Date (UTC)')
   plt.ylabel('FEB temperature (C)')
   plt.grid()
   plt.subplot(212)
   plt.plot(dt[validt],battery[validt],'+',label='DU'+str(uid))
   #plt.legend(loc='best')
   plt.xlabel('Date (UTC)')
   plt.ylabel('Voltage (V)')
   plt.grid()
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "battery_"+str(runid)+"_DU"+str(uid)
   plt.savefig(outpath+f)

   # Temperature & Voltage plots
   # (Wrapped up over one day)
   plt.figure(51)
   plt.subplot(211)
   plt.plot(dth[validt],temp[validt],'+',label='DU'+ str(uid))
   #plt.legend(loc='best')
   plt.xlim([0,24])
   plt.axvspan(11.2, 21.2, facecolor='yellow', alpha=0.5)  # Sun up @ Malargue / Aug 21st
   plt.xlabel('Time of day (UTC)')
   plt.ylabel('FEB temperature (C)')
   plt.grid()
   plt.subplot(212)
   plt.plot(dth[validt],battery[validt],'+',label='DU'+str(uid))
   plt.xlim([0,24])
   plt.axvspan(11.2, 21.2, facecolor='yellow', alpha=0.5)  # Sun up @ Malargue / Aug 21st
   #plt.legend(loc='best')
   plt.xlabel('Time of day (UTC)')
   plt.ylabel('Voltage (V)')
   plt.grid()
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "battery_"+str(runid)+"_DU"+str(uid)+"_oneday"
   plt.savefig(outpath+f)

def buildStats(uid, fn):
    '''
    Computes basic run stats and returns it
    '''
    print("### Running buildStats for DU",uid)
    fn = outpath + fn
    if os.path.isfile(fn) == False:
        print("No file",fn,". Aborting buildStats")
        return np.zeros((17,1))  # Return empty array

    print("Loading file",fn)
    # Extract numpy arrays from npz file
    a = np.load(fn)
    time = a['time']
    sig = a['sig']
    battery = a['bat']
    temp = a['temp']
    trace = a['trace']
    freq = a['freq']
    mfft = a['mfft']
    nentries = len(time)
    _,indn = np.unique(time,return_index=True)  # Identify non-duplicates
    nev = len(indn)
    duplicates = nentries - nev
    # Remove duplicates
    time = time[indn]
    temp = temp[indn]
    battery = battery[indn]
    sig = sig[indn,:]
    # COmput evarious stats
    nhot = np.sum(temp>55) # T over 55°C
    nlowv = np.sum(battery<12) # FEB battery below 12V
    nvlowv = np.sum(battery<11)
    nvvlowv = np.sum(battery<10)
    meansig = np.mean(sig,axis = 0)
    nlow = np.zeros((3,1))
    nhigh = np.zeros((3,1))
    for j in range(nch):
        nlow[j] = sum(sig[:,j]<5)  # Std dev below 5LSB
        nhigh[j] = sum(sig[:,j]>100) # Std dev abov 100LSB

    validt = (time>0) & (time < 1700000000) # Valid GPS second
    nonvalidgps = nev- sum(validt)
    time = time[validt]   #Consider valid GPS time only
    if 0:  # Check plots
        deltat = np.diff(time)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(time-time[0],'+-')
        plt.subplot(212)
        plt.hist(deltat,1000)
        plt.show()
    gpsmin = min(time)
    gpsmax = max(time)
    duration = gpsmax-gpsmin
    # Returns reduced info
    return(runid,subid,nhot,nlowv,nvlowv,nvvlowv,uid,nentries,nev,duplicates,nonvalidgps,gpsmin,gpsmax,duration,meansig,nlow.T[0],nhigh.T[0])

##
## Main
##
buildstat = 0  # Do we want to build reduced run stat?
if buildstat == True:
  list_of_lists = []

runid = 2006  # 4 digits
# Loop on units/runs
for subid in range(2,23):
    #for uid in [58]:
    #for uid in [83, 70, 58, 59, 60, 69, 84, 151]:
    for uid in [59, 60, 70, 84]:
        fn = "td"+f'{runid:06}'+"_f"+f'{subid:04}'+"_DU"+str(uid)+".npz"  # npz file name
        #if 1:
        if os.path.isfile(outpath+fn) == False:
            print("Could not find file",fn,", generating it.")
            fillHists(uid,runid,subid)  # Generate file with output histos
        if buildstat:  # We fetch run stats
            rid,sid,nhot,nlowv,nvlowv,nvvlowv,unid,nentries,nev,duplicates,nonvalidgps,gpsmin,gpsmax,duration,meansig,nlow,nhigh = buildStats(uid,fn)
            if rid>0:
                list_of_lists.append([rid,sid,nhot,nlowv,nvlowv,nvvlowv,unid,nentries,nev,duplicates,nonvalidgps,gpsmin,gpsmax,duration,meansig[0],meansig[1],meansig[2],int(nlow[0]),int(nlow[1]),int(nlow[2]),int(nhigh[0]),int(nhigh[1]),int(nhigh[2])])
        plotHists(uid,fn) # Plot output histos
plt.show()
if buildstat:
    # Save run stats to csv file
    pdlist = pd.DataFrame(list_of_lists, columns=['Run ID', 'Subrun ID', 'T>55°','FEB<12V','FEB<11V','FEB<10V','Unit ID','Nentries','Nevents','Nduplicates','Bad GPS','GPSmin','GPSmax','Dur [s]','mSigX [ADC]','mSigY [ADC]','mSigZ [ADC]','Low X','Low Y','Low Z','High X','High Y','High Z'])
    print(pdlist)
    pdlist.to_csv(outpath+"runstat.csv")
