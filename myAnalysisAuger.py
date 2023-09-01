#ccenv root
import grand.dataio.root_trees as rt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os.path
import glob
import numpy as np
#from scipy import optimize
#from scipy import signal
from scipy.stats import norm
import ROOT
import datetime
import pandas as pd

runpath = '/home/olivier/GRAND/data/GP300/argentina/auger/GRANDfiles/'
outpath = '/home/olivier/GRAND/GP300/ana/auger/Aug2023/'

check = 0
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
    # Following parameters to be read from TRunVoltage and other TTrees. To be done.
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
        ind = int(ind[0])  # UID index in trace matrix
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

def plotHists(uid,ffn):
   '''
   Loads npz files, extracts reduced info and plots it
   '''
   global check
   print("### Running plotHistos for DU",uid)
   fn = outpath + ffn
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
   nev = np.shape(sig)[0]
   print("Nb of events for DU",uid,":",nev)
   deltat = np.insert(np.diff(time), 0, 0., axis=0)  # How ugly is this??
   validt = (time>1690000000) & (time < 1700000000) & (deltat>=0)  # Valid GPS time
   dt = []
   dth = []
   for i in range(nev):
     dt.append(datetime.datetime.fromtimestamp(time[i]))  # Date
     dth.append(datetime.datetime.fromtimestamp(time[i]).hour+datetime.datetime.fromtimestamp(time[i]).minute/60.)  # Hour of day
   dt = np.array(dt)
   dth = np.array(dth)

   if do_all == False: # When treating all data reduce nb of plots to speed up process
     trace = a['trace']
     freq = a['freq']
     mfft = a['mfft']
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
   tit = ffn + "- DU" + str(uid)
   if shortlab == True:
       labch = ["DU" + str(uid), "DU" + str(uid), "DU" + str(uid)]
       if check == 0:  #
           check = subid
   else:
       labch = [tit + "- X", tit + "- Y",tit + "- Z"]
       check = subid

   colch = ["blue","orange","green"]
   coluid = np.array(["blue","orange","green","red","purple","brown","pink","gray"])
   luid = [83, 70, 58, 59, 60, 69, 84, 151]
   cind = np.argwhere( np.array(luid) == int(uid))

   ## Amplitude histos
   if do_all == False:
       plt.figure(1)
       nbins = 50
       plt.figure(1)
       for j in range(nch):
           alldata = trace[:,j,:].flatten()
           print("Std dev Channel",j,":",np.std(alldata))
           plt.subplot(311+j)
           if check == subid:
               plt.hist(alldata,nbins,label=labch[j], lw=3)
           else:
               plt.hist(alldata,nbins, lw=3)
           plt.yscale('log')
           plt.grid()
           #plt.legend(loc='best')
           if j==nch-1:
               plt.xlabel('All amplitudes (LSB)')
       plt.suptitle(tit)
       mng = plt.get_current_fig_manager()
       #mng.resize(*mng.window.maxsize())
       f = "hist_"+str(runid)+"_DU"+str(uid)
       plt.savefig(outpath+f)

       ## FFT
       plt.figure(2)
       if plotgal:
           gal = np.loadtxt("galaxyrfftX18h.txt",delimiter=',')  # Expected level for Galactic signal
           sel = gal[:,0]<=250
           freq_gal = gal[sel,0]
           sig_galX18 = gal[sel,1]
       u = "(V$^2$/MHz)"
       for j in range(nch):
           indplot = 311+j
           plt.subplot(indplot)
           pfft[j,0:10] = pfft[j,10]
           #plt.semilogy(freq,pfft[j],label=labch[j])
           if check == subid:
             plt.semilogy(freq,pfft[j],label=labch[j])
           else:
             plt.semilogy(freq,pfft[j])
           plt.xlim(0,max(freq))
           plt.ylabel('FFT' + u + "(VGA corrected)")
           plt.grid()
           if j == nch-1:
             plt.legend(loc="best")
             plt.xlabel('Frequency (MHz)')

       plt.subplot(311)
       if plotgal:
           plt.semilogy(freq_gal,sig_galX18,label="Galaxy - X@18hLST (obs)")
       plt.suptitle(tit)
       mng = plt.get_current_fig_manager()
       #mng.resize(*mng.window.maxsize())
       f = "FFT_"+str(runid)+"_DU"+str(uid)
       plt.savefig(outpath+f)
       print("Std dev from FFT @ DAQ level:", np.sqrt(2*np.sum(mfft,axis=1))*gainlin)

   ## Time variation plots
   plt.figure(4)
   plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%HH:%mm'))
   for j in range(nch):
      indplot = 311+j
      plt.subplot(indplot)
      if check == subid:
        plt.plot(dt[validt],sig[validt,j],linestyle="None",color=coluid[cind][0][0],marker='+',label=labch[j])
      else:
        plt.plot(dt[validt],sig[validt,j],linestyle="None",color=coluid[cind][0][0],marker='+')
      plt.ylabel('Std dev (LSB)')
      plt.grid()
      if j == nch-1:
          plt.legend(loc='best')
   plt.xlabel('Date (UTC)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   #mng.resize(*mng.window.maxsize())
   f = "sig_"+str(runid)+"_DU"+str(uid)+"_oneday"
   plt.savefig(outpath+f)

   ## Time variation plots (folded over day)
   plt.figure(41)
   for j in range(nch):
      indplot = 311+j
      plt.subplot(indplot)
      if check == subid:
          plt.plot(dth[validt],sig[validt,j],linestyle="None",color=coluid[cind][0][0],marker='+',label=labch[j])
      else:
          plt.plot(dth[validt],sig[validt,j],linestyle="None",color=coluid[cind][0][0],marker='+')
      if j == nch-1:
          plt.legend(loc='best')
      plt.xlim([0,24])
      plt.axvspan(11.2, 21.2, facecolor='yellow', alpha=0.5)  # Sun up @ Malargue / Aug 21st
      plt.ylabel('Std dev (LSB)')
      plt.grid()
   plt.xlabel('Time of day (UTC)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   #mng.resize(*mng.window.maxsize())
   f = "sig_"+str(runid)+"_DU"+str(uid)
   plt.savefig(outpath+f)

   # Temperature & Voltage plots
   plt.figure(5)
   plt.subplot(211)
   if check == subid:
       plt.plot(dt[validt],temp[validt],linestyle="None",color=coluid[cind][0][0],marker='+',label='DU'+ str(uid))
   else:
       plt.plot(dt[validt],temp[validt],linestyle="None",color=coluid[cind][0][0],marker='+')
   plt.legend(loc='best')
   plt.xlabel('Date (UTC)')
   plt.ylabel('FEB temperature (C)')
   plt.grid()
   plt.subplot(212)
   if check == subid:
       plt.plot(dt[validt],battery[validt],linestyle="None",color=coluid[cind][0][0],marker='+',label='DU'+str(uid))
   else:
       plt.plot(dt[validt],battery[validt],linestyle="None",color=coluid[cind][0][0],marker='+')
   plt.legend(loc='best')
   plt.xlabel('Date (UTC)')
   plt.ylabel('Voltage (V)')
   plt.grid()
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   #mng.resize(*mng.window.maxsize())
   f = "battery_"+str(runid)+"_DU"+str(uid)
   plt.savefig(outpath+f)

   # Temperature & Voltage plots
   # (Wrapped up over one day)
   plt.figure(51)
   plt.subplot(211)
   #
   if check == subid:
       plt.plot(dth[validt],temp[validt],linestyle="None",color=coluid[cind][0][0],marker='+',label='DU'+ str(uid))
   else:
       plt.plot(dth[validt],temp[validt],linestyle="None",color=coluid[cind][0][0],marker='+')
   plt.legend(loc='best')
   plt.xlim([0,24])
   plt.axvspan(11.2, 21.2, facecolor='yellow', alpha=0.5)  # Sun up @ Malargue / Aug 21st
   plt.xlabel('Time of day (UTC)')
   plt.ylabel('FEB temperature (C)')
   plt.grid()
   plt.subplot(212)
   if check == subid:
       plt.plot(dth[validt],battery[validt],linestyle="None",color=coluid[cind][0][0],marker='+',label='DU'+str(uid))
   else:
       plt.plot(dth[validt],battery[validt],linestyle="None",color=coluid[cind][0][0],marker='+')
   plt.xlim([0,24])
   plt.axvspan(11.2, 21.2, facecolor='yellow', alpha=0.5)  # Sun up @ Malargue / Aug
   plt.legend(loc='best')
   plt.xlabel('Time of day (UTC)')
   plt.ylabel('Voltage (V)')
   plt.grid()
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   #mng.resize(*mng.window.maxsize())
   f = "battery_"+str(runid)+"_DU"+str(uid)+"_oneday"
   plt.savefig(outpath+f)

def buildStats(uid, fn):
    '''
    Computes basic run stats and returns it
    '''
    print("### Running buildStats for DU",uid)
    fn = outpath + fn
    if os.path.isfile(fn) == False:
        print("No file",fn,". Aborting buildStats.")
        return np.zeros((19,1))  # Return empty array

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
    # Remove non valid time infos
    deltat = np.insert(np.diff(time), 0, 0., axis=0)  # How ugly is this?
    validt = (time>1690000000) & (time < 1700000000) & (deltat>=0)  # Valid GPS time
    nonvalidgps = nev- sum(validt)
    time = time[validt]
    temp = temp[validt]
    battery = battery[validt]
    sig = sig[validt,:]

    # Computes various stats
    nhot = np.sum(temp>55) # T over 55°C
    nlowv = np.sum(battery<12) # FEB battery below 12V
    nvlowv = np.sum(battery<11)
    nvvlowv = np.sum(battery<10)
    meansig = np.mean(sig,axis = 0)
    nlow = np.zeros((3,1))
    nhigh = np.zeros((3,1))
    for j in range(nch):
        nlow[j] = sum(sig[:,j]<10)  # Std dev below 5LSB
        nhigh[j] = sum(sig[:,j]>100) # Std dev abov 100LSB

    if 0:  # Check plots
        deltat = np.diff(time)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(time-time[0],'+-')
        plt.subplot(212)
        plt.hist(deltat,1000)
        plt.show()

    if len(time)>0:
        gpsmin = time[0]  # First event
        gpsmax = time[-1] # Last event
        datemin = datetime.datetime.utcfromtimestamp(gpsmin).strftime('%Y-%m-%d %H:%M:%S')
        datemax = datetime.datetime.utcfromtimestamp(gpsmax).strftime('%Y-%m-%d %H:%M:%S')
        duration = gpsmax-gpsmin
        return(runid,subid,nhot,nlowv,nvlowv,nvvlowv,uid,nentries,nev,duplicates,nonvalidgps,datemin,datemax,gpsmin,gpsmax,duration,meansig,nlow.T[0],nhigh.T[0])
    else: # No valid GPS time
        print("Wrong GPS info. Aborting buildStats.")
        return np.zeros((19,1))  # Return empty array
    # Returns reduced info

def cleanStats(pdlist):
  '''
  Sets same (max) duration for all UIDs in same (run,subrun)
  '''
  print("### cleanStats")
  runs = pdlist["Run ID"]
  subruns = pdlist["Subrun ID"]
  uid = pdlist["Unit ID"]
  for runid in np.unique(runs):  # Loop on runs
      selrun = runs==runid
      for subid in np.unique(subruns[selrun]): # Loop on subruns
        sel = subruns == subid
        thisUID = uid[selrun][sel]   # List of units for this (run,subrun)
        thisMin = pdlist["GPSmin"][selrun][sel]
        thisMax = pdlist["GPSmax"][selrun][sel]
        goodmin = abs(thisMin-np.median(thisMin))<1*3600  # Discard crazy gps times (more than 1h off from median)
        goodmax = abs(thisMax-np.median(thisMax))<12*3600 # Discard crazy gps times (more than 12h off from median)
        if (sum(goodmin)>0) & (sum(goodmax)>0):
          start = min(thisMin[goodmin])
          stop = max(thisMax[goodmax])
          thisDur = stop-start
          startd = datetime.datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
          stopd = datetime.datetime.utcfromtimestamp(stop).strftime('%Y-%m-%d %H:%M:%S')
          # Weird way of doing due to strange handling of copies with pandas for multiple indexing
          # See https://stackoverflow.com/questions/12307099/modifying-a-subset-of-rows-in-a-pandas-dataframe
          pdlist.loc[(pdlist["Run ID"]==runid) & (pdlist["Subrun ID"]==subid),'GPSmin'] = start
          pdlist.loc[(pdlist["Run ID"]==runid) & (pdlist["Subrun ID"]==subid),'GPSmax'] = stop
          pdlist.loc[(pdlist["Run ID"]==runid) & (pdlist["Subrun ID"]==subid),'Start date'] = startd
          pdlist.loc[(pdlist["Run ID"]==runid) & (pdlist["Subrun ID"]==subid),'Stop date'] = stopd
          pdlist.loc[(pdlist["Run ID"]==runid) & (pdlist["Subrun ID"]==subid),'Dur [s]'] = thisDur
        else:
          print("#######"" Error! No valid GPS time in this subrun. Discard.")

  pdlist.to_csv(outpath+"runstat.csv",float_format='%.2f')

def sumStats(cpdlist):
  '''
  Adds up statistics per time slice
  '''
  print("### sumStats")
  if len(cpdlist)==0:
      print("Empty Stats list! Aborting sumStats.")
      return
  sumstats = []
  gpsmin = cpdlist["GPSmin"]
  gpsmax = cpdlist["GPSmax"]
  datemin =  pdlist["Start date"]
  datemax =  pdlist["Stop date"]
  dur = cpdlist["Dur [s]"]
  uid = cpdlist["Unit ID"]
  runs = cpdlist["Run ID"]
  subruns = cpdlist["Subrun ID"]
  nent = cpdlist["Nentries"]
  nev = cpdlist["Nevents"]
  ndup = cpdlist["Nduplicates"]
  highT = cpdlist["T>55°"]
  feb12V = cpdlist["FEB<12V"]
  feb11V = cpdlist["FEB<11V"]
  feb10V = cpdlist["FEB<10V"]
  lowX = cpdlist["Low X"]
  lowY = cpdlist["Low Y"]
  lowZ = cpdlist["Low Z"]
  highX = cpdlist["High X"]
  highY = cpdlist["High Y"]
  highZ = cpdlist["High Z"]
  allIDs = [83, 70, 58, 59, 60, 69, 84, 151]
  d = min(gpsmin)
  while d<max(gpsmax):
      print("######")
      print("######")
      print("Time slice start:",datetime.datetime.utcfromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S'))
      sel = (gpsmin>=d) & (gpsmin<d + 24*3600)  # Select all runs starting in this time slice
      d += 24*3600 # Prepare next time slice
      if len(gpsmin[sel]) ==0:
          print("No data recorded in this time slice.")
          startd = datetime.datetime.utcfromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S')
          stopd = datetime.datetime.utcfromtimestamp(d+24*3600).strftime('%Y-%m-%d %H:%M:%S')
          sumstats.append([startd,stopd,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])  # Add ine with zeros
      else:
          # First compute total acquisition time
          startd = datetime.datetime.utcfromtimestamp(min(gpsmin[sel])).strftime('%Y-%m-%d %H:%M:%S')
          stopd = datetime.datetime.utcfromtimestamp(max(gpsmax[sel])).strftime('%Y-%m-%d %H:%M:%S')
          totdur = np.sum(np.unique(dur[sel]))
          # Same duration for all units in (sub)runs ==> sum of unique() gives total DAQ acquisition time
          allruns = np.unique(runs[sel])
          allsubruns = np.unique(subruns[sel]) # All subruns in this time slice
          print("######")
          print("True period:",startd,stopd)
          print("Duration (hours):",totdur/3600)
          print("Subruns:",allsubruns)
          for i, uuid in enumerate(allIDs):
              seluid = uid[sel] == uuid
              thisNev = sum(nent[sel][seluid])
              if (sum(dur[sel][seluid])>0) & (thisNev>0):
                  print("Unit",uuid, "% livetime:",thisNev*10/totdur)
                  print("Unit",uuid, "% duplicates:",sum(ndup[sel][seluid])/thisNev)
                  print("Unit",uuid, "% T>55:",sum(highT[sel][seluid])/thisNev)
                  print("Unit",uuid, "% FEB supply<12V:",sum(feb12V[sel][seluid])/thisNev)
                  print("Unit",uuid, "% FEB supply<11V:",sum(feb11V[sel][seluid])/thisNev)
                  print("Unit",uuid, "% FEB supply<10V:",sum(feb10V[sel][seluid])/thisNev)
                  print("Unit",uuid, "% low X:",sum(lowX[sel][seluid])/thisNev)
                  print("Unit",uuid, "% low Y:",sum(lowY[sel][seluid])/thisNev)
                  print("Unit",uuid, "% low Z:",sum(lowZ[sel][seluid])/thisNev)
                  print("Unit",uuid, "% high X:",sum(highX[sel][seluid])/thisNev)
                  print("Unit",uuid, "% high Y:",sum(highY[sel][seluid])/thisNev)
                  print("Unit",uuid, "% high Z:",sum(highZ[sel][seluid])/thisNev)

                  # Write reduced stats to list
                  sumstats.append([startd,stopd,totdur/3600,allruns,allsubruns,uuid,sum(nent[sel][seluid]),sum(nev[sel][seluid]), \
                  thisNev*10/totdur,sum(ndup[sel][seluid])/thisNev,sum(highT[sel][seluid])/thisNev, \
                  sum(feb12V[sel][seluid])/thisNev,sum(feb11V[sel][seluid])/thisNev,sum(feb10V[sel][seluid])/thisNev, \
                  sum(lowX[sel][seluid])/thisNev,sum(lowY[sel][seluid])/thisNev,sum(lowZ[sel][seluid])/thisNev, \
                  sum(highX[sel][seluid])/thisNev,sum(highY[sel][seluid])/thisNev,sum(highZ[sel][seluid])/thisNev])
              else:
                  print("Unit",uuid, "% livetime: 0.0")
                  # Fill in table eventhough unit is not present
                  sumstats.append([startd,stopd,totdur/3600,allruns,allsubruns,uuid,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

              # Write list to pandas DataFrame
              daylist = pd.DataFrame(sumstats, columns=['Slice begins', 'Slice ends', 'Live [h]', "Runs","Subruns","Unit ID", "Nentries",\
              'Nevents', '% up', '% duplicates', '%T>55', '% FEB<12V','% FEB<11V','% FEB<10V','%Low X','%Low Y','%Low Z','%High X','%High Y','%High Z'])
  # Save data frame to csv file
  daylist.to_csv(outpath+"daystat.csv",float_format='%.2f')

def plotSumStats():
    '''
    Generate a few basic plots to visualise data from csv file built with sumStats
    '''
    print("### plotSumStats")
    plt.rcParams["date.autoformatter.minute"] = "%yyyy:%mm%%dd"
    a = pd.read_excel(outpath+"fullstat.xlsx") # File built manualy from daystat.csv importation
    startd = a["Slice begins"].to_numpy()
    stopd = a["Slice ends"].to_numpy()
    pup = a["% up"].to_numpy()
    uid = a["Unit ID"].to_numpy()
    lowx = a["%Low X"].to_numpy()
    lowy = a["%Low Y"].to_numpy()
    lowz = a["%Low Z"].to_numpy()
    highx = a["%High X"].to_numpy()
    highy = a["%High Y"].to_numpy()
    highz = a["%High Z"].to_numpy()
    lowbat = a["% FEB<12V"].to_numpy()
    live = a['Live [h]'].to_numpy()
    nev = a['Nevents'].to_numpy()
    dup = a['% duplicates'].to_numpy()
    print("Duration of data taking (days):", sum(np.unique(live))/24)

    plt.figure(1)
    for id in np.unique(uid):
        if id==0:
            continue
        sel0 = (uid == id)
        sel = (uid == id) & (live>5)
        print("DU",id,": nevts =",sum(nev[sel0]),"(","{:.2f}".format(sum(nev[sel])*10/3600/24),"days).")
        plt.figure(1)
        plt.subplot(211)
        plt.plot(startd[sel0],pup[sel0]*live[sel0],'*-',label="DU"+str(id))
        plt.subplot(212)
        plt.plot(startd[sel],pup[sel],'*-',label="DU"+str(id))
        sel = (uid == id) & (live>5) & (pup>0)
        plt.figure(2)
        plt.plot(startd[sel],lowbat[sel],'*-',label="DU"+str(id))
        plt.figure(3)
        plt.subplot(311)
        plt.plot(startd[sel],lowx[sel],'*-',label="X - DU"+str(id))
        plt.subplot(312)
        plt.plot(startd[sel],lowy[sel],'*-',label="Y - DU"+str(id))
        plt.subplot(313)
        plt.plot(startd[sel],lowz[sel],'*-',label="Z - DU"+str(id))
        plt.figure(4)
        plt.subplot(311)
        plt.plot(startd[sel],highx[sel],'*-',label="X - DU"+str(id))
        plt.ylim([0, 1])
        plt.subplot(312)
        plt.plot(startd[sel],highy[sel],'*-',label="Y - DU"+str(id))
        plt.subplot(313)
        plt.plot(startd[sel],highz[sel],'*-',label="Z - DU"+str(id))
        plt.figure(5)
        plt.plot(startd[sel],dup[sel],'*-',label="Z - DU"+str(id))

    # Add labels etc
    plt.figure(1)
    plt.subplot(211)
    plt.plot(startd,live,'o-',label="DAQ")
    plt.ylabel("Live [h]")
    plt.xlabel("Date")
    plt.legend(loc="best")
    plt.subplot(212)
    plt.legend(loc="best")
    plt.ylabel("% up")
    plt.figure(2)
    plt.xlabel("Date")
    plt.ylabel("% FEB<12V")
    plt.legend(loc="best")
    plt.figure(3)
    plt.subplot(311)
    plt.legend(loc="best")
    plt.subplot(312)
    plt.legend(loc="best")
    plt.subplot(313)
    plt.legend(loc="best")
    plt.xlabel("Date")
    plt.suptitle("% low signal")
    plt.figure(4)
    plt.subplot(311)
    plt.legend(loc="best")
    plt.subplot(312)
    plt.legend(loc="best")
    plt.subplot(313)
    plt.legend(loc="best")
    plt.xlabel("Date")
    plt.suptitle("% high signal")
    plt.legend(loc="best")
    plt.figure(5)
    plt.xlabel("Date")
    plt.suptitle("% Duplicates")
    plt.legend(loc="best")
    plt.show()

def dumpToFile(uid,fn):
   '''
   Dump monitoring values to txt file (for Matias
   '''
   print("### Running dumpToFile for DU",uid)
   fn = outpath + fn
   if os.path.isfile(fn) == False:
       print("No file",fn,". Aborting dumpToFile.")
       return

   print("Loading file",fn)
   # First extract arrays
   a = np.load(fn)
   time = np.array(a['time'],dtype=np.int32)
   battery = np.array(a['bat'],dtype=float)
   temp = np.array(a['temp'],dtype=float)
   date = []
   for t in time:
       date.append(datetime.datetime.fromtimestamp(t).strftime("%m/%d/%Y, %H:%M:%S").encode('utf-8'))  # Build human readable date
   with open(outpath+"dump_DU"+str(uid)+".txt", "ab") as f:
       np.savetxt(f, np.vstack([time,temp, battery]).T, fmt=["%d","%3.2f","%3.2f"])
       # Gave up trying to include date in file because of stupid string format
       #np.savetxt(outpath+"dump_DU"+str(uid)+".txt",np.vstack([date,time,temp, battery]).T, fmt=["%32s","%d","%3.2f","%3.2f"])


##
## Main - end of functions
##

# When all is ready simply plot summary statistics
plotSumStats()

# Generic settings
shortlab = True # Show short legends
plotgal = False # Plot galactic signal
buildstat = True  # Do we want to build reduced run stat?
if buildstat == True:
  list_of_lists = []
do_all = True
new_bats = [59, 60, 70, 84]  # Units equipped with new batteries

if do_all:  # Loop on all npz files... These have to be built first!
    all_npz = glob.glob(outpath+'*.npz')
    for fn in all_npz:
        runid = int(fn.split("_")[0][-4:])
        subid = int(fn.split("_")[1][-2:])
        uid = int(fn.split("_")[2].split(".")[0][2:])
        print(runid,subid,uid)
        #if np.isin(uid,new_bats)==0: # UID is not a new batterie
        #    continue
        #if runid!=2008:
        #      continue
        # if uid != 83:
        #      continue
        fn = "td"+f'{runid:06}'+"_f"+f'{subid:04}'+"_DU"+str(uid)+".npz"  # (Re) build npz file name
        if buildstat:  # We fetch run stats
            rid,sid,nhot,nlowv,nvlowv,nvvlowv,unid,nentries,nev,duplicates,nonvalidgps,datemin,datemax,\
            gpsmin,gpsmax,duration,meansig,nlow,nhigh = buildStats(uid,fn)
            if rid>0:
                # Write result of buld stat to list
                list_of_lists.append([rid,sid,nhot,nlowv,nvlowv,nvvlowv,unid,nentries,nev,duplicates,nonvalidgps, \
                datemin,datemax,gpsmin,gpsmax,duration,meansig[0],meansig[1],meansig[2],int(nlow[0]),int(nlow[1]),int(nlow[2]),int(nhigh[0]),int(nhigh[1]),int(nhigh[2])])
        dumpToFile(uid,fn)
        plotHists(uid,fn) # Plot output histos

else:  # Select specific runs & UIDs
    runid = 2007  # 4 digits
    # Loop on units/runs
    #for subid in range(4,17): # 2006
    #for subid in range(2,25): # 2006
    #for subid in range(1,20): # 2007
    #for subid in range(1,9): # 2008
    for subid in [10]:
        #for uid in [58]:
        for uid in [83, 70, 58, 59, 60, 69, 84, 151]:
        #for uid in [59, 60, 70, 84]: # New batteries
            fn = "td"+f'{runid:06}'+"_f"+f'{subid:04}'+"_DU"+str(uid)+".npz"  # npz file name
            #if 1:
            if os.path.isfile(outpath+fn) == False:
                print("Could not find file",fn,", generating it.")
                fillHists(uid,runid,subid)  # Generate file with output histos
            if buildstat:  # We fetch run stats
                rid,sid,nhot,nlowv,nvlowv,nvvlowv,unid,nentries,nev,duplicates,nonvalidgps,datemin,datemax, \
                gpsmin,gpsmax,duration,meansig,nlow,nhigh = buildStats(uid,fn)
                if rid>0:
                    list_of_lists.append([rid,sid,nhot,nlowv,nvlowv,nvvlowv,unid,nentries,nev,duplicates,nonvalidgps, \
                    datemin,datemax,gpsmin,gpsmax,duration,meansig[0],meansig[1],meansig[2],int(nlow[0]),int(nlow[1]),int(nlow[2]),int(nhigh[0]),int(nhigh[1]),int(nhigh[2])])
            dumpToFile(uid,fn)
            plotHists(uid,fn) # Plot output histos
plt.show()

if buildstat:
    # Save run stats to csv file
    rpdlist = pd.DataFrame(list_of_lists, columns=['Run ID', 'Subrun ID', 'T>55°','FEB<12V','FEB<11V','FEB<10V','Unit ID','Nentries','Nevents','Nduplicates','Bad GPS','Start date','Stop date','GPSmin','GPSmax','Dur [s]','mSigX [ADC]','mSigY [ADC]','mSigZ [ADC]','Low X','Low Y','Low Z','High X','High Y','High Z'])
    rpdlist.to_csv(outpath+"rawrunstat.csv",float_format='%.2f')
    cleanStats(rpdlist)  # Adjust run times for all participating DUs
    pdlist = pd.read_csv(outpath+"runstat.csv") # Save result to csv
    sumStats(pdlist)  # Compute daily stats
    daylist = pd.read_csv(outpath+"daystat.csv") # Save result to csv
