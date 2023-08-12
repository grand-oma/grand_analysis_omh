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

# Analysis of 10s data @ Nancay
# July 2022 OMH

kadc = 1.8/16384 # Conversion from LSB to V
showhisto=1
dttrigall = [];
#args are rootfile and number of events to show

print(">python myAnalysisAuger.py file unit_id [Nevents to show]")
if len(sys.argv)==3:
    uid=int(sys.argv[1])
    f=sys.argv[2]
if len(sys.argv)==2:
    uid=sys.argv[1]
    f='/home/olivier/GRAND/data/GP300/argentina/auger/GRANDfiles/td002004_f0001.root'
if len(sys.argv)==1:
    uid=70
    #f='/home/olivier/GRAND/data/GP300/argentina/auger/GRANDfiles/td002000_f0004.root'
    f='/home/olivier/GRAND/data/GP300/argentina/auger/GRANDfiles/td002004_f0001.root'

print(f"Reading file ",f)
#df = rt.DataFile(f)
#df.print()
if 1:
  tf = ROOT.TFile(f)
  for key in tf.GetListOfKeys():
      t = tf.Get(key.GetName())
      try:
         t.Print()
      except:
         print(key+ "tree print() failed.")

df = rt.DataFile(f)
trun = df.trun
#print(trun)
trawv = df.trawvoltage
#trawv.get_entry(0)
tadc = df.tadc
#tadc.get_entry(0)
#print(tadc)
listevt=trawv.get_list_of_events()  # (evt number, run number)
nevents = len(listevt)

runpath = f.split("/")[-1].split(".")[0]
runid=runpath.split("_")[0]
print("Run:",runid)
subrun=runpath.split("_")[-1]
print("Subrun:",subrun)
tadc.get_event(listevt[0][0],listevt[0][1]) # Evt nb  & run nb of first event
fsamp=tadc.adc_sampling_frequency[0]*1e6 #Hz #
#print(tadc.get_traces_length())
ndus = len(trawv.get_list_of_all_used_dus())
print(ndus,"DUs in run:",tadc.get_list_of_all_used_dus())
#print(tadc.adc_input_channels_ch)
channels = [k for k in range(len(tadc.adc_enabled_channels_ch[0])) if tadc.adc_enabled_channels_ch[0][k]==True]
nch = len(channels)
# Assume same length for all channels
ib = tadc.adc_samples_count_ch[0][1] # Problem with channel 0?? = 192
t = np.linspace(0,ib/fsamp,ib)*1e6 #mus
print("Traces lengths:",tadc.get_traces_lengths()) # Problem with this function?
gaindb = 20 # Hardcoded until TRunVoltage exists
gainlin = pow(10,gaindb/20) # voltage gain
print("VGA gain = ",gainlin,"(",gaindb,"dB)")
print("Sampling frequency (MHz):",fsamp/1e6)
print("Trace length (samples):",ib)
print("Nb of events in run:",nevents)
print('######')
print('######')


N = 6
def singleTraceProcess(s):
    th = N*np.std(s)
    ipk = s>th
    ttrig = t[ipk]  # Pulses above threshold
    xtrig = s[ipk]
    dttrig = np.diff(ttrig) # Time differences
    #if len(xtrig)>0:
    if 0:
        #dttrig = np.insert(dttrig, 0, [ttrig[0]]) # Insert time delay to beginning of trace
        plt.figure()
        plt.plot(t,s,'-')
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (ADC)")
        plt.show()
    itrue = dttrig>0.20 #
    #print(ttrig,dttrig,itrue)
    #dttrigall = np.append(dttrigall,np.diff(ttrig[itrue]))
    ntrigs = sum(itrue)
    fft=np.abs(np.fft.rfft(s))
    freq=np.fft.rfftfreq(len(s))*fsamp/1e6
    return ntrigs, freq, fft

def fillHists(uid,fn):
    print("### Running fillHisto for DU",uid)
    trace=np.zeros((len(listevt),nch,ib),dtype='int')
    sig=[] # Std dev for trace
    gpstime=[] # GPS time
    sgps=[] # DU second
    mfft = [] # FFT
    battery = [] # Battery level
    evtido = [] # Evt nb
    nev = 0
    for i, v in enumerate(trawv):
        #print("DU IDs in event",i,":",tadc.du_id)
        ind = np.argwhere( np.array(v.du_id) == int(uid))
        if len(ind) == 0:  # target ID not found in this event
          print("Skipping event",listevt[i][0])
          continue
        ind = int(ind)  # UID index in trace matrix
        #print("DU",uid,"found at index",ind)

        for j in range(nch):
            if len(v.trace_ch[ind][j])>0:
                trace[i][j]=np.array(v.trace_ch[ind][j])/1e6/kadc  # Back to LSB

        #print("Event", i, "nCh:",np.shape(evt.trace_0)[0])
        if nev/100 == int(nev/100):
            print("Event", i, "DU",v.du_id[ind])
            for j in range(nch):
                print("Std dev Ch",j,":",np.std(trace[i][j]))

        # Get standard deviation
        gpstime.append(v.du_seconds[ind])
        sgps.append(v.du_seconds[ind])
        sig.append(np.std(trace[i,:,:],axis=1))
        battery.append(v.battery_level[ind])
        evtido.append(listevt[i][0])

        # Analyse traces
        ntrigsx, freqx, fftx = singleTraceProcess(trace[i][0])
        ntrigsy, freqy, ffty = singleTraceProcess(trace[i][1])
        ntrigsz, freqz, fftz = singleTraceProcess(trace[i][2])
        if len(mfft)==0:  # Initialize
            mfft = np.array([fftx, ffty, fftz])
            ntrigs = np.array([ntrigsx, ntrigsy,ntrigsz])
        else:
            mfft = mfft + [fftx, ffty, fftz]
            ntrigs = ntrigs + [ntrigsx, ntrigsy,ntrigsz]
        nev += 1
    gpstime= np.array(gpstime)
    inds = np.argsort(gpstime, axis=0)
    gpstime = gpstime[inds]
    sig = np.array(sig)
    sig = sig[inds]
    battery = np.array(battery)
    if len(sig)>0:
      np.savez(fn, sig=sig, time = gpstime, freq = freqx, bat = battery, mfft = mfft/nev, trace = trace)
    else:
      print("Empty histos! No event from DU",uid,"in file",f,"?")
    #return sig,gpstime,battery,freqx,mfft,trace

def plotHists(uid,fn):
   print("### Running plotHistos for DU",uid)
   if os.path.isfile(fn) == False:
       print("No file",fn,". Aborting plotHistos")
       return
   a = np.load(fn)
   time = a['time']
   sig = a['sig']
   battery = a['bat']
   trace = a['trace']
   freq = a['freq']
   mfft = a['mfft']
   nev = np.shape(sig)[0]
   print("Nb of events for DU",uid,":",nev)
   validt = (time>0) & (time < 1692000000)
   print(sum(validt),len(validt),nev)
   tmn = (time-time[0])/60
   dt = []
   for i in range(nev):
     dt.append(datetime.datetime.fromtimestamp(time[i]))
   dt = np.array(dt)
   dgps = np.diff(time)
   mfft = mfft*kadc # Now to volts
   mfft = mfft/gainlin # Back to board input
   #
   # Now power
   pfft = mfft # Back to 1MHz step
   for i in range(3):
       pfft[i] = mfft[i]*mfft[i]/ib/ib  # Now normalized to size
   dnu = (freq[1]-freq[0])/1 # MHz/MHz unitless
   pfft = pfft/dnu

   # Now do plots
   plt.rcParams["date.autoformatter.minute"] = "%d - %H:%M"
   tit = runpath + "- DU" + str(uid)
   labch = ["DU"+str(uid)+"- X","DU"+str(uid)+ "- Y","DU"+str(uid)+"- Z"]
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
       plt.legend(loc='best')
       if j==nch-1:
           plt.xlabel('All amplitudes (LSB)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "hist_"+str(runid)+"_DU"+str(uid)
   plt.savefig(f)

   ## FFT
   plt.figure(2)
   gal = np.loadtxt("galaxyrfftX18h.txt",delimiter=',')
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
       plt.legend(loc="best")
       plt.grid()
       if j == 3:
         plt.xlabel('Frequency (MHz)')

   #plt.subplot(311)
   #plt.semilogy(freq_gal,sig_galX18,label="Galaxy - X@18hLST (obs)")
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "FFT_"+str(runid)+"_DU"+str(uid)
   plt.savefig(f)
   print("Std dev from FFT @ DAQ level:", np.sqrt(2*np.sum(mfft,axis=1))*gainlin)

   ## GPS plots
   plt.figure(3)
   plt.subplot(221)
   plt.plot(time,'+-', label='DU'+str(uid))
   #plt.plot(sgps,'+-', label='GPS second')
   plt.xlabel('Index')
   plt.ylabel('Unix time')
   plt.legend(loc='best')
   plt.subplot(222)
   plt.plot(dt[validt],'+-', label='DU'+str(uid))
   #plt.plot(dts[validt],'+-', label='GPS second (valid)')
   plt.legend(loc='best')
   plt.xlabel('Index')
   plt.ylabel('Date (UTC)')
   plt.subplot(223)
   plt.hist(dgps,100)
   plt.xlabel("$\Delta$ t GPS (s)")
   plt.subplot(224)
   plt.plot(tmn[validt],dt[validt],'+-', label='DU'+str(uid))
   #plt.plot(tmn[validt],dts[validt],'+-', label='GPS second (valid)')
   plt.legend(loc='best')
   plt.xlabel('Run duration (min)')
   plt.ylabel('Date (UTC)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "GPS_"+str(runid)+"_DU"+str(uid)
   plt.savefig(f)

   ## Time variation plots
   plt.figure(4)
   plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%HH:%mm'))
   # for j in range(nch):
   #     plt.plot(sig[:,j],'+',label=labch[j])
   # plt.legend(loc='best')
   # plt.xlabel('Index')
   # plt.ylabel('Std dev (LSB)')
   # plt.grid()
   for j in range(nch):
      indplot = 311+j
      plt.subplot(indplot)
      plt.plot(dt[validt],sig[validt,j],'+',label=labch[j])
      plt.legend(loc='best')
      plt.ylabel('Std dev (LSB)')
      plt.grid()
   #plt.xticks(rotation=45)
   plt.xlabel('Date (UTC)')
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "sig_"+str(runid)+"_DU"+str(uid)
   plt.savefig(f)

   plt.figure(5)
   plt.subplot(211)
   plt.plot(battery,'+',label='DU'+str(uid))
   plt.legend(loc='best')
   plt.xlabel('Index')
   plt.ylabel('Voltage (V)')
   plt.grid()
   plt.subplot(212)
   plt.plot(dt[validt],battery[validt],'+',label='DU'+str(uid))
   #plt.xticks(rotation=45)
   #plt.locator_params(axis='both', nbins=4) # myAnalysisAuger.py:344: UserWarning: 'set_params()' not defined for locator of type <class 'matplotlib.dates.AutoDateLocator'> plt.locator_params(axis='both', nbins=4)
   plt.legend(loc='best')
   plt.xlabel('Date (UTC)')
   plt.ylabel('Voltage (V)')
   plt.grid()
   plt.suptitle(tit)
   mng = plt.get_current_fig_manager()
   mng.resize(*mng.window.maxsize())
   f = "battery_"+str(runid)+"_DU"+str(uid)
   plt.savefig(f)


uid = 83
fns = ["td002002_f0010_DU"+str(uid)+".npz","td002004_f0001_DU"+str(uid)+".npz"]
for fn in fns:
   if os.path.isfile(fn) == False:
     fillHists(uid,fn)  # Generate file with output histos
   plotHists(uid,fn) # Plot output histos
plt.show()

# ### Loop on units/runs
# for uid in [83, 70, 49, 58, 59, 60, 144, 151]:
#     fn = "td002004_f0001_DU"+str(uid)+".npz"
#     if os.path.isfile(fn) == False:
#         fillHists(uid,fn)  # Generate file with output histos
#     plotHists(uid,fn) # Plot output histos
# plt.show()


#Summary stats
#ntrigs = len(dttrigall)
#dur_ev = ib/fsamp
#dur_tot = nevents*dur_ev
#trig_rate = ntrigs/dur_tot
#print(ntrigs,"pulses above",N,"sigmas in",nevents,"timetraces of length",dur_ev*1e6,"mus =",dur_tot,"s.")
#print("Mean = ", ntrigs/nevents,"pulses/trace, Rate = ",trig_rate,"Hz")
