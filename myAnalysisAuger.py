#ccenv root
import grand.dataio.root_trees as rt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
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
if len(sys.argv)>3:
    nshow=int(sys.argv[3])
    uid=int(sys.argv[2])
    f=sys.argv[1]
if len(sys.argv)==3:
    nshow=0
    uid=int(sys.argv[2])
    f=sys.argv[1]
if len(sys.argv)==2:
    nshow=0
    uid=0
    f=sys.argv[1]
if len(sys.argv)==1:
    nshow=0
    uid=0
    #f='/home/olivier/GRAND/data/GP300/argentina/auger/GRANDfiles/td002000_f0004.root'
    f='/home/olivier/GRAND/data/GP300/argentina/auger/GRANDfiles/td002002_f0009.root'

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
tadc = df.tadc
#tadc.get_entry(0)
#print(tadc)
trawv = df.trawvoltage
#trawv.get_entry(0)
listevt=trawv.get_list_of_events()  # (evt number, run number)
nevents = len(listevt)
if nshow>nevents:
    nshow=nevents

runpath = f.split("/")[-1].split(".")[0]
runid=runpath.split("_")[0]
print("Run:",runid)
subrun=runpath.split("_")[-1]
print("Subrun:",subrun)
tadc.get_event(listevt[0][0],listevt[0][1]) # Evt nb  & run nb of first event
fsamp=tadc.adc_sampling_frequency[0]*1e6 #Hz #
#print(tadc.get_traces_length())
ndus = len(tadc.get_list_of_all_used_dus())
print(ndus,"DUs in run:",tadc.get_list_of_all_used_dus())
#print(tadc.adc_input_channels_ch)
  #assuming the same configuration for all events of the run
channels = [k for k in range(len(tadc.adc_enabled_channels_ch[0])) if tadc.adc_enabled_channels_ch[0][k]==True]
nch = len(channels)
# Assume same length for all channels
ib = tadc.adc_samples_count_ch[0][1] # Problem with channel 0?? = 192
print("Traces lengths:",tadc.get_traces_lengths()) # Problem with channel 0?? = 192
#evt.adc_samples_count_channel0[0]
#ib1=evt.adc_samples_count_channel1[0]
#ib2=evt.adc_samples_count_channel2[0]

gaindb = 20 # Hardcoded until TRunVoltage exists
gainlin = pow(10,gaindb/20) # voltage gain
print("VGA gain = ",gainlin,"(",gaindb,"dB)")
print("Sampling frequency (MHz):",fsamp/1e6)
print("Trace length (samples):",ib)
print("Nb of events in run:",nevents)
print("Nb shown:",nshow)
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

t = np.linspace(0,ib/fsamp,ib)*1e6 #mus
trace=np.zeros((len(listevt),nch,ib),dtype='int')
sig=[]
gpstime=[]
sgps=[]
mfft = []
battery = []
nev = 0
for i in range(nevents):
    tadc.get_event(listevt[i][0],listevt[i][1])
    trawv.get_event(listevt[i][0],listevt[i][1])
    #tadc.get_entry(listevt[i][0])
    #trawv.get_entry(listevt[i][0])

    #print("DU IDs in event",i,":",tadc.du_id)
    ind = np.argwhere( np.array(tadc.du_id) == uid)
    if len(ind) == 0:  # target ID not found in this event
      continue
    ind = int(ind)  # UID index in trace matrix
    #print("DU",uid,"found at index",ind)

    for j in range(nch):
        if len(tadc.trace_ch[ind][j])>0:
            trace[i][j]=np.array(tadc.trace_ch[ind][j])

    #print("Event", i, "nCh:",np.shape(evt.trace_0)[0])
    if nev/100 == int(nev/100):
        print("Event", i, "DU",tadc.du_id[ind])
        for j in range(nch):
            print("Std dev Ch",j,":",np.std(trace[i][j]))

    # Get standard deviation
    gpstime.append(tadc.gps_time[ind])
    sgps.append(tadc.du_seconds[ind])
    sig.append(np.std(trace[i,:,:],axis=1))
    battery.append(trawv.battery_level[ind])

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
        # if nev/100 == int(nev/100):
        #     plt.figure(22)
        #     fftp = mftx[0]*kadc/gainlin
        #     fftp = fftp*fftp/ib0/ib0
        #     dnu = (freqx[1]-freqx[0])/1
        #     fftp = fftp/dnu
        #     plt.semilogy(freqx,fftp)

    # except:
    #     print("Exception! Reset at event",i)
    #     nev = 1
    #     plt.figure(22)
    #     fftp = fftx*kadc/gainlin
    #     fftp = fftp*fftp/ib0/ib0
    #     dnu = (freqx[1]-freqx[0])/1
    #     fftp = fftp/dnu
    #     plt.semilogy(freqx,fftp)

print('######')
print('######')
gpstime= np.array(gpstime)
inds = np.argsort(gpstime, axis=0)
gpstime = gpstime[inds]
dgps = np.diff(gpstime)
sgps= np.array(sgps)
sgps = sgps[inds]
sig = np.array(sig)
sig = sig[inds]

validt = (gpstime>0) & (gpstime < 1692000000)
print(sum(validt),len(validt),nev)
tmn = (gpstime-gpstime[0])/60
dt = []
dts = []
for i in range(nev):
  dt.append(datetime.datetime.fromtimestamp(gpstime[i]))
  dts.append(datetime.datetime.fromtimestamp(sgps[i]))
dt = np.array(dt)
dts = np.array(dts)
battery = np.array(battery)

print("Nb of events for DU",uid,":",nev)
mfft = mfft/nev  # mean fft
mfft = mfft*kadc # Now to volts
mfft = mfft/gainlin # Back to board input
#
# Now power
pfft = mfft # Back to 1MHz step
for i in range(3):
    pfft[i] = mfft[i]*mfft[i]/ib/ib  # Now normalized to size
dnu = (freqx[1]-freqx[0])/1 # MHz/MHz unitless
pfft = pfft/dnu

#Summary stats
#ntrigs = len(dttrigall)
dur_ev = ib/fsamp
dur_tot = nevents*dur_ev
trig_rate = ntrigs/dur_tot
print(ntrigs,"pulses above",N,"sigmas in",nevents,"timetraces of length",dur_ev*1e6,"mus =",dur_tot,"s.")
print("Mean = ", ntrigs/nevents,"pulses/trace, Rate = ",trig_rate,"Hz")

labch = ["X channel", "Y channel", "Z channel"]
colch = ["blue","orange","green"]

plt.rcParams["date.autoformatter.minute"] = "%d - %H:%M"
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
# plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.gca().yaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

tit = runpath + "- DU" + str(uid)

## Amplitude histos
plt.figure(1)
nbins = 50
plt.figure(1)
for j in range(nch):
    alldata = trace[:,j,:].flatten()
    print("Std dev Channel",j,":",np.std(alldata))
    plt.subplot(311+j)
    plt.hist(alldata,nbins,label=labch[j], color = "white", ec=colch[j], lw=3)
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
lab = ["X channel","Y channel","Z channel"]
for j in range(nch):
    pfft[j,0:10] = pfft[j,10]
    #plt.subplot(311+i)
    plt.semilogy(freqx,pfft[j],label=labch[j])
    plt.xlim(0,max(freqx))
    plt.ylabel('FFT' + u + "(VGA corrected)")

plt.semilogy(freq_gal,sig_galX18,label="Galaxy - X@18hLST (obs)")
plt.xlabel('Frequency (MHz)')
plt.legend(loc="best")
plt.suptitle(tit)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
f = "FFT_"+str(runid)+"_DU"+str(uid)
plt.savefig(f)
print("Std dev from FFT @ DAQ level:", np.sqrt(2*np.sum(mfft,axis=1))*gainlin/kadc)

## GPS plots
plt.figure(3)
plt.subplot(221)
plt.plot(gpstime,'+-', label='GPS time')
plt.plot(sgps,'+-', label='GPS second')
plt.xlabel('Index')
plt.ylabel('Unix time')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(dt[validt],'+-', label='GPS time (valid)')
plt.plot(dts[validt],'+-', label='GPS second (valid)')
plt.legend(loc='best')
plt.xlabel('Index')
plt.ylabel('Date (UTC)')
plt.subplot(223)
plt.hist(dgps,100)
plt.xlabel("$\Delta$ t GPS (s)")
plt.subplot(224)
plt.plot(tmn[validt],dt[validt],'+-', label='GPS time (valid)')
plt.plot(tmn[validt],dts[validt],'+-', label='GPS second (valid)')
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
plt.subplot(221)
for j in range(nch):
    plt.plot(sig[:,j],'+',label=labch[j])
plt.legend(loc='best')
plt.xlabel('Index')
plt.ylabel('Std dev (LSB)')
plt.grid()
plt.subplot(222)
plt.plot(battery,'+',label='Battery level')
plt.legend(loc='best')
plt.xlabel('Index')
plt.ylabel('Voltage (V)')
plt.grid()
plt.subplot(223)
for j in range(nch):
    plt.plot(dt[validt],sig[validt,j],'+',label=labch[j])
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.xlabel('Date (UTC)')
plt.ylabel('Std dev (LSB)')
plt.xlim(min(dt[validt]),max(dt[validt]))
plt.grid()
plt.subplot(224)
plt.plot(dt[validt],battery[validt],'+',label='Battery level')
plt.xticks(rotation=45)
#plt.locator_params(axis='both', nbins=4) # myAnalysisAuger.py:344: UserWarning: 'set_params()' not defined for locator of type <class 'matplotlib.dates.AutoDateLocator'> plt.locator_params(axis='both', nbins=4)

plt.legend(loc='best')
plt.xlabel('Date (UTC)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.suptitle(tit)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
f = "sig_"+str(runid)+"_DU"+str(uid)
plt.savefig(f)

plt.show()
