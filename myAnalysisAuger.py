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
nev = 0
for i in range(nevents):
    tadc.get_event(listevt[i][0],listevt[i][1])

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
dgps = np.diff(gpstime)
sgps= np.array(sgps)
sig = np.array(sig)
sgps = np.array(sgps)
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
if 1:
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
    plt.subplot(311)
    tit = runpath + "- DU" + str(uid)
    plt.title(tit)
    f = "hist_"+str(runid)+"_DU"+str(uid)
    plt.savefig(f)


if 1:
    gal = np.loadtxt("galaxyrfftX18h.txt",delimiter=',')
    sel = gal[:,0]<=250
    freq_gal = gal[sel,0]
    sig_galX18 = gal[sel,1]
    # gal = np.loadtxt("galaxyrfftX0h.txt",delimiter=',')
    # sel = gal[:,0]<=250
    # sig_galX0 = gal[sel,1]
    # gal = np.loadtxt("galaxyrfftX7h.txt",delimiter=',')
    # sel = gal[:,0]<=250
    # sig_galX7 = gal[sel,1]
    # gal = np.loadtxt("galaxyrfftY18h.txt",delimiter=',')
    # sel = gal[:,0]<=250
    # sig_galY18 = gal[sel,1]
    # gal = np.loadtxt("galaxyrfftZ18h.txt",delimiter=',')
    # sel = gal[:,0]<=250
    # sig_galZ18 = gal[sel,1]

#u = "(V$^2$/" + f'{fsamp/1e6/ib0:4.2f}' + "MHz)"
u = "(V$^2$/MHz)"
lab = ["X channel","Y channel","Z channel"]
plt.figure(2)
for j in range(nch):
    pfft[j,0:10] = pfft[j,10]
    #plt.subplot(311+i)
    plt.semilogy(freqx,pfft[j],label=labch[j])
    plt.xlim(0,max(freqx))
    plt.ylabel('FFT' + u + "(VGA corrected)")

plt.plot(freq_gal,sig_galX18,label="Galaxy - X@18hLST (obs)")
#plt.plot(freq_gal,sig_galY18,label="Galaxy - Y@18hLST (sim)")
#plt.plot(freq_gal,sig_galZ18,label="Galaxy - Z@18hLST (sim)")
plt.xlabel('Frequency (MHz)')
plt.legend(loc="best")
plt.title(tit)
f = "FFT_"+str(runid)+"_DU"+str(uid)
plt.savefig(f)
print("Std dev from FFT @ DAQ level:", np.sqrt(2*np.sum(mfft,axis=1))*gainlin/kadc)


plt.figure(3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%MM-%dd %HH:%mm'))
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
plt.ylabel('Date')
plt.subplot(223)
plt.hist(dgps,100)
plt.xlabel("$\Delta$ t GPS (s)")
#plt.plot(dt,'+-', label='GPS time')
#plt.plot(dts,'+-', label='GPS second')
#plt.xlabel('Index')
#plt.ylabel('Date')
#plt.legend(loc='best')
plt.subplot(224)
plt.plot(tmn[validt],dt[validt],'+-', label='GPS time (valid)')
plt.plot(tmn[validt],dts[validt],'+-', label='GPS second (valid)')
plt.legend(loc='best')
plt.xlabel('Run duration (min)')
plt.ylabel('Date')
f = "GPS_"+str(runid)+"_DU"+str(uid)
plt.savefig(f)


# plt.figure(4)
# plt.title("GPS 1")
# plt.subplot(211)
# plt.plot(gpstime[validt],'+-', label='GPS time')
# plt.plot(sgps[validt],'+-', label='GPS second')
# plt.xlabel('Index')
# plt.ylabel('Unix time')
# plt.legend(loc='best')
# plt.subplot(212)
# plt.plot(tmn[validt],gpstime[validt],'+-', label='GPS time')
# plt.plot(tmn[validt],sgps[validt],'+-', label='GPS second')
# plt.xlabel('Run duration (mn)')
# plt.ylabel('Unix time')
# plt.legend(loc='best')
#
# plt.figure(5)
# plt.title("GPS 2")
# plt.plot(tmn[validt],dt[validt],'+',label='GPS time')
# plt.plot(tmn[validt],dts[validt],'+', label='GPS second')
# plt.xlabel('Run duration (mn)')
# plt.ylabel('Date')
# plt.legend(loc='best')

plt.figure(6)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%MM-%dd %HH:%mm'))
plt.subplot(212)
for j in range(nch):
    plt.plot(sig[:,j],label=labch[j])
plt.legend(loc='best')
plt.xlabel('Index')
plt.ylabel('Std dev (LSB)')
plt.grid()
plt.subplot(211)
for j in range(nch):
    plt.plot(dt[validt],sig[validt,j],label=labch[j])
plt.legend(loc='best')
plt.xlabel('Date (UTC)')
plt.ylabel('Std dev (LSB)')
plt.xlim(min(dt[validt]),max(dt[validt]))
plt.grid()
plt.title(tit)
f = "sig_"+str(runid)+"_DU"+str(uid)
plt.savefig(f)

plt.show()

if 0:
    for i in range(3):
        plt.figure(10)
        #plt.subplot(311+i)
        plt.semilogy(freqx,pfft[i],label=lab[i])
        plt.xlim(0,max(freqx))
        plt.ylabel('FFT' + u + "(VGA corrected)")
        if 0:
            if i == 0:
                plt.semilogy(freq_gal,sig_galX7,label="Simulated galaxy - X@7hLST (min)")
                plt.semilogy(freq_gal,sig_galX0,label="Simulated galaxy - X@0hLST (max)")
            if i == 1:
                plt.semilogy(freq_gal,sig_galY18,label="Galaxy - Y@18hLST (sim)")
            if i == 2:
                plt.semilogy(freq_gal,sig_galZ18,label="Galaxy - Z@18hLST (sim)")
        plt.grid()
        plt.xlabel('Frequency (MHz)')
        plt.legend(loc="best")
        plt.title(tit)
        plt.figure(10)
        plt.savefig(tit)

    plt.figure()
    for i in range(3):
        plt.plot(tmn,sig[:,i],".",label=lab[i])
    plt.legend(loc="best")
    plt.ylabel('Std Dev (LSB)')
    plt.xlabel('Time (mn)')
    tit = runpath + "-DU" + str(uid)
    plt.title(tit)
    plt.show()
