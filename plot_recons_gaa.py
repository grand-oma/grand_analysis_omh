import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/olivier/GRAND/soft/l3.mplstyle')
# Taken from Valentin's code
# OMH June 8, 2024

################################################################################
kc = 299792458. #m/s
kc_ns = 299792458.*1.e-9 #m/ns
kn = 1.003
kcn_ns = kc_ns / kn
sig_t = 15
sigma_t = (sig_t*1.e-9*kc)**2 # For the fit chi2 computation
np.set_printoptions(precision=2)
################################################################################
##Read
recons_path = "/home/olivier/GRAND/GP300/ana/auger/may2024/pos_AERA/"
#recons_path = "C:/Users/martinea/Documents/GRAND/GP300/ana/auger/may2024/"
idant, xant, yant, zant = np.loadtxt(recons_path+"coord_antennas.txt").T
_, _, theta_rec, _, phi_rec, _, chi2_plane, _ = np.loadtxt(recons_path+"Rec_plane_wave_recons.txt").T
recoinc, nant, chi2_sphere, _, xrec, yrec, zrec, trec = np.loadtxt(recons_path+"Rec_sphere_wave_recons.txt").T
antid, coincid, peakt, peaka = np.loadtxt(recons_path+"Rec_coinctable.txt").T
tsphere_rec, tplan_rec, peakt_sel, chi2_exp, chi2ndf_delays = [], [], [], [], []
ds_daq, t_datastack = np.zeros((len(nant),5)), np.zeros((len(nant),5)) # Ugly, works for May 2024 data
phi_sph = np.arctan2(xrec,yrec)*180/np.pi
theta_sph = np.arctan2(np.sqrt(xrec**2+yrec**2),zrec)*180/np.pi
print(phi_sph, theta_sph)


antids = [59, 70, 83, 151, 84] # Triggered DUs in April data (time ordered)
uid = np.sort(antids)

#/!\ crappy angular convention correction
phi_rec +=180.
phi_rec = phi_rec%360.
################################################################################
##

# Build variables
for icoinc in np.unique(recoinc):
    sel_coinc = np.where(coincid==icoinc)[0]
    xant_coinc = xant[sel_coinc] ; yant_coinc = yant[sel_coinc] ; zant_coinc = zant[sel_coinc]
    peakt_coinc = peakt[sel_coinc]
    sel_rec = np.where(recoinc==icoinc)[0][0]
    trec_coinc = trec[sel_rec]
    dist_ant_rec = np.linalg.norm(np.array([xant_coinc - xrec[sel_rec], yant_coinc - yrec[sel_rec], zant_coinc - zrec[sel_rec]]), axis=0)
    tsphere_rec.append(dist_ant_rec/kcn_ns)
    peakt_sel.append(peakt_coinc*1.e9)

    k= -np.array([np.cos(phi_rec[sel_rec]*np.pi/180)*np.sin(theta_rec[sel_rec]*np.pi/180), np.sin(phi_rec[sel_rec]*np.pi/180)*np.sin(theta_rec[sel_rec]*np.pi/180), np.cos(theta_rec[sel_rec]*np.pi/180)])
    detPos = np.array([xant_coinc, yant_coinc, zant_coinc]).T
    plandelays = np.dot(detPos,k)/kcn_ns;  # In ns
    #plandelays = plandelays-min(plandelays)
    tplan_rec.append(plandelays)
    #print(detPos)
    #print(plandelays-min(plandelays))


################################################################################
##Plots
plot_folder = './G@Auger_plots/'
if not os.path.exists(plot_folder): os.makedirs(plot_folder)


## Plane wave model
for i in range(len(tsphere_rec)):
    print("Plan DelayPlot - coinc ", i)

    sel_rec = np.where(coincid==i)
    t_data = peakt_sel[i]-min(peakt_sel[i])
    t_model = tplan_rec[i]-min(tplan_rec[i])
    d_model = t_model-t_data
    t_model = t_model-np.mean(d_model)
    d_model = d_model-np.mean(d_model)
    chi2 = sum((d_model)**2)/sig_t**2/(len(t_data)-2)
    chi2ndf_delays.append(chi2)

    #fb, (b1x,b2x) = plt.subplots(2,1)
    fb = plt.figure()
    fb.set_figheight(8)
    fb.set_figwidth(8)
    fb.suptitle(f"Coinc {i:d}")
    b1x = plt.subplot2grid(shape=(3, 1), loc=(0, 0), rowspan=2)
    b2x = plt.subplot2grid(shape=(3, 1), loc=(2, 0), rowspan=1)
    b1x.set_ylabel(r"$\rm t_{\rm model}\ (ns)$")
    b1x.errorbar(t_data, t_model,yerr = sig_t,fmt='.',color='b',label='fit')

    s= 'Plane wave reconstructed at'
    b1x.text(max(t_data)*0.05,max(t_model)*0.9,s,fontsize='x-small')
    s= '$\Theta$ = {0:0.2f}$^o$, $\phi$ = {1:0.2f}$^o$'.format(theta_rec[i],phi_rec[i])
    b1x.text(max(t_data)*0.05,max(t_model)*0.8,s,fontsize='x-small')
    s = '$\chi^2$/ndf = {0:0.2f}'.format(chi2)
    b1x.text(max(t_data)*0.05,max(t_model)*0.7,s,fontsize='x-small')
    print(s)
    for j in range(len(t_data)):
         b1x.text(t_data[j]+50,t_model[j]-0,int(antids[j]),fontsize='x-small')
    b1x.plot([0,max(t_data)],[0,max(t_data)],'--r')
    b1x.grid(True)
    #bx.legend()
    b2x.errorbar(t_data, d_model,yerr = sig_t,fmt='.',color='b',label='fit')
    b2x.grid(True)
    b2x.set_xlabel(r"$\rm t_{\rm mes}\ (ns)$")
    b2x.set_ylabel(r"$\rm t_{\rm model} -t_{\rm mes}\ (ns)$")
    b2x.legend()
    fb.tight_layout()
    fb.savefig(plot_folder+"plandelay_coinc"+str(int(recoinc[i]))+".pdf", format='pdf') #; plt.show()

##
## Now same for the spherical model
for i in range(len(tsphere_rec)):
    print("Spherical DelayPlot - coinc ", i)
    sel_rec = np.where(coincid==i)
    t_data = peakt_sel[i]-min(peakt_sel[i])
    t_model = tsphere_rec[i]-min(tsphere_rec[i])
    d_model = t_model-t_data
    t_model = t_model-np.mean(d_model)
    d_model = d_model-np.mean(d_model)
    #fb, (b1x,b2x) = plt.subplots(2,1)
    fb = plt.figure()
    fb.set_figheight(8)
    fb.set_figwidth(8)
    fb.suptitle(f"Coinc {i:d}")
    b1x = plt.subplot2grid(shape=(3, 1), loc=(0, 0), rowspan=2)
    b2x = plt.subplot2grid(shape=(3, 1), loc=(2, 0), rowspan=1)
    b1x.set_ylabel(r"$\rm t_{\rm model}\ (ns)$")
    b1x.errorbar(t_data, t_model,yerr = sig_t,fmt='.',color='b',label='fit')


    if (len(t_data) == 5) & (chi2<100):
        isort = np.argsort(antid[sel_rec])
        t_datastack[i] = t_data[isort] # Save raw trig time (DU ID ascending order)

    chi2 = sum((d_model)**2)/sig_t**2/(len(t_data)-4)
    chi2ndf_delays.append(chi2)
    s= 'Source reconstructed at'
    b1x.text(max(t_data)*0.05,max(t_model)*0.9,s,fontsize='x-small')
    s= 'SN = {0:0.2f} m, EW = {1:0.2f}m, al t ={2:0.2f}m asl'.format(xrec[i],yrec[i],zrec[i])
    b1x.text(max(t_data)*0.05,max(t_model)*0.85,s,fontsize='x-small')
    s = '$\chi^2$/ndf = {0:0.2f}'.format(chi2)
    b1x.text(max(t_data)*0.05,max(t_model)*0.8,s,fontsize='x-small')
    print(s)
    for j in range(len(t_data)):
         b1x.text(t_data[j]+50,t_model[j]-0,int(antids[j]),fontsize='x-small')
    b1x.plot([0,max(t_data)],[0,max(t_data)],'--r')
    b1x.grid(True)
    #bx.legend()
    b2x.errorbar(t_data, d_model,yerr = sig_t,fmt='.',color='b',label='fit')
    b2x.grid(True)
    b2x.set_xlabel(r"$\rm t_{\rm mes}\ (ns)$")
    b2x.set_ylabel(r"$\rm t_{\rm model} -t_{\rm mes}\ (ns)$")
    b2x.legend()
    fb.tight_layout()
    fb.savefig(plot_folder+"sphdelay_coinc"+str(int(recoinc[i]))+".pdf", format='pdf') #; plt.show()


# Reconstructed positions/directions
fa, (a1x, a2x) = plt.subplots(1, 2)
#fa.suptitle(r"$\rm GRAND@Auger$")
a1x.set_xlabel(r"$\rm Northing\ (m)$")
a1x.set_ylabel(r"$\rm Westing\ (m)$")
a2x.set_xlabel(r"$\rm Northing\ (m)$")
a2x.set_ylabel(r"$\rm Up\ (m)$")
a1x.scatter(xant+.1, yant+.1, label='DUs')
a1x.scatter(xrec, yrec, label='recons')
len_arrow = np.max(np.sqrt(xrec**2+yrec**2))
for i in range(len(xrec)):
    a1x.arrow(xant[1],yant[1],len_arrow*np.cos(phi_rec[i]*np.pi/180),len_arrow*np.sin(phi_rec[i]*np.pi/180))
a1x.legend()
a1x.axis('equal')
#a1x.set_xscale('log')
#a1x.set_yscale('log')
a2x.scatter(xant+.1, zant+.1, label='DUs')
a2x.scatter(xrec, zrec, label='recons')
len_arrow = np.max(np.sqrt(xrec**2+zrec**2))
for i in range(len(xrec)):
    a2x.arrow(xant[1],zant[1],len_arrow*np.cos(phi_rec[i]*np.pi/180)*np.sin(theta_rec[i]*np.pi/180),len_arrow*np.cos(theta_rec[i]*np.pi/180))
a2x.legend()
a2x.axis('equal')
#a2x.set_xscale('log')
#a2x.set_yscale('log')
#plt.show()
fa.savefig(plot_folder+"source_pos_array.pdf", format='pdf')


plt.figure()
plt.plot(chi2_sphere, theta_sph,'+')
plt.savefig(plot_folder+"chi2_theta.pdf", format='pdf')
## All delays
fc, cx = plt.subplots()
#fc.suptitle(r"$\rm GRAND@Auger$")
cx.set_xlabel(r"$\rm t_{\rm data}\ (ns)$")
cx.set_ylabel(r"$\rm t_{\rm PS}\ (ns)$")
[cx.scatter(peakt_sel[i]-min(peakt_sel[i]), tsphere_rec[i]-min(tsphere_rec[i]), label='rec') for i in range(len(tsphere_rec))]
fc.savefig(plot_folder+"delay_all.pdf", format='pdf')

## Histo theta, phi plan
fd, (d1x, d2x) = plt.subplots(1, 2)
#fd.suptitle(r"$\rm GRAND@Auger$")
d1x.set_xlabel(r"$\rm \theta_{\rm rec}\ (deg)$")
d2x.set_xlabel(r"$\rm \phi_{\rm rec}\ (deg)$")
d1x.hist(theta_rec, bins=np.linspace(45,90,45),histtype='step', stacked=True, fill=True,label='PWF')
d1x.hist(theta_sph, bins=np.linspace(45,90,45),histtype='step', stacked=True, fill=True,label='SWF')
d1x.legend()
d2x.hist(phi_rec, bins=np.linspace(0,90,180),histtype='step', stacked=True, fill=True,label='PWF')
d2x.hist(phi_sph, bins=np.linspace(0,90,180),histtype='step', stacked=True, fill=True,label='SWF')
d2x.legend()
fd.savefig(plot_folder+"histo_theta_phi.pdf", format='pdf')

## Histo X,Y,Z sph
fe, (e1x, e2x, e3x) = plt.subplots(1, 3)
fe.suptitle(r"$\rm GRAND@Auger$")
e1x.set_xlabel(r"$\rm x_{\rm source, rec}\ (m)$")
e2x.set_xlabel(r"$\rm y_{\rm source, rec}\ (m)$")
e3x.set_xlabel(r"$\rm z_{\rm source, rec}\ (m)$")
e1x.hist(xrec, bins=int(2*np.sqrt(len(xrec))))
e2x.hist(yrec, bins=int(2*np.sqrt(len(yrec))))
e3x.hist(zrec, bins=int(2*np.sqrt(len(zrec))))
fe.savefig(plot_folder+"histo_x_y_z.pdf", format='pdf')

## Histo CHi2 fit
ff, (f1x, f2x) = plt.subplots(1, 2)
ff.suptitle(r"$\rm GRAND@Auger$")
#f1x.set_xlabel(r"$\rm \chi^2_{\rm plane}$")
#f2x.set_xlabel(r"$\rm \chi^2_{\rm sphere}$")
f1x.set_xlabel(r"$\rm \frac{\chi^2_{\rm plane}}{N_{\rm ant}-2}$")
f2x.set_xlabel(r"$\rm \frac{\chi^2_{\rm sphere}}{N_{\rm ant}-4}$")
#f1x.hist(chi2_plane/sigma_t, bins=int(2*np.sqrt(len(chi2_plane))),label='Plane')
#f2x.hist(chi2_sphere/sigma_t, bins=int(2*np.sqrt(len(chi2_sphere))),label='Sphere')
f1x.hist(chi2_plane/(nant-2)/sigma_t, bins=int(2*np.sqrt(len(chi2_plane))))
f2x.hist(chi2_sphere/(nant-4)/sigma_t, bins=int(2*np.sqrt(len(chi2_sphere))))
ff.savefig(plot_folder+"histo_chi2.pdf", format='pdf') #; plt.show()

plt.figure()
for j in range(1,5):
    plt.subplot(2,2,j)
    plt.gca().set_title("DU"+str(int(uid[j])))
    sel = t_datastack[:,j]>0
    h = plt.hist(t_datastack[sel,j],histtype='step', fill=False)
    s = "mean = {0:0.2f} ns".format(np.mean(t_datastack[sel,j]))
    plt.text(np.mean(t_datastack[sel,j]),0.8*np.max(h[0]),s,fontsize='xx-small')
    s = "std dev = {0:0.2f} ns".format(np.std(t_datastack[sel,j]))
    plt.text(np.mean(t_datastack[sel,j]),0.7*np.max(h[0]),s,fontsize='xx-small')
    plt.xlabel("$t_{mes}$ (ns)")
plt.savefig(plot_folder+"sig_timing.pdf", format='pdf')
#; plt.show()
