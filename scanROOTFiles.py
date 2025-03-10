import uproot
import ROOT
import sys
import matplotlib.pyplot as plt
import numpy as np

f = sys.argv[1]
#df = rt.DataFile(f)
#df.print()
rootdir = "/home/olivier/GRAND/data/GP300/GRANDFiles/"
rootdir = "/home/olivier/GRAND/data/GP300/GP13/onsite/sept2024/"
f = rootdir+f
print(f"Reading file ",f)


rfile = ROOT.TFile(f)
for key in rfile.GetListOfKeys():
    t = rfile.Get(key.GetName())
    try:
       print("######", key)
       t.Print()
    except:
       print(key+ "tree print() failed.")

upfile = uproot.open(f)
print(upfile.keys())
if 1:
    for treename in upfile.keys():
        print("########")
        print(treename[:-2])
        print("########")
        t = uproot.open(f+":"+treename[:-2])
        for b in t.keys():
            a = t[b].array()
            print("Branch",b,", shape=",np.shape(a))
            print(a)
            if 0:
            #len(np.shape(a))==1:
                plt.figure()
                plt.title(b)
                plt.hist(a)
        plt.show()
