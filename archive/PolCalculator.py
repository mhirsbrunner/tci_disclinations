import numpy as np
import subprocess
import os

#put the directory where you extracted the files here
cwd = "C:\\Users\\Julian\\Desktop\\Disclination_Surface_Charge_Code\\Disclination_Surface_Charge_Code"
M =  [-2.8,-2, ,-1.2]
Ls = [20]
pvsm = np.zeros(np.size(M)*np.size(Ls))

for i in range(0, np.size(M)):
    gap = np.amin([np.abs(M[i]-3),np.abs(M[i]-1),np.abs(M[i]+1),np.abs(M[i]+3)])
    for j in range(np.size(Ls)):
        Mi = str(M[i])
        msi = str(gap)
        Lj = int(Ls[j])
        res = subprocess.check_output(['python.exe', cwd+'\\Disclination.py', f'--L={Lj}', f'--Lz={Lj}', f'--M={Mi}', f'--ms={msi}'])
        print(res)
        pvsm[i*np.size(Ls)+j] = float(res)

#out1 = np.append([M], [pvsm], axis=0)
np.save(cwd+'\\data\\polar_finiteSizes.npy', pvsm)
print(pvsm)