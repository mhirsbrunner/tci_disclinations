import numpy as np
import argparse
from scipy import linalg
import matplotlib.pyplot as plt


# parser = argparse.ArgumentParser(description='input a value for L, Lz and M, L must be even')
# parser.add_argument('--L', type=float, help='Value for L')
# parser.add_argument('--Lz', type=float, help='Value for Lz')
# parser.add_argument('--M', type=float, help='Value for M')
# parser.add_argument('--ms', type=float, help='Value for ms')

'''Basic Parameters'''
# L = int(parser.parse_args().L)
# Lz = int(parser.parse_args().Lz)
L = 4
Lz = 16
M = -1.1
ms = .1
Lxy = (3*L**2)//4
hL = L//2
xHop = np.zeros((Lxy * Lz, Lxy * Lz))
yHop = np.zeros((Lxy * Lz, Lxy * Lz))
zHop = np.zeros((Lxy * Lz, Lxy * Lz))
disHop = np.zeros((Lxy * Lz, Lxy * Lz))
Onsite = np.identity(Lxy * Lz)
Sur = np.zeros((Lxy * Lz, Lxy * Lz))
ChePot = np.zeros((Lxy * Lz, Lxy * Lz))
U = np.diag((1, 1j, 1, 1j, -1j, 1, -1j, 1))
diag_1 = np.exp(1j*np.pi/4)*np.exp(1*1j*np.pi/4)
diag_2 = np.exp(-1j*np.pi/4)*np.exp(1*1j*np.pi/4)
#diag_1 = np.exp(1j*np.pi/4)
#diag_2 = np.exp(-1j*np.pi/4)
diag_3 = np.exp(-3j*np.pi/4)
Us = np.diag((diag_1, diag_1, diag_2, diag_2))
Us_2 = np.diag((diag_1, np.conjugate(diag_3), diag_1, np.conjugate(diag_3), diag_2, diag_1, diag_2, diag_1))
S0 = np.array([[1, 0], [0, 1]], float)+1j*np.array([[0, 0], [0, 0]], float)
Sx = np.array([[0, 1], [1, 0]], float)+1j*np.array([[0, 0], [0, 0]], float)
Sy = np.array([[0, -1j], [1j, 0]])+np.array([[0, 0], [0, 0]], float)
Sz = np.array([[1, 0], [0, -1]], float)+1j*np.array([[0, 0], [0, 0]], float)
Gx = np.kron(Sx, S0)
Gy = np.kron(Sy, S0)
Gz = np.kron(Sz, Sz)
G0 = np.kron(Sz, Sx)
G5 = np.kron(Sz, Sy)
I0 = np.kron(S0, S0)
Tx = 1j * Gx + G0
Ty = 1j * Gy + G0
# M = parser.parse_args().M
# ms = parser.parse_args().ms
#print([L,Lz,M,ms])
#print(Gx)
#print(Us)


'''Generate the lattice
for z in range(Lz):
    for y in range(0, 4):
        for x in range(8):
            if x < 7:
                xHop[z * Lxy + 8 * y + x + 1, z * Lxy + 8 * y + x] = 1
            if y < 3:
                yHop[z * Lxy + 8 * (y + 1) + x, z * Lxy + 8 * y + x] = 1
    for y in range(4, 8):
        for x in range(4):
            if x < 3:
                xHop[z * Lxy + 4 * y + x + 16 + 1, z * Lxy + 4 * y + x + 16] = 1
            if y == 4:
                yHop[z * Lxy + 4 * y + x + 16, z * Lxy + 4 * y + x + 16 - 8] = 1
            else:
                yHop[z * Lxy + 4 * y + x + 16, z * Lxy + 4 * y + x + 16 - 4] = 1
    disHop[z * Lxy + 35, z * Lxy + 28] = 1
    disHop[z * Lxy + 39, z * Lxy + 29] = 1
    disHop[z * Lxy + 43, z * Lxy + 30] = 1
    disHop[z * Lxy + 47, z * Lxy + 31] = 1
    if z < Lz - 1:
        for i in range(Lxy):
            zHop[(z + 1) * Lxy + i, z * Lxy + i] = 1'''

'''General Lattice'''
for z in range(Lz):
    for y in range(0, hL):
        for x in range(L):
            if x < L-1:
                xHop[z * Lxy + L * y + x + 1, z * Lxy + L * y + x] = 1
            if y < hL-1:
                yHop[z * Lxy + L * (y + 1) + x, z * Lxy + L * y + x] = 1
    for y in range(hL, L):
        for x in range(hL):
            if x < hL-1:
                xHop[z * Lxy + y * hL + x + L**2//4 + 1, z * Lxy + y * hL + x + L**2//4] = 1
            if y == hL:
                yHop[z * Lxy + y * hL + x + L**2//4, z * Lxy + y * hL + x + L**2//4 - L] = 1
            else:
                yHop[z * Lxy + y * hL + x + L**2//4, z * Lxy + y * hL + x + L**2//4 - hL] = 1
    for i in range(1, hL+1):
        disHop[z*Lxy+(hL-1)*(L+1)+hL*i+hL, z*Lxy+(hL-1)*(L+1)+i] = 1
    if z < Lz - 1:
        for i in range(Lxy):
            zHop[(z + 1) * Lxy + i, z * Lxy + i] = 1


'''Surface Terms'''
for z in range(Lz):
    if z == 0 or z == Lz - 1:
        for i in range(Lxy):
            Sur[z * Lxy + i, z * Lxy + i] = 1
    else:
        for x in range(L):
            Sur[z*Lxy+x, z*Lxy+x] = 1
        for y in range(1, hL):
            Sur[z*Lxy+y*L, z*Lxy+y*L] = 1
            Sur[z*Lxy+y*L+L-1, z*Lxy+y*L+L-1] = 1
        for y in range(hL):
            #Sur[z*Lxy+y*hL+L**2//4, z*Lxy+y*hL+L**2//4] = 1
            Sur[z*Lxy+y*hL+hL*L, z*Lxy+y*hL+hL*L] = 1
        for x in range(hL):
            Sur[z*Lxy+3*L**2//4-hL+x, z*Lxy+3*L**2//4-hL+x] = 1


'''Hamiltonian, Eigenstates and Eigenvalues'''
Hx = 1j * np.kron(xHop, Gx) + np.kron(xHop, G0)
Hy = 1j * np.kron(yHop, Gy) + np.kron(yHop,G0)
Hz = 1j * np.kron(zHop, Gz) + np.kron(zHop,G0)
disH = 1j*np.kron(disHop, np.dot(np.conjugate(Us), Gy)) + np.kron(disHop, np.dot(np.conjugate(Us), G0))
Hop = (Hx + Hy + Hz + disH) / 2.0
H = Hop + np.conj(Hop.T) + M * np.kron(Onsite, G0) + ms * np.kron(Sur, G5)
Eng, Sta = linalg.eig(H)
'''Eng = np.load('C:/Users/Julian/Desktop/Code (1)/data/M_-2.0_ms_0.49999999999999983_L_8_Lz_16_Eng.npy')'''
'''Sta = np.load('C:/Users/Julian/Desktop/Code (1)/data/M_-2.0_ms_0.49999999999999983_L_8_Lz_16_Sta.npy')'''
Eng = Eng.real
#Engsor = np.sort(Eng.real)
#print(Engsor[Lz*Lxy*2-2:Lz*Lxy*2+2])

'''Density Distribution and Polarization'''
FillSta = np.empty([0, 4 * Lxy * Lz], complex)
for i in range(4 * Lxy * Lz):
    if Eng[i] < 0:
        FillSta = np.append(FillSta, [Sta[:, i].T], axis=0)
QFillSta, RFillSta = np.linalg.qr(FillSta.T)
OrthoFillSta = QFillSta.T
FillDen = np.zeros(Lz)
for z in range(Lz):
    for i in range(np.size(OrthoFillSta, 0)):
        V = OrthoFillSta[i, 4*z*Lxy:4*(z+1)*Lxy]
        FillDen[z] = FillDen[z]+(np.vdot(V, V)).real


'''Surface States
Sta0 = np.empty([0, 4 * Lxy * Lz], complex)
for i in range(4 * Lxy * Lz):
    if abs(Eng[i]) < 0.3:
        Sta0 = np.append(Sta0, [Sta[:, i].T], axis=0)
zDen = np.empty([0, Lz])
QSta0, RSta0 = np.linalg.qr(Sta0.T)
OrthoSta0 = QSta0.T
for i in range(np.size(OrthoSta0, 0)):
    Den = np.zeros([1, Lz])
    for z in range(Lz):
        V = OrthoSta0[i, 4 * z * Lxy:4 * (z + 1) * Lxy]
        Den[0, z] = (np.vdot(V, V)).real
    zDen = np.append(zDen, Den, axis=0)'''


'''Output'''
Polar = -np.sum(FillDen[0:(Lz//2)])+np.sum(FillDen[0:Lz])/2
'''Polar = np.sum(FillDen[0:(Lz//2)])'''
#print(FillDen[0:(Lz//2)])
print(Polar)


'''save results'''

'''np.save('C:/Users/Julian/Desktop/Code (1)/data/M_-2.0_ms_0.5_disc_den.npy', FillDen)
outputname_1 = f'spindata/M_{M}_ms_{ms}_L_{L}_Lz_{Lz}_Eng.npy'
outputname_2 = f'spindata/M_{M}_ms_{ms}_L_{L}_Lz_{Lz}_Sta.npy'
np.save(outputname_1, Eng)
np.save(outputname_2, Sta)'''

