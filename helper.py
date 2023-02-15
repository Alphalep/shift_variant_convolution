import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def seidel (p0,q0,x,y,coeffs):
    beta = np.arctan2(q0,p0)
    h2 = np.sqrt(p0**2 + q0**2)
    #rotation of grid
    xr = x*np.cos(beta)+y*np.sin(beta)
    yr = -x*np.sin(beta) + y*np.cos(beta)

    #Seidel Aberration function

    rho2 = xr**2 + yr**2

    W = coeffs[0]*rho2 + coeffs[1]*rho2**2 + coeffs[2]*h2*rho2*xr + coeffs[3]*h2**2*xr**2 + coeffs[4]*h2**2*rho2 + coeffs[5]*h2**3*xr
    return W

def circular(N,dim = [512,512],center=[256,256]):
    x = np.linspace(1,dim[0],dim[0])
    y = x
    X = x-center[0]
    Y = y-center[1]
    P,Q = np.meshgrid(X/N,Y/N)
    out = P**2+Q**2
    out = out<=1
    return out.astype(float)
def circ(X):
    out = X<=1
    return out.astype(float)

def main():
    out = circular(4)
    print(out)

if __name__ == "__main__":
    main()