import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import probfit.pdf
import numpy as np

def cruijff(x, norm, mu, sigmaL, sigmaR, alphaL, alphaR):
    return norm * probfit.vector_apply(probfit.pdf.cruijff, x, mu, sigmaL, sigmaR, alphaL, alphaR)

with open("eles_Threshold0.pkl", 'rb') as f:
    t0 = pickle.load(f)['hist']

with open("eles_Threshold135.pkl", 'rb') as f:
    t135 = pickle.load(f)['hist']

with open("eles_Badae.pkl", 'rb') as f:
    ae = pickle.load(f)['hist']

etabins = ae.axes['eta'].edges

def fit_cruijff(h, label):
    #h = h[]
    values = h.values(flow=False);
    #print(values.dtype)
    centers = h.axes[0].centers
    errors = np.sqrt(h.values(flow=False)+1)

    #print(values)
    #print(centers)
    #print(errors)

    norm0 = np.max(values)
    mu0 = np.sum(centers*values)/np.sum(values)
    sigma0 = np.sum(centers*centers*values)/np.sum(values)
    p0 = [norm0, mu0, sigma0, sigma0, 1, 1]
    bounds = [(0, -np.inf, 0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)]
    popt, perr, infodict, mesg, ier = curve_fit(cruijff, centers, values, p0=p0, full_output=True, bounds=bounds, sigma=errors)
    print(popt)
    #print(perr)
    #print(infodict)
    #print(mesg)
    #print(ier)

    color = next(plt.gca()._get_lines.prop_cycler)['color']
    h.plot(label=label, color = color)
    plt.plot(centers, cruijff(centers, *popt), label=None, c = color)

    mu = popt[1]*100
    sigma = 50*(popt[2]+popt[3])/(1+popt[1])
    return mu, sigma


def do_etabin(i):
    start = 20
    end = 80
    plt.title("$%0.2f < |\eta| < %0.2f$"%(etabins[i], etabins[i+1]))
    pt0 = fit_cruijff(t0[{'eta':i, 'genPt':slice(5,25,sum)}].project('ptErr')[start:end], label='Threshold 0')
    pt135 = fit_cruijff(t135[{'eta':i, 'genPt':slice(5,25,sum)}].project('ptErr')[start:end], label='Threshold 1.35')
    pae = fit_cruijff(ae[{'eta':i, 'genPt':slice(5,25,sum)}].project('ptErr')[start:end], label='Rohans ancient AE')
    text = 'Threshold 0:\n\t$\mu=%0.2f\%%,\sigma/\mu=%0.2f\%%$'%(pt0[0], pt0[1])
    text += '\nThreshold 1.35:\n\t$\mu=%0.2f\%%,\sigma=%0.2f\%%$'%(pt135[0], pt135[1])
    text += '\nRohans ancient AE:\n\t$\mu=%0.2f\%%,\sigma=%0.2f\%%$'%(pae[0], pae[1])
    plt.text(0.05, 0.7, text, transform=plt.gca().transAxes)
    plt.legend(loc='upper right')
    plt.savefig("eta_%d.png"%i)
    #plt.show()
    plt.clf()


for i in range(len(etabins)-1):
    do_etabin(i)
