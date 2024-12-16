import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import probfit.pdf
import pandas as pd
import numpy as np
import os
import seaborn as sns
import shutil

def cruijff(x, norm, mu, sigmaL, sigmaR, alphaL, alphaR):
    return norm * probfit.vector_apply(probfit.pdf.cruijff, x, mu, sigmaL, sigmaR, alphaL, alphaR)

# Files is the list of files you want to run, and the names is how you will name them

files = ["eles_FloatingpointoptimizedCAEv1DummyHistomaxDummynTuple.pkl","eles_FloatingpointoptimizedCAEv2DummyHistomaxDummynTuple.pkl","eles_FloatingpointThreshold0DummyHistomaxDummynTuple.pkl","eles_FloatingpointThreshold135DummyHistomaxDummynTuple.pkl"]#,

names = ["Optimized_v1","Optimized_v2","Threshold0",
        "Threshold135"]#,

loaded_files = {}
for f in files:
    with open(f, 'rb') as cur_f:
        loaded_files[f] = pickle.load(cur_f)

t0 = loaded_files['eles_FloatingpointThreshold0DummyHistomaxDummynTuple.pkl']
etabins = t0['hist'].axes['eta'].edges
ptbins = t0['hist'].axes['genPt'].edges

import hist
def create_histogram(data):
    H = hist.Hist(
        hist.axis.Regular(80, -1.5, 1.5, name='ptErr', label='(reco pt - Th0 pt)/Th0 pt'),
        hist.axis.Regular(50, 0, 100, name='genPt', label='gen pt'),
        hist.axis.Regular(5, 1.4, 3.0, name='eta', label='gen eta'),
    )
    H.fill(
        ptErr=data['ptErr'],
        genPt=data['genPt'],
        eta=np.abs(data['eta']),
    )
    return H

def plot_hist(h, label, index = None):
    values = h.values(flow=False);
    centers = h.axes[0].centers
    errors = np.sqrt(h.values(flow=False)+1)
    
    if index is not None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color = color_cycle[index]
    else:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
    h.plot(label=label, color = color, alpha = 0.5, density = True)

    
def fit_cruijff(h, label):
    #h = h[]
    
    values = h.values(flow=False);
    centers = h.axes[0].centers
    errors = np.sqrt(h.values(flow=False)+1)


    norm0 = np.max(values)
    mu0 = np.sum(centers*values)/np.sum(values)
    sigma0 = np.sum(centers*centers*values)/np.sum(values)
    p0 = [norm0, 0, 0.1, 0.1,0.1,0.1]
    bounds = [(0, -np.inf, 0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)]
    popt, perr, infodict, mesg, ier = curve_fit(cruijff, centers, values, p0=p0, full_output=True, bounds=bounds, sigma=errors,absolute_sigma = True)

    color = next(plt.gca()._get_lines.prop_cycler)['color']
    h.plot(label=label, color = color, alpha = 0.5)
    filename = f"{label}_data.txt"
    mu = popt[1]*100
    sigma = 50*(popt[2]+popt[3])/(1+popt[1])

    with open(filename, 'a') as file:
        file.write(f"{mu}, {sigma}\n")
    return mu, sigma

def convert_to_numpy(data):
    return {key: np.array(value) for key, value in data.items()}

def do_etabin(i,pt_min, pt_max,opath):
    
    start = 20
    end = 180
    plt.title("$%0.2f < |\eta| < %0.2f$"%(etabins[i], etabins[i+1]))
    plt.xlim(-2,2)
    filename = f"eta_ranges.txt"
    with open(filename, 'a') as file:
        file.write(f"{etabins[i]}, {etabins[i+1]}\n")
    for j,f in enumerate(files):
        plot_hist(loaded_files[f]['hist'][{'eta':i, 'genPt':slice(pt_min, pt_max,sum)}].project('ptErr')[start:end], label=names[j])


    plt.legend(loc='upper right')
    plt.savefig(f"{opath}/eta_%d.png"%i)
    #plt.show()
    plt.clf()
    
    
    
    
    # Function to extract histogram data
    from scipy.stats import pearsonr
    def calculate_correlations(reference_hist, histograms, eta_index, genPt, start, end):
        correlations = []
        optimized_parameters = {}  # Dictionary to store optimized parameters
        def match_data(base_data, compare_data):
            filtered_ids = {}
            for idx, event_id in enumerate(base_data['event_id']):
                # Check if eta and genPt values fall within the specified ranges
                eta = base_data['eta'][idx]
                gen_pt = base_data['genPt'][idx]

                # Assume you have etabins[i] and etabins[i+1] as the eta range
                # and genPt[0] and genPt[1] as the genPt range
                eta_range_min = etabins[eta_index]
                eta_range_max = etabins[eta_index+1]
                gen_pt_min = genPt[0]
                gen_pt_max = genPt[1]

                if eta_range_min <= eta <= eta_range_max and gen_pt_min <= gen_pt <= gen_pt_max:
                    # Add to filtered_ids if the conditions are met
                    filtered_ids[event_id] = idx


            data = {'base': [], 'compare': [],'error': []}
            for i, event_id in enumerate(compare_data['event_id']):
                if event_id in filtered_ids:
                    base_idx = filtered_ids[event_id]
                    data['base'].append(base_data['ptErr'][base_idx]*base_data['genPt'][base_idx]+base_data['genPt'][base_idx])
                    data['compare'].append(compare_data['ptErr'][i]*base_data['genPt'][base_idx]+base_data['genPt'][base_idx])
                    data['error'].append(compare_data['ptErr'][i]*base_data['genPt'][base_idx]-base_data['ptErr'][base_idx]*base_data['genPt'][base_idx])
            return data
        def optimize_transformation(x, y):
            """
            Find A, B, and C such that correlation between x and A*y^2 + B*y + C is maximized.
            """
            def correlation_loss(params):
                A, B, C = params
                y_transformed = A * y**2 + B * y + C
                correlation, _ = pearsonr(x, y_transformed)
                return -correlation  # Negative because we want to maximize

            # Initial guess for A, B, and C
            initial_guess = [1.0, 1.0, 0.0]

            # Perform optimization
            result = minimize(correlation_loss, initial_guess, method='BFGS')
            return result.x

        for i, data in enumerate(histograms):
            d = match_data(reference_hist, data)
            d = pd.DataFrame(d)
            d['abs_error'] = d['error'].abs()

            # Find the 10th and 90th percentiles of the absolute error
            lower_percentile = d['abs_error'].quantile(0.10)
            upper_percentile = d['abs_error'].quantile(0.90)
            filtered_data = d[(d['abs_error'] >= lower_percentile) & (d['abs_error'] <= upper_percentile)]
            filtered_data = filtered_data.drop(columns=['abs_error'])

            base_data = filtered_data['base']
            compare_data = filtered_data['compare']

            # Find the optimal A, B, and C
            A_opt, B_opt, C_opt = optimize_transformation(base_data, compare_data)
            transformed_compare_data = A_opt * compare_data**2 + B_opt * compare_data + C_opt

            # Save the optimized parameters in the dictionary
            optimized_parameters[(names[i], eta_index)] = {'A': A_opt, 'B': B_opt, 'C': C_opt}

            # Calculate correlations
            original_correlation, _ = pearsonr(base_data, compare_data)
            transformed_correlation, _ = pearsonr(base_data, transformed_compare_data)
            correlations.append(original_correlation)

            # Create the subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot original data
            ax1.scatter(base_data, compare_data)
            ax1.set_xlabel('Th0 pT')
            ax1.set_ylabel(f'{names[i]} pT')
            ax1.set_title('Original Data Scatter Plot')
            # Fit line for original data
            m, c = np.polyfit(base_data, compare_data, 1)
            ax1.plot(base_data, m * base_data + c, 'r--', label=f'Fit line: y={m:.2f}x+{c:.2f}')
            ax1.text(0.05, 0.95, f'Corr: {original_correlation:.2f}', transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax1.legend()

            # Plot transformed data
            ax2.scatter(base_data, transformed_compare_data)
            ax2.set_xlabel('Th0 pT')
            ax2.set_ylabel(f'{names[i]} pT (Transformed)')
            ax2.set_title('Transformed Data Scatter Plot')
            # Fit line for transformed data
            poly_coeffs = np.polyfit(base_data, transformed_compare_data, 1)
            m, c = poly_coeffs
            ax2.plot(base_data, m * base_data + c, 'r--', label=f'Fit line: y={m:.2f}x+{c:.2f}')
            ax2.text(0.05, 0.95, f'Corr: {transformed_correlation:.2f}', transform=ax2.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax2.legend()

            # Save the combined plot
            plt.tight_layout()
            plt.savefig(f"{opath}/{names[i]}_{eta_index}_scatter_corr_combined.png")
            plt.close()

        return correlations,optimized_parameters
    
    histograms = [loaded_files[f]['data'] for f in files]

    eta_value = i  # Replace with your eta value
    genPt_range = (pt_min, pt_max)
    start_index = start
    end_index = end

    # Calculate correlations
    correlations,optimized_parameters = calculate_correlations(t0['data'], histograms, eta_value, genPt_range, start_index, end_index)

    
    # Print the correlations
    labels = names
    
    
    start = 20
    end = 180
    plt.title("$%0.2f < |\eta| < %0.2f$"%(etabins[i], etabins[i+1]))
    filename = f"eta_ranges.txt"
    with open(filename, 'a') as file:
        file.write(f"{etabins[i]}, {etabins[i+1]}\n")
    # First, calculate the result for pt0
    
    # Load data and convert to numpy arrays
    data_dict = {f:convert_to_numpy(loaded_files[f]['data']) for f in files}
    pt0_data = data_dict['eles_FloatingpointThreshold0DummyHistomaxDummynTuple.pkl']
    # Create a dictionary to map event_id to indices in pt0
    pt0_event_id_to_index = {event_id: idx for idx, event_id in enumerate(pt0_data['event_id'])}

    # Function to calculate the difference array by matching event_id
    def compute_relative_performance(base_data, compare_data):
        diff = {'ptErr': [], 'genPt': [], 'eta': []}
        for i, event_id in enumerate(compare_data['event_id']):
            if event_id in pt0_event_id_to_index:
                base_idx = pt0_event_id_to_index[event_id]
                diff['ptErr'].append(compare_data['ptErr'][i] - base_data['ptErr'][base_idx])
                diff['genPt'].append(compare_data['genPt'][i])
                diff['eta'].append(compare_data['eta'][i])
        return {key: np.array(value) for key, value in diff.items()}

    # Calculate the relative performance
    relative_performance = {f: compute_relative_performance(pt0_data, data_dict[f]) for f in files }
    relative_hist = {f: create_histogram(relative_performance[f]) for f in files }


    # Now, calculate and plot the differences
    arr = [plot_hist(relative_hist[f][{'eta':i, 'genPt':slice(pt_min, pt_max,sum)}].project('ptErr')[start:end], label=names[j], index = j) for j,f in enumerate(files) if f != 'eles_FloatingpointThreshold0DummyHistomaxDummynTuple.pkl']


    for j in range(len(names)):
        d = {}
        d['name'] = names[j]
        d['pt_min'] = pt_min
        d['pt_max'] = pt_max
        d['eta_min'] = etabins[i]
        d['eta_max'] = etabins[i+1]
        d['corr'] = correlations[j]
        row.append(d)
    
    
    plt.legend(loc='upper right')
    plt.xlim(-1,1)
    plt.savefig(f"{opath}/eta_%d_delta_th0.png"%i)
    plt.clf()
    

def do_alleta(pt_min, pt_max,opath):
    
    
    start = 20
    end = 180
    plt.title("All $|\eta|$")
    
    for j,f in enumerate(files):
        plot_hist(loaded_files[f]['hist'][{'genPt':slice(pt_min, pt_max,sum)}].project('ptErr')[start:end], label=names[j])


    plt.legend(loc='upper right')
    plt.savefig(f"{opath}/all_eta_.png")
    #plt.show()
    plt.clf()
    

pt_ranges = [[5,25],[25,50],[0,100]]
shutil.rmtree('pt_eta_plots')

row = []
os.mkdir('pt_eta_plots')
for pt_min, pt_max in pt_ranges:
    opath = f'pt_eta_plots/{pt_min}_{pt_max}'
    os.mkdir(opath)
    do_alleta(pt_min, pt_max,opath)
    for i in range(len(etabins)-1):
        do_etabin(i,pt_min, pt_max,opath)
        
df = pd.DataFrame.from_dict(row)
df.to_csv('pt_eta_plots/out.csv', index=False)  


# Load the data
data = df

# Unique models and pt bins
models = data['name'].unique()
pt_bins = data[['pt_min', 'pt_max']].drop_duplicates().values

# Plotting
fig, axs = plt.subplots(len(pt_bins), 1, figsize=(15, 5 * len(pt_bins)))
for i, (pt_min, pt_max) in enumerate(pt_bins):
    for model in models:
        # Filter data for each model and pt bin
        subset = data[(data['name'] == model) & (data['pt_min'] == pt_min) & (data['pt_max'] == pt_max)]

        # Scatter plot for mean
        axs[i].plot(subset['eta_min'], subset['corr'], label=model)
        axs[i].set_title(f'% Th0 Corr for pT {2*pt_min}-{2*pt_max}')
        axs[i].set_xlabel('Eta Min')
        axs[i].set_ylabel('Th0 Corr')
        axs[i].grid(True)
        axs[i].legend()

plt.tight_layout()
plt.savefig('pt_eta_plots/corr_trend_lines')

plt.close()
