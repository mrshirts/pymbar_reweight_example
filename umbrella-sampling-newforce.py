"""
Example illustrating the application of MBAR to compute a 1D free energy profile from an umbrella sampling simulation.

The data represents an umbrella sampling simulation for the chi torsion of
a valine sidechain in lysozyme L99A with benzene bound in the cavity.

Reference:

    D. L. Mobley, A. P. Graves, J. D. Chodera, A. C. McReynolds, B. K. Shoichet and K. A. Dill,
    "Predicting absolute ligand binding free energies to a simple model site,"
    Journal of Molecular Biology 371(4):1118-1134 (2007).
    http://dx.doi.org/10.1016/j.jmb.2007.06.002


Modified to add external forces and remove temperature dependence

"""

import numpy as np
import matplotlib.pyplot as plt

import pymbar  # multistate Bennett acceptance ratio
from pymbar import timeseries  # timeseries analysis

# Constants.
kB = 1.381e-23 * 6.022e23 / 1000.0  # Boltzmann constant in kJ/mol/K

temperature = 300  # assume a single temperature

# Parameters
K = 26  # number of umbrellas
N_max = 501  # maximum number of snapshots/simulation
beta = 1.0 / (kB * temperature)  # inverse temperature of simulations (in 1/(kJ/mol))
chi_min = -180.0  # min for free energy profile
chi_max = +180.0  # max for free energy profile
nbins = 30  # number of bins for 1D free energy profile
nplot = 1200 # number of plot points
xplot = np.linspace(chi_min, chi_max, nplot)  # which points we will plot

# Allocate storage for simulation data

# N_k[k] is the number of snapshots from umbrella simulation k
N_k = np.zeros([K], dtype=int)
# K_k[k] is the spring constant (in kJ/mol/deg**2) for umbrella simulation k
K_k = np.zeros([K])
# chi0_k[k] is the spring center location (in deg) for umbrella simulation k
chi0_k = np.zeros([K])
# chi_kn[k,n] is the torsion angle (in deg) for snapshot n from umbrella simulation k
chi_kn = np.zeros([K, N_max])
# u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
u_kn = np.zeros([K, N_max])
g_k = np.zeros([K])

# Read in umbrella spring constants and centers.
with open("data/centers.dat") as infile:
    lines = infile.readlines()

for k in range(K):
    # Parse line k.
    line = lines[k]
    tokens = line.split()
    chi0_k[k] = float(tokens[0])  # first entry is the spring center location (in deg)
    # second entry is the spring constant (read in kJ/mol/rad**2, converted to kJ/mol/deg**2)
    K_k[k] = float(tokens[1]) * (np.pi / 180) ** 2

# Read the simulation data
for k in range(K):
    # Read torsion angle data.
    filename = f"data/prod{k:d}_dihed.xvg"
    print(f"Reading {filename}...")
    n = 0
    with open(filename, "r") as infile:
        for line in infile:
            if line[0] != "#" and line[0] != "@":
                tokens = line.split()
                chi = float(tokens[1])  # torsion angle
                # wrap chi_kn to be within [-180,+180)
                while chi < -180.0:
                    chi += 360.0
                while chi >= +180.0:
                    chi -= 360.0
                chi_kn[k, n] = chi
                n += 1
    N_k[k] = n

    # Compute correlation times for potential energy and chi timeseries.

    chi_radians = chi_kn[k, 0 : N_k[k]] / (180.0 / np.pi)
    g_cos = timeseries.statistical_inefficiency(np.cos(chi_radians))
    g_sin = timeseries.statistical_inefficiency(np.sin(chi_radians))
    print(f"g_cos = {g_cos:.1f} | g_sin = {g_sin:.1f}")
    g_k[k] = max(g_cos, g_sin)
    print(f"Correlation time for set {k:5d} is {g_k[k]:10.3f}")
    indices = timeseries.subsample_correlated_data(chi_radians, g=g_k[k])
    # Subsample data.
    N_k[k] = len(indices)
    chi_kn[k, 0 : N_k[k]] = chi_kn[k, indices]

# compute bin centers
bin_center_i = np.zeros([nbins])
bin_edges = np.linspace(chi_min, chi_max, nbins + 1)
for i in range(nbins):
    bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

N = np.sum(N_k)
chi_n = pymbar.utils.kn_to_n(chi_kn, N_k=N_k)

# Evaluate reduced energies in all umbrellas
print("Evaluating reduced potential energies...")
# reduced potential energies

u_kn = np.zeros([K,N])
ntot = 0
for k in range(K):
    for n in range(N_k[k]):
        # Compute minimum-image torsion deviation from umbrella center l
        dchi = chi_kn[k, n] - chi0_k
        for l in range(K):
            if abs(dchi[l]) > 180.0:
                dchi[l] = 360.0 - abs(dchi[l])

        # Compute energy of snapshot n from simulation k in umbrella potential l
        u_kn[:, ntot] = beta * (K_k / 2.0) * dchi**2
        ntot +=1

# compute unweighted samples
u_n = np.zeros([N])

# compute new weights with different torsional forces
unew_n = np.zeros([N])
ntot = 0
period = 2
kf = 10

for k in range(K):
    for n in range(N_k[k]):
        unew_n[ntot] = beta*kf*np.cos(period*chi_kn[k, n]/(180.0/np.pi))  # we don't subtract off anything, just add a new potential
        ntot +=1

# now calculate the number of effective samples with the reweighting.  Easiest way is to do a new full solve for MBAR.

mbar = pymbar.MBAR(np.vstack([u_kn,u_n]), np.array(list(N_k) + [0]))
samp_orig = mbar.compute_effective_sample_number()   
print(f"effective sample number in unbiased state: {samp_orig[-1]:10.4f}")   # just print the last state
mbar = pymbar.MBAR(np.vstack([u_kn,unew_n]), np.array(list(N_k) + [0])) 
samp_new = mbar.compute_effective_sample_number()   
print(f"effective sample number in biased state: {samp_new[-1]:10.4f}")      # just print the last state


# initialize free energy profile with the data collected
fes = pymbar.FES(u_kn, N_k, verbose=True)
# set the number of bootstraps
n_bootstraps = 20

def generate_and_plot_stuff(u_n, suffix=""):

    '''
    u_n - array, length N_k - change in energy from the original simulation to estimate the PMF from
    inputs = strin - gsuffix for plots
    '''

    plt.clf()
    # Compute free energy profile in unbiased potential (in units of kT) in a histogram
    histogram_parameters = {}
    histogram_parameters["bin_edges"] = bin_edges

    fes.generate_fes(u_n, chi_n, fes_type="histogram", histogram_parameters=histogram_parameters, n_bootstraps = n_bootstraps)

    results = fes.get_fes(bin_center_i, reference_point="from-lowest", uncertainty_method="analytical")
    center_f_i = results["f_i"]
    center_df_i = results["df_i"]

    # Write out free energy profile
    print("free energy profile (in units of kT), from histogramming")
    print(f"{'bin':>8s} {'f':>8s} {'df':>8s}")
    for i in range(nbins):
        print(f"{bin_center_i[i]:8.1f} {center_f_i[i]:8.3f} {center_df_i[i]:8.3f}")


    # get the result it over more points    
    results = fes.get_fes(xplot, reference_point="from-lowest", uncertainty_method="analytical")
    
    #plot free energy profile
    perbin = nplot // nbins
    # get the errors in the rigtt place                  
    indices = np.arange(0, nplot, perbin) + int(perbin // 2)
    plt.errorbar(
        xplot[indices],
        results['f_i'][indices],
        yerr=results['df_i'][indices],
        fmt="none",
        ecolor="r",
        elinewidth=1.0,
        capsize=3,
        label = 'Analytic Error'
    )

    plt.plot(xplot,results['f_i'],'k')

    ### mow calculate the bootstrap errors
    results = fes.get_fes(xplot, reference_point="from-lowest", uncertainty_method="bootstrap")

    #plot free energy profile
    perbin = nplot // nbins
    # get the errors in the rigtt place                                                                                            
    indices = np.arange(0, nplot, perbin) + int(perbin // 2)
    plt.errorbar(
        xplot[indices],
        results['f_i'][indices],
        yerr=results['df_i'][indices],
        fmt="none",
        ecolor="b",
        elinewidth=1.0,
        capsize=3,
        label = 'Bootstrap error'
    )
    plt.plot(xplot,results['f_i'],'k')

    plt.title("Histogram FES (" + suffix + ")" )
    plt.legend()
    plt.savefig("histogram_fes" + "_" + suffix + ".pdf")
    
    # NOW do the same with KDE:
    plt.clf()
    kde_parameters = {}
    # set the band width
    kde_parameters["bandwidth"] = 0.25 * ((chi_max - chi_min) / nbins)

    # we only have bootstrap uncertainties for KDE
    fes.generate_fes(u_n, chi_n, fes_type="kde", kde_parameters=kde_parameters, n_bootstraps = n_bootstraps)
    results = fes.get_fes(xplot, reference_point="from-lowest", uncertainty_method='bootstrap')

    plt.errorbar(
        xplot[indices],
        results['f_i'][indices],
        yerr=results['df_i'][indices],
        fmt="none",    
        elinewidth=0.8,
        capsize=3,
        label = "Bootstrap error"
    )

    plt.plot(xplot,results['f_i'],'k')
    plt.legend()
    plt.title("Kernel density FES (" + suffix + ")")
    plt.savefig('kde_bootstrap' + '_' + suffix + '.pdf')


# now actually do the calculation.  Calculate the energy without biases    
u_n = np.zeros(N)
# Now add a new force
generate_and_plot_stuff(u_n,"original")

# figure out the new weights.  Need to call mbar

# then find the weights of these unsampled states.
# Now add a new force, and calculate the PMF

generate_and_plot_stuff(unew_n,"reweighted")


              
