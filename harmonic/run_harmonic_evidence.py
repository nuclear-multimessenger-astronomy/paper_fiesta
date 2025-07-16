"""
Run harmonic to compute the evidences

In case a multimodal base is needed, check out this branch: https://github.com/astro-informatics/harmonic/tree/multimodal_base
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import time
import corner
import argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import harmonic as hm

params = {"axes.grid": False,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        # color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "black",
                        save=False)

parser = argparse.ArgumentParser(description="Run harmonic with a multimodal base on some of the events")
# Choosing the event and run params
parser.add_argument("--outdir",
                    type=str,
                    help="Path to save plots and results",
)
# General training params
parser.add_argument("--N-samples-train",
                    type=int,
                    default=40_000,
                    help="Target number of samples, for downsampling for training",
)
parser.add_argument("--permute-samples",
                    type=bool,
                    default=True,
                    help="Whether to permute the samples or not",
)
parser.add_argument("--training-proportion",
                    type=float,
                    default=0.50,
                    help="Fraction of samples used for training the NF model vs testing")
parser.add_argument("--temperature",
                    type=float,
                    default=0.4,
                    help="Temperature scaling for the likelihood")
parser.add_argument("--nchains",
                    type=int,
                    default=20,
                    help="Number of chains for sampling")
# Spline parameters
parser.add_argument("--n-layers",
                    type=int,
                    default=2,
                    help="Number of spline layers")
parser.add_argument("--n-bins",
                    type=int,
                    default=64,
                    help="Number of bins for splines")
parser.add_argument("--hidden-size",
                    type=int,
                    nargs='+',
                    default=[128, 128],
                    help="Hidden layer sizes for the spline network")
parser.add_argument("--spline-range",
                    type=float,
                    nargs=2,
                    default=[-10.0, 10.0],
                    help="Range for the spline transformation")
parser.add_argument("--load-model",
                    type=bool,
                    default=False,
                    help="Whether to load a pretrained mode",
)
# Optimizer parameters
parser.add_argument("--learning-rate",
                    type=float,
                    default=0.001,
                    help="Learning rate for the optimizer")
parser.add_argument("--momentum",
                    type=float,
                    default=0.9,
                    help="Momentum for the optimizer")
parser.add_argument("--standardize",
                    type=bool,
                    default=True,
                    help="Whether to standardize the inputs")
parser.add_argument("--seed",
                    type=int,
                    default=10,
                    help="Random seed")
parser.add_argument("--epochs-num",
                    type=int,
                    default=60,
                    help="Number of epochs to train")
# Multimodal base
parser.add_argument("--multimodal-base",
                    type=bool,
                    default=False,
                    help="Whether to use a multimodal base distribution")
parser.add_argument("--nmodes",
                    type=int,
                    default=2,
                    help="Number of modes in the multimodal base")
parser.add_argument("--spacing",
                    type=float,
                    default=3.0,
                    help="Spacing between modes in the multimodal base")
# Some plotting stuff
parser.add_argument("--plot-flow",
                    type=bool,
                    default=True,
                    help="Whether to plot the flow or not",
)
parser.add_argument("--plot-samples",
                    type=bool,
                    default=False,
                    help="Whether to plot the samples or not",
)
parser.add_argument("--plot-marginals",
                    type=bool,
                    default=False,
                    help="Whether to plot the marginals or not",
)

def make_ladder_base_centers(ndim, nmodes, spacing):
    """
    Auxiliary function to create a ladder base for the spline model.
    """
    base_centers = []
    for i in range(nmodes):
        center = jnp.full(ndim, spacing * i)
        base_centers.append(center)
    return base_centers

def create_save_label(args, perm_seed=None):
    """Create a somewhat unique identifier for plots etc, mainly based on hyperparameters."""
    save_lab = (
        "harmonic_"
        + str(args.n_layers)
        + "l_"
        + str(args.n_bins)
        + "b_"
        + str(args.epochs_num)
        + "e_"
        + str(args.seed)
        + "s_"
        + str(int(args.training_proportion * 100))
        + "perc"
    )
    
    if args.multimodal_base:
        save_lab += f"_{args.nmodes}mm_{args.spacing}sp"

    if args.permute_samples:
        if perm_seed is None:
            raise ValueError("perm_seed must be provided when permute_samples is True")
        save_lab += f"_{perm_seed}perm"

    return save_lab

def main():
    args = parser.parse_args()
    
    # Check if the outdir exists, if not create it:
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created output directory: {args.outdir}")
    
    # Check if the file exists otherwise I messed up:
    path = os.path.join(args.outdir, "results_production.npz")
    if not os.path.exists(path):
        print(f"We did not find the file {path}, please run the production script first.")
    
    # Load the log_prob    
    data = dict(np.load(path))
    log_prob = data["log_prob"].flatten()
    chains = data["chains"]
    nb_chains, nb_steps, ndim = chains.shape
    chains = np.array(chains).reshape(nb_chains * nb_steps, ndim).T
    print(f"Loading from file = {path} was successful")
    
    ### TEST:
    print(f"Shape of log_prob: {np.shape(log_prob)}")
    print(f"Shape of chains: {np.shape(chains)}")
    
    # Downsample the chains and the log_prob:
    nb_samples = chains.shape[1]
    
    print(f"Number of samples in the chains: {nb_samples}")
    downsample_factor = nb_samples // args.N_samples_train
    chains = chains[:, ::downsample_factor]
    log_prob = log_prob[::downsample_factor]
    ndim = chains.shape[0]
    print(f"Downsampled to {chains.shape[1]} samples, ndim = {ndim}")
    
    if args.permute_samples:
        perm_seed = 2
        print("Permuting chains...")
        np.random.seed(perm_seed)
        permutation = np.random.permutation(chains.shape[1])
        print(chains[:, permutation[0]], log_prob[permutation[0]])
        chains = chains[:, permutation]
        log_prob = log_prob[permutation]

    print("Mean is ", np.mean(chains[0, :]))

    print("Plotting posterior samples...")
    samples = np.array(chains).T
    if args.plot_samples:
        hm.utils.plot_getdist(samples)
        save_name = os.path.join(args.outdir, "harmonic_posterior_samples.pdf")
        print(f"Saving posterior samples plot to {save_name}")
        plt.savefig(save_name, bbox_inches="tight")

    base_centers = make_ladder_base_centers(ndim, args.nmodes, args.spacing)
    save_lab = create_save_label(args, perm_seed=perm_seed)
    print(f"The label for the model is {save_lab}")

    model = hm.model.RQSplineModel(
        ndim,
        n_layers=args.n_layers,
        n_bins=args.n_bins,
        hidden_size=args.hidden_size,
        spline_range=args.spline_range,
        standardize=args.standardize,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        temperature=args.temperature,
        multimodal_base=args.multimodal_base,
        base_centers=base_centers,
    )

    print("Configure chains for cross validation stage...")
    chains = hm.Chains(ndim)
    chains.add_chain(samples, log_prob)
    chains.split_into_blocks(args.nchains)
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=args.training_proportion
    )

    # Fit/load model
    model_name_str = os.path.join(args.outdir, f"model_{save_lab}")
    print(f"The model will be saved/loaded to/from {model_name_str}")
    if args.load_model:
        # TODO: test this/delete this later on?
        print("Loading in model...")
        model = hm.model.RQSplineModel.deserialize(model_name_str)
    else:
        clock = time.process_time()
        print(f"Fitting model for {args.epochs_num} epochs...")
        model.fit(
            chains_train.samples,
            epochs=args.epochs_num,
            verbose=True,
            key=jax.random.PRNGKey(args.seed),
        )
        model.serialize(model_name_str)
        clock = time.process_time() - clock
        print("Execution time = {}s".format(clock))

    # =======================================================================
    # Plot flow
    # =======================================================================

    if args.plot_flow:
        # Sample from the flow model
        num_samp = 10_000
        samps_compressed = np.array(model.sample(num_samp))
        
        print("Plotting...")
        
        # First plot the original chains
        default_corner_kwargs["color"] = "blue"
        hist_kwargs = {"color": "blue", "density": True}
        default_corner_kwargs["hist_kwargs"] = hist_kwargs
        fig = corner.corner(chains_infer.samples, **default_corner_kwargs)
        
        # Second plot the NF samples
        default_corner_kwargs["color"] = "red"
        hist_kwargs = {"color": "red", "density": True}
        default_corner_kwargs["hist_kwargs"] = hist_kwargs
        
        corner.corner(samps_compressed, fig=fig, **default_corner_kwargs)
        save_name = os.path.join(args.outdir, f"corner_{save_lab}.pdf")
        print(f"Saving cornerplot of the PDFs to {save_name}")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()

    print("Done plotting!")

    clock = time.process_time()
    print("Compute evidence...")
    print("Using model ", save_lab)
    """
    Instantiates the evidence class with a given model. Adds some chains and 
    computes the log-space evidence (marginal likelihood).
    """
    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    ev.check_basic_diagnostic()
    # ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
    err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()
    clock = time.process_time() - clock

    # ===========================================================================
    # Display evidence results
    # ===========================================================================
    print("The inverse evidence in log space is:")
    print(f"ln_inv_evidence = {ev.ln_evidence_inv} +/- {err_ln_inv_evidence}")
    print(f"ln evidence = {-ev.ln_evidence_inv} +/- {-err_ln_inv_evidence[1], -err_ln_inv_evidence[0]}")
    print(f"kurtosis = {ev.kurtosis}. Aim for ~3.")
    #print("ln inverse evidence per chain ", ev.ln_evidence_inv_per_chain)
    #print(
    #    "Average ln inverse evidence per chain ",
    #    np.mean(ev.ln_evidence_inv_per_chain),
    #)
    # print(
    #     "lnargmax",
    #     ev.lnargmax,
    #     "lnargmin",
    #     ev.lnargmin,
    #     "lnprobmax",
    #     ev.lnprobmax,
    #     "lnprobmin",
    #     ev.lnprobmin,
    #     "lnpredictmax",
    #     ev.lnpredictmax,
    #     "lnpredictmin",
    #     ev.lnpredictmin,
    # )
    check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
    print(f"Check = {check} Aim for sqrt( 2/(n_eff-1) ) = {np.sqrt(2.0 / (ev.n_eff - 1))}")
    print(f"sqrt(evidence_inv_var_var) / evidence_inv_var = {check}")
    
    out_file = os.path.join(args.outdir, f"evidence_results.npz")
    np.savez(out_file, 
             ln_evidence = -ev.ln_evidence_inv,
             ln_evidence_plus = -err_ln_inv_evidence[1],
             ln_evidence_minus = -err_ln_inv_evidence[0],
             kurtosis=ev.kurtosis,
             check=check)

if __name__ == "__main__":
    main()