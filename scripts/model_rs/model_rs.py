import os
import numpy as np
import h5py
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from heteromodes.restingstate import ModelBOLD, evaluate_model
from heteromodes.params import gen_param_combs
from heteromodes.utils import load_parc

load_dotenv()
PROJ_DIR = os.getenv("PROJ_DIR")
SURF_LH = os.getenv("SURF_LH")
GLASSER360_LH = os.getenv("GLASSER360_LH")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Model resting-state fMRI BOLD data and evaluate against empirical data.")
    parser.add_argument("--method", type=str, help="Whether to generate valid parameter combinations or run the model.")
    parser.add_argument("--id", type=int, help="The id of the run for saving outputs.")
    parser.add_argument("--surf_lh", type=str, default=SURF_LH, help="The surface file for the left hemisphere. Defaults to fsLR 32k (S1200).")
    parser.add_argument("--surf_rh", type=str, default=None, help="The surface file for the right hemisphere. Defaults to None.")
    parser.add_argument("--parc_lh", type=str, default=GLASSER360_LH, help="The parcellation file for the left hemisphere. Defaults to `Glasser360`.")
    parser.add_argument("--hmap_label", type=str, default=None, help="The label of the heterogeneity map. Defaults to None (indicating homogeneity)")
    parser.add_argument("--scale_method", type=str, default="norm", help="The scaling method for the heterogeneity map. Defaults to `norm`.")
    parser.add_argument("--parc_name", type=str, default="Glasser360", help="The name of the parcellation. Defaults to `Glasser360`.")
    parser.add_argument("--alpha", type=float, default=1.0, help="The alpha parameter for scaling the heterogeneity map. Defaults to 1.0.")
    parser.add_argument("--beta", type=float, default=1.0, help="The beta parameter for scaling the heterogeneity map. Defaults to 1.0.")
    parser.add_argument("--r", type=float, default=28.9, help="The spatial length scale (mm). Defaults to 28.9.")
    parser.add_argument("--gamma", type=float, default=0.116, help="The dampening rate (ms). Defaults to 0.116.")
    parser.add_argument("--nruns", type=int, default=50, help="The number of runs to simulate. Defaults to 50.")
    parser.add_argument("--tstep", type=float, default=0.09*1e3, help="The time step for the wave model (ms). Defaults to 0.09 ms.")
    parser.add_argument("--n_modes", type=int, default=500, help="The number of modes to calculate. Defaults to 500.")
    parser.add_argument("--aniso_method", type=str, default="aniso", help="The method to calculate the modes. Defaults to `aniso`.")
    parser.add_argument("--n_subjs", type=int, default=384, help="The number of subjects in the empirical data. Defaults to 384.")
    parser.add_argument("--slurm_id", type=int, default=None, help="The slurm id for the job. Defaults to None.")
    parser.add_argument("--save_all", action='store_true', help="Whether to save all outputs. Defaults to False.")
    args = parser.parse_args()

    # Set output folder and description
    out_dir = f'{PROJ_DIR}/results/model_rs/{args.hmap_label}/id-{int(args.id)}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if args.method == "gen_params":
        # Generate valid parameter combinations
        parc = load_parc(args.parc_lh)
        medmask = np.where(parc != 0, True, False)
        valid_combs = gen_param_combs(hmap_label=args.hmap_label, scale_method=args.scale_method,
                                      medmask=medmask) 
        # Save parameter combinations as a csv file
        valid_combs.to_csv(Path(out_dir, "csParamCombs.csv"), index=None, float_format='%.4f')   

    elif args.method == "run_model":
        out_file = f"slurmID-{args.slurm_id}_output.npz"

        ##### MODEL RESTING STATE BOLD ACTIVITY #####
        # parc_file = f"{PROJ_DIR}/data/parcellations/fsLR_32k_{args.parc_name}-lh.txt"
        t1 = time.time()
        model_rs = ModelBOLD(surf_file=args.surf_lh, parc_file=args.parc_lh, hmap_file=args.hmap_label,
                                    alpha=args.alpha, beta=args.beta, r=args.r, gamma=args.gamma,
                                    scale_method=args.scale_method)
        model_rs.calc_modes(args.n_modes, method=args.aniso_method)
        t2 = time.time()
        print(f"Time to initialise model and calculate modes: {(t2 - t1)/60:.3g} min")
        
        bold_model = []
        run_time = []
        for run in range(args.nruns):
            t1 = time.time()

            # Load external input
            ext_input = np.load(f"{PROJ_DIR}/data/resting_state/extInput_den-32k_randseed-{run}.npy")
            bold_model.append(model_rs.run_rest(sim_seed=run, tstep=args.tstep))

            t2 = time.time()
            run_time.append(t2 - t1)
            
        print(f"Average time per run: {np.mean(run_time)/60:.3g} min")
        print(f"Total time: {np.sum(run_time)/60:.3g} min")
        bold_model = np.dstack(bold_model)

        ##### EVALUATE RESTING STATE BOLD ACTIVITY #####
        t1 = time.time()
        bold_emp = h5py.File(f"{PROJ_DIR}/data/empirical/HCP_unrelated-445_rfMRI_hemi-L_nsubj-{args.n_subjs}_parc-glasser360_BOLD.hdf5", 'r')['bold']
        
        # TODO: load fcd_emp and fc_emp_avg instead of bold_emp

        # Evaluate simulated FC using empirical FC as ground truth
        results = evaluate_model(bold_emp, bold_model, TR=0.72)   # TR in seconds
        t2 = time.time()
        print(f"Evaluation time: {(t2 - t1)/60:.3g} min")

        # Print results
        print(f"Edge-level FC: {np.mean(results[0]):.3g}")
        print(f"Node-level FC: {np.mean(results[1]):.3g}")
        print(f"FCD: {np.mean(results[2]):.3g}")

        # Save evaluation metrics (edge_fc, node_fc, fcd_ks, fcd_emp, fcd_model, fc_emp_avg, fc_model)
        if args.save_all:
            print("Saving all results...")
            out_file = out_file.replace('saveMetrics', 'saveAll')
            np.savez(f'{out_dir}/{out_file}', 
                    edge_fc=results[0], node_fc=results[1], fcd=results[2], fcd_emp=results[3], 
                    fcd_model=results[4], fc_emp_avg=results[5], fc_model=results[6],
                    bold_model=bold_model, bold_emp=bold_emp)
        else:
            print("Saving evaluation metrics only...")
            np.savez(f'{out_dir}/{out_file}', edge_fc=results[0], node_fc=results[1], 
                    fcd=results[2])
    else:
        print("Invalid method. Exiting...")
