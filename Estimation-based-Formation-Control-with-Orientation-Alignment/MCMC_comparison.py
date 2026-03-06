from simulation import MASsimulation
import numpy as np
from utils import *
import concurrent.futures
import time
import traceback

def run_simulation(args):
    try:
        sim = MASsimulation(args)
        return sim.eval.get_results()
    except Exception:
        traceback.print_exc()
        return None

def gen_directed_ring_adj(N, k_neighbors):
    adj_mtx = np.zeros((N, N), dtype=int)
    for i in range(N):
        adj_mtx[i, i] = 1
        adj_mtx[i, (i - 1) % N] = 1
        for j in range(1, k_neighbors): 
            adj_mtx[i, (i + j) % N] = 1
    return adj_mtx

if __name__ == "__main__":
    num_runs_per_scenario = 20
    max_concurrent_processes = 2
    
    w_std_val = 0.03
    w_std_orientation_val = 0.03
    v_std_val = 0.3
    v_std_orientation_val = 0.4
    N_agent = 10 
    connectivity_values = list(range(2, 10)) 

    control_types = [
        CtrlTypes.CtrlEstFeedback,
        CtrlTypes.DirectControl,
        CtrlTypes.DirectControlW,
        CtrlTypes.Direct_Bearing
    ]

    num_connectivity_steps = len(connectivity_values)
    args_list = []

    for ctrl_type in control_types:
        for j in range(num_connectivity_steps):
            for k in range(num_runs_per_scenario): 
                args = {} 
                args['scenario_connectivity_index'] = j
                args['run_index'] = k
                args['Ts'] = 0.1 
                args['N'] = N_agent
                args['w_std'] = w_std_val
                args['w_std_orientation'] = w_std_orientation_val
                args['v_std'] = np.ones([N_agent, 1]) * v_std_val
                args['v_std_orientation'] = np.ones([N_agent, 1]) * v_std_orientation_val
                
                num_neighbors = connectivity_values[j]
                args['c'] = gen_directed_ring_adj(args['N'], num_neighbors)
                args['L'] = get_laplacian_mtx(args['c'])
                
                args['n'] = 5 
                args['p'] = 2 
                args['Q'] = np.kron(args['L'], np.eye(args['n'])) 
                args['R'] = np.eye(args['N']) 
                args['sim_n_step'] = 200
                args['thetaF'] = -0.8
                args['gain_file_name'] = 'gain_uneven' 
                args['ctrl_type'] = ctrl_type
                
                args_list.append(args)
    
    num_simulations = len(args_list)
    print(f"Total simulations to run: {num_simulations}")

    results_dict = {ctrl: [] for ctrl in control_types}
    ordered_results = [None] * num_simulations
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_processes) as executor:
        future_to_index = {executor.submit(run_simulation, args): i for i, args in enumerate(args_list)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            res = future.result()
            ordered_results[idx] = res
            
            completed_count += 1
            print(f"Progress: {completed_count} / {num_simulations} ({ (completed_count/num_simulations)*100:.1f}%)")

    for i, res in enumerate(ordered_results):
        if res is not None:
            ctrl_type = args_list[i]['ctrl_type']
            results_dict[ctrl_type].append(res)

    for ctrl_type in control_types:
        trajs_ctrl = []
        thetas_ctrl = []
        
        for res in results_dict[ctrl_type]:
            if res.get('trajs') is not None:
                trajs_ctrl.extend(res['trajs'])
                thetas_ctrl.extend(res['thetas'])
        
        if len(trajs_ctrl) > 0:
            print(f"\nProcessing results for: {ctrl_type}")
            np_trajs = np.array(trajs_ctrl)
            np_thetas = np.array(thetas_ctrl)
            
            plot_x_y_MCMC(
                np_trajs, 
                np_thetas, 
                control_type=ctrl_type,
                n=args_list[0]['n'], 
                N_agent=args_list[0]['N']
            )
    
    print("\nAll simulations and data saving processes are finished.")