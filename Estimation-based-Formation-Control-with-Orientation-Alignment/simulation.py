import numpy as np
from agent import Agent
from utils import *
import control as ct
from synthetis import ControlEstimationSynthesis
import concurrent.futures
from eval import MASEval
import pandas as pd
import traceback
import time

class MASsimulation:    
    def __init__(self, args): 
        self.args = args
        self.Ts = args['Ts']
        self.N = args['N']
        self.w_std = args['w_std']
        self.v_std = args['v_std']
        self.L = args['L']
        self.n = args['n']
        self.p = args['p']
        self.Q = args['Q']
        self.R = args['R']
        sim_step = args['sim_n_step']
        self.ctrl_type = args['ctrl_type']
        
        self.offset = get_formation_offset_vector_circle(self.N, self.n, dist=1) 
        
        self.synthesis = ControlEstimationSynthesis(args)
        self.eval = MASEval(args)
        
        self.agents = self.synthesis.agents
        
        self.init_MAS()     
        
        self.X = np.zeros([self.N*self.n, 1])
        self.thetas = np.zeros([self.N, 1])
        self.get_states()
        self.get_thetas()
        self.run_sim(sim_step)
        self.eval_ready()    

    def run_sim(self, num_time_steps):
        print(f"Simulation Loop Started: {num_time_steps} steps")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
                for time_step in range(num_time_steps):
                    print("this is time step", time_step)
                    self.get_states()
                    self.get_thetas()
                    futures = []
                    for agent in self.agents:
                        futures.append(executor.submit(agent.set_measurement, self.X))  
                        futures.append(executor.submit(agent.set_theta_measurement, self.thetas))                                              
                
                    concurrent.futures.wait(futures)
                    a = time.time()
                    if time_step == 0:
                        for i in range(self.N):
                            self.agents[i].set_MAS_info(self.synthesis.Atilde, self.synthesis.Btilde, 
                                                       self.synthesis.w_covs, self.synthesis.v_covs)                        
                            self.agents[i].set_xhat(self.X.copy())
                            self.agents[i].set_thetahat(np.zeros((self.N, 1)))

                    if self.ctrl_type == CtrlTypes.CtrlEstFeedback:
                        f_est = [executor.submit(agent.est_step) for agent in self.agents]
                        concurrent.futures.wait(f_est)
                        
                        f_step = [executor.submit(agent.step) for agent in self.agents]
                        concurrent.futures.wait(f_step)
                        
                        f_theta = [executor.submit(agent.theta_step) for agent in self.agents]
                        concurrent.futures.wait(f_theta)
                        
                    elif self.ctrl_type == CtrlTypes.DirectControl:
                        f_step = [executor.submit(agent.step_direct) for agent in self.agents]
                        concurrent.futures.wait(f_step)
                        
                        f_theta = [executor.submit(agent.theta_step_direct) for agent in self.agents]
                        concurrent.futures.wait(f_theta)

                    elif self.ctrl_type == CtrlTypes.DirectControlW:
                        f_step = [executor.submit(agent.step_direct) for agent in self.agents]
                        concurrent.futures.wait(f_step)
                        
                        f_theta = [executor.submit(agent.theta_step_direct) for agent in self.agents]
                        concurrent.futures.wait(f_theta)
            
                    elif self.ctrl_type == CtrlTypes.Direct_Bearing:
                        def wrap_pi(x): return (x + np.pi) % (2*np.pi) - np.pi
                        all_z = np.hstack([agent.z.copy() for agent in self.agents])
                        pos_idx = [0, 2] if self.n >= 4 else [0, 1]
                        
                        bearing = np.zeros((self.N, self.N))
                        for i in range(self.N):
                            col_i = all_z[:, i].reshape(self.N, self.n)
                            pos_i_frame = col_i[:, pos_idx]
                            pi = pos_i_frame[i]
                            for j in range(self.N):
                                rel = pos_i_frame[j] - pi
                                bearing[i, j] = wrap_pi(np.arctan2(rel[1], rel[0]))
                            
                        new_rel_mat = np.zeros((self.N, self.N))
                        if not hasattr(self, 'rel_mat_storage'): 
                            self.rel_mat_storage = np.zeros((self.N, self.N))
                            
                        for i in range(self.N):
                            for j in range(self.N):
                                if i == j: continue
                                if np.random.rand() < 0.3: 
                                    new_rel_mat[i, j] = self.rel_mat_storage[i, j]
                                else: 
                                    new_rel_mat[i, j] = wrap_pi(bearing[i, j] - bearing[j, i] + np.pi)
                        self.rel_mat_storage = new_rel_mat.copy()
                            
                        for i in range(self.N):
                            self.agents[i].set_thetas(new_rel_mat[i].reshape(-1, 1))
                            
                        f_step = [executor.submit(agent.step_direct) for agent in self.agents]
                        concurrent.futures.wait(f_step)
                        
                        f_theta = [executor.submit(agent.theta_step_Bearing_direct) for agent in self.agents]
                        concurrent.futures.wait(f_theta)
                    b= time.time()

        except Exception:
            traceback.print_exc()

    def eval_ready(self):
        try:
            self.eval.trajs = [np.array(a.get_traj(), dtype=float) for a in self.agents]
            
            self.eval.thetas = []
            for a in self.agents:
                raw_data = getattr(a, 'theta_mem', [])
                clean_data = [float(np.array(val).item()) if isinstance(val, (np.ndarray, list)) 
                             else float(val) for val in raw_data]
                self.eval.thetas.append(np.array(clean_data, dtype=float))

            self.eval.est_trajs = self.eval.trajs
            self.eval.n = self.n
            print(f"[eval_ready] Data collected for {len(self.agents)} agents.")
        except Exception as e:
            traceback.print_exc()

    def expand_lqr_gain_with_zero_z(self, gain_4n, N):
        gain_5n = np.zeros((2 * N, 5 * N))
        for i in range(N):
            block = gain_4n[:, i * 4:(i + 1) * 4]  # shape (2, 4)
            # [px, vx, py, vy] → [px, vx, py, vy, pz=0]
            block_expanded = np.insert(block, 4, 0, axis=1)  # insert zero at pz position
            gain_5n[:, i * 5:(i + 1) * 5] = block_expanded
        return gain_5n

    def get_inputs(self): self.U = np.vstack([a.get_input() for a in self.agents])
    def get_states(self): self.X = np.vstack([a.get_x() for a in self.agents]).copy()        
    def get_thetas(self): self.thetas = np.vstack([a.get_theta() for a in self.agents]).copy()  

    def init_MAS(self):
        try:
            gain = self.synthesis.lqr_gain
            F = np.array([[gain[0,0]/(self.N-1), gain[0,1]/(self.N-1), 0, 0],
                        [0, 0, gain[0,0]/(self.N-1), gain[0,1]/(self.N-1)]])
            partial_gain = np.kron(self.L, F)

            if self.n == 5:
                partial_gain = self.expand_lqr_gain_with_zero_z(partial_gain, self.N)

            thetas_init = np.linspace(0.0, 1.0, self.N)
            full_state = (np.random.randn(self.N * self.n, 1) - 0.5) * 1.0
            
            for i in range(self.N):
                start, end = i * self.n, (i + 1) * self.n
                tmp_state = full_state[start:end].copy()
                if self.n == 5: 
                    tmp_state[-1, 0] = 1.0
                
                self.agents[i].set_x(tmp_state)
                self.agents[i].set_theta(thetas_init[i])
                self.agents[i].set_gain(partial_gain)
                self.agents[i].set_offset(self.offset)
                
                self.agents[i].x_mem = [tmp_state.copy()]
                self.agents[i].theta_mem = [thetas_init[i]]
        except Exception as e:
            traceback.print_exc()
            raise e

if __name__ == "__main__": 
    args = {} 
    args['Ts'] = 0.1
    N_agent = 10
    num_neighbors = 2
    args['N'] = N_agent 
    args['w_std'] = 0.03
    args['w_std_orientation'] = 0.03
    args['v_std'] = np.ones([N_agent,1])*0.3
    args['v_std_orientation'] = np.ones([N_agent,1])*0.4
    args['c'] = gen_directed_ring_adj(args['N'], num_neighbors)
    args['L'] = get_laplacian_mtx(args['c']) 
    args['n'] = 5
    args['p'] = 2
    args['Q'] = np.kron(args['L'], np.eye(args['n'])) 
    args['R'] = np.eye(N_agent) 
    args['sim_n_step'] = 200
    args['gain_file_name'] = 'gain_uneven' 
    args['ctrl_type'] = CtrlTypes.CtrlEstFeedback
    #args['ctrl_type'] = CtrlTypes.DirectControl
    #args['ctrl_type'] = CtrlTypes.Direct_Bearing
    #args['ctrl_type'] = CtrlTypes.DirectControlW
    args['thetaF'] = -0.8
    
    obj = MASsimulation(args)
    obj.eval.eval_init()