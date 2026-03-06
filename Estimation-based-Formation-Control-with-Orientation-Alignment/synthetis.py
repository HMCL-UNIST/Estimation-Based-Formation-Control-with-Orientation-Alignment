import numpy as np
import os 
import pickle
from agent import Agent
from utils import *
from scipy.linalg import solve_discrete_are

class ControlEstimationSynthesis:    
    def __init__(self, args):        
        self.args = args
        self.Ts = args['Ts']
        self.N = args['N']
        self.n = args['n']
        self.p = args['p']
        self.Q = args['Q']
        self.R = args['R']
        self.adj = args['c']    

        self.Q_tilde = self.Q 
        self.R_tilde = np.kron(self.R, np.eye(self.p))        
        self.agents = []
        
        self.Atilde = []
        self.Btilde = []
        self.w_covs = []
        self.v_covs = []
        
        for i in range(self.N):
            tmp_args = args.copy()
            tmp_args['id'] = i
            tmp_agent = Agent(tmp_args)               
            self.agents.append(tmp_agent)         
            self.Atilde.append(tmp_agent.A)
            self.Btilde.append(tmp_agent.B)
            self.w_covs.append(tmp_agent.w_cov)
            self.v_covs.append(tmp_agent.v_cov)
            
        self.Atilde = block_diagonal_matrix(self.Atilde)        
        self.Btilde = block_diagonal_matrix(self.Btilde)        
        self.w_covs = block_diagonal_matrix(self.w_covs)
        self.v_covs = block_diagonal_matrix(self.v_covs)

        file_name_ = args.get('gain_file_name', 'gains.pkl')
        data_load = self.load_gains(file_name=file_name_)   
        
        if data_load:
            self.lqr_gain = self.data['lqr_gain']
            print(f"Gains loaded from {file_name_}")
        else:                                   
            self.lqr_gain = self.compute_lpr_gain()              
            print('LQR solution found and saving...')     
            self.save_gains(file_name=file_name_)
     
    def load_gains(self, file_name='gains.pkl'):                     
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.data = pickle.load(file)                                          
            return True
        return False
    
    def save_gains(self, file_name='gains.pkl'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        data = {
            'lqr_gain': self.lqr_gain,
            'w_covs': self.w_covs,
            'v_covs': self.v_covs,
            'adj_matrix': self.adj,
            'args': self.args
        }
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
            print(f"File '{file_path}' Saved.")

    def compute_lpr_gain(self):      
        if self.n == 5:
            A_lqr = np.array([
                [1, self.Ts, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, self.Ts],
                [0, 0, 0, 1]
            ])
            B_lqr = np.array([
                [self.Ts**2 / 2, 0],
                [self.Ts, 0],
                [0, self.Ts**2 / 2],
                [0, self.Ts]
            ])
            
            current_A = np.kron(np.eye(self.N), A_lqr)
            current_B = np.kron(np.eye(self.N), B_lqr)
            idx_4n = [i for i in range(self.N * 5) if i % 5 != 4]
            current_Q = self.Q_tilde[np.ix_(idx_4n, idx_4n)]
            current_R = self.R_tilde.copy()
            
        else:
            current_A = self.Atilde
            current_B = self.Btilde
            current_Q = self.Q_tilde
            current_R = self.R_tilde.copy()

        max_attempts = 100
        K_raw = None
        R_reg = current_R.copy()

        for attempt in range(max_attempts):
            try:
                S_ = solve_discrete_are(current_A, current_B, current_Q, R_reg)
                K_raw = np.linalg.inv(R_reg + current_B.T @ S_ @ current_B) @ (current_B.T @ S_ @ current_A)
                break 
            except Exception:            
                print(f"Attempt {attempt + 1}: Failed to solve DARE. Adding regularization.")                               
                R_reg += np.diagflat(np.random.rand(R_reg.shape[0]) * 0.01)                
        
        if K_raw is None:
            raise ValueError("Maximum attempts reached. Unable to solve DARE.")
            
        gain = -1 * K_raw

        if self.n == 5:
            gain_5n = np.zeros((gain.shape[0], 5 * self.N))
            for i in range(self.N):
                block = gain[:, i * 4 : (i + 1) * 4]
                block_expanded = np.insert(block, 4, 0, axis=1)
                gain_5n[:, i * 5 : (i + 1) * 5] = block_expanded
            gain = gain_5n

        return gain