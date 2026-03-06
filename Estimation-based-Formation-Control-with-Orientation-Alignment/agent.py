import numpy as np
import math
from utils import *
import pandas as pd
import os
from numba import njit
import cvxpy as cp
from scipy.linalg import block_diag  

@njit
def block_rotation(thetahat_i, thetahat_j, n, N):
    block_size = n * N
    big_block = np.zeros((block_size * N, block_size * N))

    for i in range(N):
        theta_i = thetahat_i[i, 0]
        tiled_i = np.full((N, 1), theta_i)
        theta_j = thetahat_j[i, 0]
        tiled_j = np.full((N, 1), theta_j)

        R = block_diagonal_rotation(tiled_i, tiled_j, n)

        for row in range(block_size):
            for col in range(block_size):
                big_block[i * block_size + row, i * block_size + col] = R[row, col]

    return big_block

@njit
def block_diagonal_rotation(thetas1, thetas2, n):
    N = thetas1.shape[0]
    block_size = n
    out = np.zeros((block_size * N, block_size * N))

    for i in range(N):
        theta1 = thetas1[i, 0]
        theta2 = thetas2[i, 0]

        T = rotation_matrix_2d(theta1, theta2)
        row_start = i * block_size
        col_start = i * block_size
        out[row_start:row_start+block_size, col_start:col_start+block_size] = T

    return out

@njit
def rotation_matrix_2d(theta_i, theta_j):
    cos_i, sin_i = np.cos(theta_i), np.sin(theta_i)
    cos_j, sin_j = np.cos(theta_j), np.sin(theta_j)

    R_i = np.array([[cos_i, -sin_i],
                    [sin_i,  cos_i]])
    R_j = np.array([[cos_j, -sin_j],
                    [sin_j,  cos_j]])

    R_ij = R_j.T @ R_i

    T = np.eye(5)
    T[0, 0] = R_ij[0, 0]
    T[0, 2] = R_ij[0, 1]
    T[0, 4] = 0
    T[2, 0] = R_ij[1, 0]
    T[2, 2] = R_ij[1, 1]
    T[2, 4] = 0

    T[1, 1] = R_ij[0, 0]
    T[1, 3] = R_ij[0, 1]
    T[3, 1] = R_ij[1, 0]
    T[3, 3] = R_ij[1, 1]

    return T

@njit
def inv_block_diagonal_rotation(thetas1, thetas2, n):
    """
    Constructs a block-diagonal matrix where each block is a 5×5 homogeneous transformation matrix
    from (theta2, p_bar2) to (theta1, p_bar1).
    """
    N = thetas1.shape[0]
    block_size = n
    out = np.zeros((block_size * N, block_size * N))

    for i in range(N):
        theta1 = thetas1[i, 0]
        theta2 = thetas2[i, 0]

        T = inv_rotation_matrix_2d( theta1,theta2)
        row_start = i * block_size
        col_start = i * block_size
        out[row_start:row_start+block_size, col_start:col_start+block_size] = T
    return out

@njit
def inv_rotation_matrix_2d(theta_i, theta_j):
    cos_i, sin_i = np.cos(theta_i), np.sin(theta_i)
    cos_j, sin_j = np.cos(theta_j), np.sin(theta_j)
    R_i = np.array([[cos_i, -sin_i],
                    [sin_i,  cos_i]])
    R_j = np.array([[cos_j, -sin_j],
                    [sin_j,  cos_j]])
    R_ij = R_j.T @ R_i

    T = np.eye(5)
    T[0, 0] = R_ij[0, 0]
    T[0, 2] = R_ij[0, 1]
    T[0, 4] = 0
    T[2, 0] = R_ij[1, 0]
    T[2, 2] = R_ij[1, 1]
    T[2, 4] = 0

    T[1, 1] = R_ij[0, 0]
    T[1, 3] = R_ij[0, 1]
    T[3, 1] = R_ij[1, 0]
    T[3, 3] = R_ij[1, 1]
    return T

@njit
def block_diagonal_transformation_with_pbar(thetas1, ts1, thetas2, ts2, n):
    """
    Constructs a block-diagonal matrix where each block is a 5×5 homogeneous transformation matrix
    from (theta2, p_bar2) to (theta1, p_bar1).
    """
    N = thetas1.shape[0]
    block_size = n
    out = np.zeros((block_size * N, block_size * N))

    for i in range(N):
        theta1 = thetas1[i, 0]
        theta2 = thetas2[i, 0]
        t1 = ts1[i*2:(i+1)*2, 0]
        t2 = ts2[i*2:(i+1)*2, 0]

        T = transformation_matrix_2d_with_pbar(theta1, t1[0], t1[1], theta2, t2[0], t2[1])
        row_start = i * block_size
        col_start = i * block_size
        out[row_start:row_start+block_size, col_start:col_start+block_size] = T

    return out

@njit
def transformation_matrix_2d_with_pbar(theta_i, px_i, py_i, theta_j, px_j, py_j):
    cos_i, sin_i = np.cos(theta_i), np.sin(theta_i)
    cos_j, sin_j = np.cos(theta_j), np.sin(theta_j)

    R_i = np.array([[cos_i, -sin_i],
                    [sin_i,  cos_i]])
    R_j = np.array([[cos_j, -sin_j],
                    [sin_j,  cos_j]])

    R_ij = R_j.T @ R_i
    t_ij = R_j.T @ (np.array([[px_i], [py_i]]) - np.array([[px_j], [py_j]]))

    T = np.eye(5)
    T[0, 0] = R_ij[0, 0]
    T[0, 2] = R_ij[0, 1]
    T[0, 4] = t_ij[0, 0]
    T[2, 0] = R_ij[1, 0]
    T[2, 2] = R_ij[1, 1]
    T[2, 4] = t_ij[1, 0]

    T[1, 1] = R_ij[0, 0]
    T[1, 3] = R_ij[0, 1]
    T[3, 1] = R_ij[1, 0]
    T[3, 3] = R_ij[1, 1]

    return T

class Agent:    
    def __init__(self, args): 
        self.N = args['N']
        self.id = args['id']
        self.thetahat = np.zeros((self.N, 1)) 
        self.relative_theta = np.zeros((self.N, 1)) 
        self.updated_states = np.zeros((self.N*3,1))
        self.updated_states_theta = np.zeros((self.N*3,1))
        self.ctrl_type = args['ctrl_type']        
        self.id = args['id'] 
        self.Ts = args['Ts']        
        self.n = args['n']         
        self.N = args['N']
        self.p = args['p'] 
        if self.n ==4 :
            self.A = np.array([
                [1, self.Ts, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, self.Ts],
                [0, 0, 0, 1]
            ])
            self.B = np.array([
                [self.Ts**2 / 2, 0],
                [self.Ts, 0],
                [0, self.Ts**2 / 2],
                [0, self.Ts]
            ])
        elif self.n ==5 :
            self.A = np.array([
                [1, self.Ts, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, self.Ts, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ])
            self.B = np.array([
                [self.Ts**2 / 2, 0],
                [self.Ts, 0],
                [0, self.Ts**2 / 2],
                [0, self.Ts],
                [0, 0]
            ])
        else:            
            assert 1==2
        self.Mbar_sum = []

        self.Mibar_sum = [] 
        self.ci_sum = [] 
    
        self.Atilde_sum = []
        self.Btilde_sum = []
        self.w_covs_sum = []
        self.v_covs_sum = []
        self.v_covs_list_sum =[]
        self.Hi_sum = []

        self.Mbar = []

        self.Mibar = [] 
        self.ci = [] 
    
        self.Atilde = []
        self.Btilde = []
        self.w_covs = []
        self.v_covs = []
        self.v_covs_list =[]
        self.Hi = []
        self.x = np.zeros([self.n,1])
        self.xhat = np.zeros([self.N*self.n,1])
        self.theta = np.zeros([1,1])
        self.thetas = np.zeros([self.N,1])
        self.relative_thetas = np.zeros([self.N,1])
        self.thetahat = np.zeros([self.N,1])
        self.xcov = np.eye((self.n+1)*self.N)  
        self.P = np.eye((self.n+1)*self.N)            
        self.u = np.zeros([self.p,1])   
        self.theta_u = np.zeros([1,1]) 
        self.cov = np.eye(self.n*self.N*self.N)
        self.cov_integration = np.eye((self.n+1)*self.N)
        self.theta_cov = np.eye(self.N*self.N)
        self.N = args['N']                 
        self.pos_weights = np.array([1.0, 1.0, 1.0, 1.0, 1e-8]) 
        self.w_std = args['w_std']
        self.w_std_orientation = args['w_std_orientation']
        self.w_cov = np.diagflat(np.kron( self.w_std**2, self.pos_weights))   
        self.v_std = args['v_std']
        self.v_std_orientation = args['v_std_orientation']
        self.v_cov = np.diagflat(np.kron( self.v_std**2,np.ones([self.n, 1])))
        self.v_cov_theta = np.diagflat(np.kron( self.v_std_orientation**2,np.ones([1, 1])))
        self.L = args['L']
        self.c = args['c']

        alpha = 0.01
        n = (self.n+1)*self.N
        kappa = 0
        self.controltype = args['ctrl_type']
        self.lambda_ = alpha**2 * (n + kappa) - n
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - alpha**2 + 2)
        self.Wc[1:] = 1 / (2 * (n + self.lambda_))

        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wm[1:] = 1 / (2 * (n + self.lambda_))

        self.Ci = np.zeros([int(self.n * np.sum(self.c[self.id,:])), self.n * self.N])
        r_count = 0
        for j in np.where(self.c[self.id] == 1)[0]:
            self.Ci[r_count * self.n:(r_count + 1) * self.n, j * self.n:(j + 1) * self.n] = np.eye(self.n)
            r_count += 1
        self.theta_Mbar = []
        self.x_mem = []
        self.theta_mem =[]
        self.xhat_mem = []
        self.thetahat_mem = []          
        self.z_mem = []      
        self.est_gain = self.Ci.transpose()             
        self.Hi = self.Ci 

        c_row = self.c[self.id]
        self.Hi = np.zeros((self.N*self.n,self.N*self.n))

        for agent_idx, can_measure in enumerate(c_row):
            if can_measure == 1:

                self.Hi[self.n * agent_idx:self.n * (agent_idx + 1),
                        self.n * agent_idx:self.n * (agent_idx + 1)] = np.eye(self.n)
        self.z = np.zeros([self.Ci.shape[0],1])

        self.pre_input = np.zeros([self.N*self.n,1])
        self.real_input = np.zeros([self.n,1])
        self.ci = np.zeros([self.n*self.N,self.n*self.N]) 
        self.ci[self.n * self.id:self.n * (self.id+1), self.n * self.id:self.n * (self.id+1)] = np.eye(self.n)
        self.Mi = np.zeros([self.p,self.N*self.p])
        self.Mi[:,self.p * self.id:self.p * (self.id+1)] = np.eye(self.p)
        self.Mibar = np.zeros([self.N*self.p,self.N*self.p])
        self.Mibar[self.p * self.id:self.p * (self.id+1), self.p * self.id:self.p * (self.id+1)] = np.eye(self.p)
        self.p1 = 1 
        self.theta_Mibar = np.zeros([self.N*self.p1,self.N*self.p1])
        self.theta_Mibar[self.p1 * self.id:self.p1 * (self.id+1), self.p1 * self.id:self.p1 * (self.id+1)] = 1
        self.F = None 
        self.F_partial = None
        self.offset = np.zeros([self.Ci.shape[0],1])
        
        for i in range(self.N):
            tmp_args = args.copy()
            tmp_args['id'] = i                   
            self.Mibar_sum.append(self.Mibar)
            self.ci_sum.append(self.ci)
            self.Mbar_sum.append(self.Mibar)            
            self.Atilde_sum.append(self.A)
            self.Btilde_sum.append(self.B)
            self.w_covs_sum.append(self.w_cov)
            self.v_covs_sum.append(self.v_cov)
            self.Hi_sum.append(self.Hi)
            self.theta_Mbar.append(self.theta_Mibar)
            
        self.Atilde_sum = block_diagonal_matrix(self.Atilde_sum)        
        self.Btilde_sum = block_diagonal_matrix(self.Btilde_sum)        
        self.Bbar_sum = np.kron(np.ones([1,self.N]),self.Btilde_sum)
        self.Mbar_sum = block_diagonal_matrix(self.Mbar_sum)
        self.w_covs_sum = block_diagonal_matrix(self.w_covs_sum)
        self.v_covs_list_sum = self.v_covs.copy()
        self.v_covs_sum = block_diagonal_matrix(self.v_covs_sum)
        self.thata_hat = np.zeros([self.N,1])
        self.relative_theta = np.zeros([self.N,1])
        self.theta_Mbar = block_diagonal_matrix(self.theta_Mbar)
        self.thetaF = args['thetaF']

    def set_offset(self,offset):
        self.offset =offset.copy()
        
    def set_MAS_info(self, Atilde, Btilde,w_covs, v_covs ):
        self.Atilde = Atilde 
        self.Btilde = Btilde 
        self.w_covs = w_covs 
        self.v_covs = v_covs
    
    def set_xhat(self, xhat):
        self.xhat = xhat.copy() 

    def set_thetahat(self,thetahat):
        self.thetahat = thetahat.copy() 

    def set_p_barhat(self,p_barhat):
        self.p_barhat = p_barhat.copy() 

    def set_thetas(self, xhat):
        self.relative_thetas = xhat.copy() 
    
    def est_step(self):
        xhat = np.vstack((self.xhat, self.thetahat))
        sigma_points = self.generate_sigma_points(xhat, self.P)
        sigma_points_pred = np.array([self.state_transition(sp) for sp in sigma_points])

        dim_linear = (self.n + 1) * self.N
        R_linear = np.eye(dim_linear) * self.v_cov[0,0]
        R_linear[-self.N:, -self.N:] = 1e-6
        xhat_predict, P_pred = self.predict_mean_and_covariance(sigma_points_pred, np.eye((self.n+1)*self.N)*self.w_std)
        self.est_gain, self.P = self.compute_est_gain_integration(P_pred, R_linear)

        xhat_end = self.n * self.N
        theta_end = xhat_end + self.N

        xhat = xhat_predict + np.dot(self.est_gain, np.dot(self.Hi,(self.z - xhat_predict[:xhat_end])))

        self.xhat = xhat[:xhat_end]
        self.thetahat = (xhat[xhat_end:theta_end] + np.pi) % (2 * np.pi) - np.pi

        self.xhat_mem.append(self.xhat)
        self.thetahat_mem.append(self.thetahat)
        self.z_mem.append(self.z.copy())
        return

    def compute_est_gain_integration(self,cov,R):
        J12 = np.zeros((self.N*self.n,self.N))
        Hi = np.block([
            [self.Hi, J12]
        ])
        opt_L = None
        L = self.compute_LC_integration(cov,Hi,R)
        eye_lc = np.eye(self.N*(self.n+1), self.N*(self.n+1))
        e_cov_next = (eye_lc - L @ Hi) @ cov
        #I_KH = eye_lc - L @ Hi
        #e_cov_next = I_KH @ cov @ I_KH.T + L @ R @ L.T # R 대신 실제 노이즈 공분산 사용
        opt_L = L.copy()       
        cov = e_cov_next.copy()
        return opt_L, cov

    def compute_LC_integration(self,p_cov,Hi,R):
        L = []
        S_i = Hi @ p_cov @ Hi.T + Hi @ R @ Hi.T
        jitter = 1e-6
        S_i_jitter = S_i + np.eye(len(S_i)) * jitter

        try:
            S_i_inv = la.solve(S_i_jitter, np.eye(S_i_jitter.shape[0]))
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(S_i_jitter)
            s_inv = np.diag(1.0 / s)
            S_i_inv = np.dot(Vt.T, np.dot(s_inv, U.T))

        L_i = p_cov @ Hi.T @ S_i_inv
        L = L_i
        return L

    def generate_sigma_points(self, x, P):
        n = x.shape[0]
        #P = (P + P.T) / 2
        P = P + np.eye(n) * 1e-6
        scale = n + self.lambda_
        if scale <= 0:
            raise ValueError(f"(n + lambda) must be > 0 for Cholesky. Got: {scale}")
        sqrt_P = np.linalg.cholesky(scale * P)

        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x.flatten()

        for i in range(n):
            sigma_points[i + 1] = x.flatten() + sqrt_P[:, i]
            sigma_points[n + i + 1] = x.flatten() - sqrt_P[:, i]

        return sigma_points
    
    def state_transition(self, x):
        """State and orientation transition model (f function in UKF)."""
        B = np.eye(self.N)*self.Ts
        A = np.eye(self.N)
        F = self.thetaF * self.L
        xhat_reshaped = x.reshape(((self.n+1) * self.N, 1))
        xhat_end = self.n * self.N
        theta_end = xhat_end + self.N

        thetahat  = xhat_reshaped[xhat_end:theta_end]     # (N, 1)
        
        thetahat_i = np.tile(thetahat[self.id],(self.N,1)).copy()
        thetahat_j = thetahat.copy()

        block_rotationi_j = block_rotation(thetahat_i, thetahat_j, self.n, self.N)
        A_rotation_i_j_inv = inv_block_diagonal_rotation(thetahat_i, thetahat_j, self.n)

        stacked_xhat = np.tile(xhat_reshaped[:xhat_end],(self.N,1))
        block_F = self.create_block_diag_F()
        stacked_offset = np.tile(self.offset,(self.N,1))
        
        state_propagation = self.Atilde @ xhat_reshaped[:xhat_end]
        state_input = A_rotation_i_j_inv @ self.Btilde @ block_F @ (np.dot(block_rotationi_j ,stacked_xhat) - stacked_offset)

        e_i = np.zeros((self.N, 1))
        e_i[self.id, 0] = 1.0
        one = np.ones((self.N, 1))
        Ci = one @ e_i.T  

        H = np.zeros((self.n, self.n))
        H[0, 0] = 1.0 
        H[2, 2] = 1.0 

        Ci_tilde = np.kron(Ci, H)

        theta_input = np.dot(B, np.dot(F, xhat_reshaped[xhat_end:theta_end]))
        theta_update = np.dot(A, xhat_reshaped[xhat_end:theta_end]) + theta_input

        thetahat_i_k = np.tile(thetahat[self.id],(self.N,1))
        thetahat_i_k_1 = np.tile(theta_update[self.id],(self.N,1))
        zeros = np.zeros((self.N*2,1))

        R_k_to_k_1 = block_diagonal_transformation_with_pbar(thetahat_i_k, zeros, thetahat_i_k + thetahat_i_k_1, zeros, self.n)

        state_update = state_propagation + state_input
        state_update_local = R_k_to_k_1 @ (np.eye(self.n * self.N) - Ci_tilde) @ state_update

        result = np.vstack((state_update_local, theta_update))

        return result    

    def predict_mean_and_covariance(self, sigma_points, Q):
        n = sigma_points.shape[1]
        x_mean = np.zeros((n, 1))
        for i in range(2 * n + 1):
            x_mean += self.Wm[i] * sigma_points[i].reshape(-1, 1)

        P_pred = Q.copy()
        for i in range(2 * n + 1):
            dx = (sigma_points[i].reshape(-1, 1) - x_mean)
            P_pred += self.Wc[i] * (dx @ dx.T)
        return x_mean, P_pred

    def theta_step(self, u = None):
        disturbances = np.random.normal(loc=0, scale=self.w_std_orientation, size=(1, 1))
        F = self.L[self.id,:]*(self.thetaF)
        input = np.dot(F,self.thetahat.copy())
        self.theta_u = input.copy()
        new_theta = self.theta + self.Ts*self.theta_u + disturbances.reshape(1,1)
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        self.theta = new_theta.copy()
        self.theta_mem.append(self.theta.copy())
        return
        
    def set_gain(self,F):
        self.F = F

    def set_partial_gain(self,F):
        self.F_partial = F

    def set_theta_gain(self,F):
        self.thetaF = F
        
    def set_measurement(self,x_all):        
        noise_scale = np.diag(np.sqrt(self.v_cov)).reshape(-1, 1)
        if self.controltype == CtrlTypes.DirectControlW:
            noise_scale = np.diag(np.sqrt(self.v_cov*0)).reshape(-1, 1)
        noise = np.random.normal(loc=0, scale=noise_scale)
        if self.n == 5:
            for i in range(self.N):
                noise[(i + 1) * self.n - 1, 0] = 0 
        moly = np.tile(self.theta, (self.N, 1))
        self.z = np.dot(np.linalg.inv(self.block_diagonal_rotation_original(moly.flatten())), x_all + noise)
        start_idx = self.id * self.n
        end_idx = (self.id + 1) * self.n
        z_i = self.z[start_idx:end_idx].copy()
        z_i[1] = 0
        z_i[3] = 0
        self.z = self.z - np.tile(z_i,(self.N,1)).copy()
        indices = np.arange(self.n - 1, self.N * self.n, self.n)
        self.z[indices] = 1

    def set_theta_measurement(self,x_all):     
        noise_scale = np.diag(np.sqrt(self.v_cov_theta)).reshape(-1, 1)   
        if self.controltype == CtrlTypes.Direct_Bearing or self.controltype == CtrlTypes.DirectControlW:
            noise_scale = np.diag(np.sqrt(self.v_cov_theta*0)).reshape(-1, 1)   
        noise = np.random.normal(loc=0, scale=noise_scale)
        self.thetas = x_all + noise

    def step(self, u = None): 
        holymoly = self.theta.copy()
        rotation = self.block_diagonal_rotation_2D(holymoly.flatten())
        scale = np.diag(np.sqrt(self.w_cov)).reshape(1, self.n)
        disturbances = np.random.normal(loc=0, scale=scale, size=(1, self.n))
        input = np.dot(rotation,np.dot(np.dot(self.Mi,self.F),self.xhat - self.offset)).copy()
        self.u = input.copy()
        new_x = np.dot(self.A,self.x) + np.dot(self.B, self.u)+ disturbances.reshape(self.n,1)
        self.x = new_x.copy()
        self.x_mem.append(self.x.copy())
        return
    
    def step_direct(self, u = None): 
        holymoly = self.theta.copy()
        rotation = self.block_diagonal_rotation_2D(holymoly.flatten())
        scale = np.diag(np.sqrt(self.w_cov)).reshape(1, self.n)        
        disturbances = np.random.normal(loc=0, scale=scale, size=(1, self.n))
        input = np.dot(rotation,np.dot(np.dot(self.Mi,self.F),self.z-self.offset)).copy()
        self.u = input.copy()
        new_x = np.dot(self.A,self.x) + np.dot(self.B, self.u)+ disturbances.reshape(self.n,1)
        self.x = new_x.copy()
        self.x_mem.append(self.x.copy())
        return

    def theta_step_direct(self, u = None):
        disturbances = np.random.normal(loc=0, scale=self.w_std_orientation, size=(1, 1))
        F = self.L[self.id,:]*(self.thetaF)
        input = np.dot(F,self.thetas.copy())
        self.theta_u = input.copy()
        new_theta = self.theta + self.Ts*self.theta_u + disturbances.reshape(1,1)
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        self.theta = new_theta.copy()
        self.theta_mem.append(self.theta.copy())
        return

    def theta_step_Bearing_direct(self, u = None):
        c = self.c[self.id,:]
        disturbances = np.random.normal(loc=0, scale=self.w_std_orientation, size=(1, 1))
        total_theta = c @ self.relative_thetas
        input = -total_theta*self.thetaF
        self.theta_u = input.copy()
        new_theta = self.theta + self.Ts*self.theta_u + disturbances.reshape(1,1)
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        self.theta = new_theta.copy()
        self.theta_mem.append(self.theta.copy())
        return

    def block_diagonal_rotation_original(self, thetas):
        """Create a block diagonal matrix of 2D or 4D rotation matrices for a list of angles."""
        n = len(thetas)
        block_size = self.n
        block_matrix = np.zeros((block_size * n, block_size * n))
        for i, theta in enumerate(thetas):
            if isinstance(theta, np.ndarray):
                theta = theta.item()
            R = self.rotation_matrix_2d_original(theta) 
            block_matrix[block_size * i:block_size * (i+1), block_size * i:block_size * (i+1)] = R
        return block_matrix
    
    def rotation_matrix_2d_original(self, theta):
        """Create a rotation matrix for 2D, 4D, or 5D states."""
        if isinstance(theta, np.ndarray):
            theta = float(theta)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        if self.n == 2:
            return np.array([
                [cos_theta, -sin_theta],
                [sin_theta,  cos_theta]
            ])
        elif self.n == 4:
            return np.array([
                [cos_theta, 0, -sin_theta, 0],
                [0, cos_theta, 0, -sin_theta],
                [sin_theta, 0, cos_theta, 0],
                [0, sin_theta, 0, cos_theta]
            ])
        elif self.n == 5:
            return np.array([
                [cos_theta, 0, -sin_theta, 0, 0],
                [0, cos_theta, 0, -sin_theta, 0],
                [sin_theta, 0, cos_theta, 0, 0],
                [0, sin_theta, 0, cos_theta, 0],
                [0, 0, 0, 0, 1]
            ])
        else:
            raise ValueError("Invalid self.n: should be 2, 4, or 5.")

    def block_diagonal_rotation_2D(self, thetas):
        """Create a block diagonal matrix of 2D rotation matrices for a list of angles."""
        n = len(thetas)
        block_size = 2
        block_matrix = np.zeros((block_size * n, block_size * n))
        for i, theta in enumerate(thetas):
            R = self.rotation_matrix_2d_2D(theta).squeeze() 
            block_matrix[block_size * i:block_size * (i+1), block_size * i:block_size * (i+1)] = R
        return block_matrix

    def rotation_matrix_2d_2D(self, theta):
        """Create a 2D or 4D rotation matrix depending on the value of self.n."""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        return np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

    def create_block_diag_F(self):
        blocks = []
        if self.F.shape[1] == 50: 
            block_width = 50
        elif self.F.shape[1] == 20:  
            block_width = 20
        elif self.F.shape[1] == 25:  
            block_width = 25
        elif self.F.shape[1] == 10:
            block_width = 10
        else:
            raise ValueError("Unexpected self.F shape detected")

        for i in range(self.N):
            Fi = self.F[i * 2: (i * 2) + 2, :]  
            blocks.append(Fi)

        block_diag_F = np.zeros((2 * self.N, block_width * self.N))

        for i in range(self.N):
            block_diag_F[i * 2: (i + 1) * 2, i * block_width: (i + 1) * block_width] = blocks[i]

        return block_diag_F
    
    def get_x(self):
        return self.x.copy()

    def get_theta(self):
        return self.theta.copy()

    def get_input(self):
        return self.u.copy()
    
    def get_theta_input(self):
        return self.theta_u.copy()

    def set_input(self,input):
        self.u = input
    
    def set_x(self,state):
        self.x = state

    def set_theta(self,theta):
        self.theta = np.array([theta])

    def get_traj(self):
        traj = None
        if len(self.x_mem) > 0:
            traj = np.array(self.x_mem)
        return traj.copy()

    def get_thetas(self):
        traj = []
        if len(self.theta_mem) > 0:
            traj = np.array(self.theta_mem)
        return traj.copy()