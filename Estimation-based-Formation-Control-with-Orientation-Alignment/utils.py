import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import copy
from scipy import linalg as la
from numpy.linalg import det
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import os

class CtrlTypes(Enum):
    # LQROutputFeedback = 0
    # SubOutpFeedback = 1 
    # CtrlEstFeedback = 2
    # LQGFeedback = 3
    # COMLQG = 4
    # DirectControl = 5
    # BearingOnly = 6
    # KF = 7
    # Direct_Bearing = 8
    # DirectControlW = 9

    CtrlEstFeedback = 0
    DirectControl = 1
    Direct_Bearing = 2
    DirectControlW = 3


def matrixEquationSolver(A, B, F):
    if not isinstance(A, list) or not isinstance(B, list):
        raise ValueError("Defining matrices are not in a list.")

    if len(A) != len(B):
        raise ValueError("Ambiguous number of terms in the matrix equation.")

    nA, mA = A[0].shape
    nB, mB = B[0].shape

    if nA != mA or nB != mB:
        raise ValueError("Rectangular matrices are not allowed.")

    maxSize = 5000

    if nA * nB > maxSize:
        raise MemoryError("A very large matrix will be formed.")

    C = np.zeros((nA * nB, nA * nB), dtype=A[0].dtype)
    for j in range(len(A)):
        C += np.kron(B[j], A[j])

    x = np.linalg.solve(C, F.T.ravel())
    C_inv = np.linalg.solve(C, np.eye(C.shape[0]))
    X = x.reshape(nB, nA).T

    return X
    
def get_laplacian_mtx(adj_mtx):
    laplacian_mtx = np.zeros(adj_mtx.shape)
    D_mtx = np.diagflat([np.sum(adj_mtx,axis=0)]) 
    laplacian_mtx =D_mtx - adj_mtx    
    return laplacian_mtx
    
def gen_directed_ring_adj(N, k_neighbors):
    adj_mtx = np.zeros((N, N), dtype=int)
    for i in range(N):
        adj_mtx[i, i] = 1
        adj_mtx[i, (i - 1) % N] = 1
        for j in range(1, k_neighbors): 
            adj_mtx[i, (i + j) % N] = 1
    return adj_mtx    

def block_diagonal_matrix(matrix_list):
    num_matrices = len(matrix_list)
    row_sizes = [matrix.shape[0] for matrix in matrix_list]
    col_sizes = [matrix.shape[1] for matrix in matrix_list]
    block_diag_size = (sum(row_sizes), sum(col_sizes))
    block_diag_matrix = np.zeros(block_diag_size)
    row_idx = 0
    col_idx = 0
    for i in range(num_matrices):
        row_size = row_sizes[i]
        col_size = col_sizes[i]

        block_diag_matrix[row_idx:row_idx + row_size, col_idx:col_idx + col_size] = matrix_list[i]

        row_idx += row_size
        col_idx += col_size

    return block_diag_matrix

def plot_x_y(traj_list, theta_list, n=5):
    traj_list = copy.deepcopy(traj_list)
    theta_list = copy.deepcopy(theta_list)

    if n == 5 or n == 4:
        x_data = np.concatenate([traj[:, 0] for traj in traj_list])
        y_data = np.concatenate([traj[:, 2] for traj in traj_list])
    else:
        x_data = np.concatenate([traj[:, 0] for traj in traj_list])
        y_data = np.concatenate([traj[:, 1] for traj in traj_list])

    min_x, max_x = x_data.min(), x_data.max()
    min_y, max_y = y_data.min(), y_data.max()

    padding_x = 0.1 * (max_x - min_x) if max_x - min_x != 0 else 1
    padding_y = 0.1 * (max_y - min_y) if max_y - min_y != 0 else 1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectories in X-Y Plane with Orientation Arrows')

    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)

    trajectory_color = 'gray'

    N = len(theta_list)
    for idx, (traj, thetas) in enumerate(zip(traj_list, theta_list)):
        if n == 5 or n == 4:
            x = traj[:, 0]
            y = traj[:, 2]
        else:
            x = traj[:, 0]
            y = traj[:, 1]
        ax.plot(x, y, color=trajectory_color, linewidth=2.0)
        ax.plot(x[0], y[0], 'o', markerfacecolor='none', markeredgecolor='red',
                markersize=10, markeredgewidth=2.0)
        ax.plot(x[-1], y[-1], 'o', markerfacecolor='none', markeredgecolor='blue',
                markersize=10, markeredgewidth=2.0)

        for step, color in zip([0, len(traj) - 1], ['red', 'blue']):
            x_arrow = x[step]
            y_arrow = y[step]
            theta = thetas[step]
            dx = 6.0 * np.cos(theta)
            dy = 6.0 * np.sin(theta)
            ax.quiver(x_arrow, y_arrow, dx, dy, angles='xy', scale_units='xy', scale=4, 
                      color=color, width=0.005, headwidth=5, headlength=6, alpha=0.8)

    start_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                 markersize=10, markerfacecolor='none', markeredgewidth=2.0, label='Initial Formation')
    end_marker = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                               markersize=10, markerfacecolor='none', markeredgewidth=2.0, label='Final Formation')
    ax.legend(handles=[start_marker, end_marker], loc='upper right', fontsize=12)

    ax.grid(True, linestyle='-', linewidth=1.5, color='gray')

    plt.tight_layout()

    df = pd.DataFrame()

    for i, (traj, thetas) in enumerate(zip(traj_list, theta_list)):
        if n == 5:
            x = traj[:, 0].squeeze()
            x_dot = traj[:, 1].squeeze()
            y = traj[:, 2].squeeze()
            y_dot = traj[:, 3].squeeze()
        elif n == 4:
            x = traj[:, 0].squeeze()
            x_dot = traj[:, 1].squeeze()
            y = traj[:, 2].squeeze()
            y_dot = traj[:, 3].squeeze()
        else:
            x = traj[:, 0].squeeze()
            x_dot = np.zeros_like(x)
            y = traj[:, 1].squeeze()
            y_dot = np.zeros_like(y)

        theta = np.array(thetas).squeeze()

        df[f'Agent{i+1}.x'] = x
        df[f'Agent{i+1}.x_dot'] = x_dot
        df[f'Agent{i+1}.y'] = y
        df[f'Agent{i+1}.y_dot'] = y_dot
        df[f'Agent{i+1}.theta'] = theta
    plt.show()

def plot_x_y_MCMC(traj_list, theta_list, control_type, n=5, N_agent=10):
    if traj_list is None or len(traj_list) == 0:
        print("No data found to plot.")
        return

    # 1. 폴더 생성 로직
    # control_type이 Enum인 경우 .name을 쓰고, 문자열인 경우 그대로 사용
    folder_name = control_type.name if hasattr(control_type, 'name') else str(control_type)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder created: {folder_name}")

    traj_list = copy.deepcopy(traj_list)
    theta_list = copy.deepcopy(theta_list)
    N_total = len(traj_list)
    num_scenarios = N_total // N_agent

    print(f"Saving {num_scenarios} scenario files into folder '{folder_name}'...")

    for s_idx in range(num_scenarios):
        df = pd.DataFrame()
        start, end = s_idx * N_agent, (s_idx + 1) * N_agent
        s_trajs = traj_list[start:end]
        s_thetas = theta_list[start:end]

        for a_idx, (traj, th) in enumerate(zip(s_trajs, s_thetas)):
            prefix = f'Agent{a_idx+1}'
            df[f'{prefix}.x'] = traj[:, 0].flatten()
            if n >= 4:
                df[f'{prefix}.x_dot'] = traj[:, 1].flatten()
                df[f'{prefix}.y'] = traj[:, 2].flatten()
                df[f'{prefix}.y_dot'] = traj[:, 3].flatten()
            else:
                df[f'{prefix}.y'] = traj[:, 1].flatten()
            
            th_array = np.array(th).flatten().astype(float)
            target_len = len(df)
            if len(th_array) > target_len:
                th_array = th_array[:target_len]
            elif len(th_array) < target_len:
                th_array = np.pad(th_array, (0, target_len - len(th_array)), 'edge')
            df[f'{prefix}.theta'] = th_array

        # 2. 파일 저장 경로 수정
        excel_filename = f"Test{s_idx + 1}.xlsx"
        file_path = os.path.join(folder_name, excel_filename) # 폴더명과 파일명 결합
        df.to_excel(file_path, index=False)
        print(f"Saved: {file_path} (Rows: {len(df)})")

    # # 그래프 출력
    # fig, ax = plt.subplots(figsize=(10, 8))
    # y_idx = 2 if n >= 4 else 1
    # for traj in traj_list:
    #     ax.plot(traj[:, 0].flatten(), traj[:, y_idx].flatten(), color='gray', alpha=0.3)
    # plt.title(f"MCMC Trajectories - {folder_name} (Total: {N_total} Agents)")
    # plt.show()

def plot_theta(theta_list):
    print("Starting plot_theta...")
    print("theta_list shapes:", [t.shape for t in theta_list])
    
    time_steps = np.arange(theta_list[0].shape[0])
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for traj in theta_list:
        traj_values = traj.flatten()
        ax.plot(time_steps, traj_values)
    
    ax.set_ylim([-3.14, 3.14])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Orientation')
    ax.set_title('Orientation Trajectories of Agents')
    ax.grid(True)

    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    plt.show()

def get_formation_offset_vector_circle(N, n, dist=1.0):    
    rel_pos = np.zeros((N, 2))
    radius = dist / (2 * np.sin(np.pi / N))

    for i in range(N):
        theta = 2 * np.pi / N * i 
        rel_pos[i, :] = [radius * np.cos(theta), radius * np.sin(theta)]

    xref_tmp = np.zeros((N, N, 2))
    xref = np.zeros((n * N, N))
    offset_mtx = np.zeros((n * N, 1))

    for i in range(N):
        for j in range(N):
            xref_tmp[i, j, :] = rel_pos[i, :] - rel_pos[j, :]
            if n == 4:
                xref[j * n:j * n + n, i] = [xref_tmp[i, j, 0], 0, xref_tmp[i, j, 1], 0]
            elif n == 5:
                xref[j * n:j * n + n, i] = [xref_tmp[i, j, 0], 0, xref_tmp[i, j, 1], 0, 0]
            elif n == 2:
                xref[j * n:j * n + n, i] = [xref_tmp[i, j, 0], xref_tmp[i, j, 1]]
            else:
                assert 1 == 2

    xref = -1 * xref

    for id in range(N):
        if n == 4:
            offset_mtx[id * n:id * n + n, 0] = [rel_pos[id, 0], 0, rel_pos[id, 1], 0]
        elif n == 5:
            offset_mtx[id * n:id * n + n, 0] = [rel_pos[id, 0], 0, rel_pos[id, 1], 0, 0]
        elif n == 2:
            offset_mtx[id * n:id * n + n, 0] = [rel_pos[id, 0], rel_pos[id, 1]]
        else:
            assert 1 == 2

    return offset_mtx