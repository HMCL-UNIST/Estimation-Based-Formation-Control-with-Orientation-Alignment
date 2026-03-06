import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

FOLDER_CONFIG = {
    "Proposed Method (UKF)": "./CtrlEstFeedback",
    "Communication-based": "./Direct_Bearing",
    "Direct (with noise)": "./DirectControl",
    "Direct (no noise)": "./DirectControlW"
}

NUM_CONNECTIVITY_STEPS = 8
NUM_RUNS_PER_SCENARIO = 20

TOTAL_SCENARIOS = NUM_CONNECTIVITY_STEPS

N_AGENT = 10      
N_STATE = 4  # 2-D position and velocity
DIST = 1.0        

START_STEP = 150

CONNECTIVITY_LIST = list(range(2, 10)) 
if len(CONNECTIVITY_LIST) != NUM_CONNECTIVITY_STEPS:
    print(f"NUM_CONNECTIVITY_STEPS =! CONNECTIVITY_LIST")

def calculate_displacement_errors(df, N, n, dist):
    timesteps = df.shape[0]
    if n == 4 and f'Agent1.x_dot' not in df.columns:
        n = 2
        
    actual_positions = np.zeros((timesteps, N, n))
    for i in range(N):
        if n == 4:
            actual_positions[:, i, 0] = df[f'Agent{i+1}.x'].values
            actual_positions[:, i, 1] = df[f'Agent{i+1}.x_dot'].values
            actual_positions[:, i, 2] = df[f'Agent{i+1}.y'].values
            actual_positions[:, i, 3] = df[f'Agent{i+1}.y_dot'].values
        elif n == 2:
            actual_positions[:, i, 0] = df[f'Agent{i+1}.x'].values
            actual_positions[:, i, 1] = df[f'Agent{i+1}.y'].values

    thetas = np.zeros((timesteps, N))
    for i in range(N):
        thetas[:, i] = df[f'Agent{i+1}.theta'].values

    radius = dist / (2 * np.sin(np.pi / N))
    rel_pos = np.zeros((N, 2))
    for i in range(N):
        theta = 2 * np.pi / N * i
        rel_pos[i, :] = [radius * np.cos(theta), radius * np.sin(theta)]
        
    xref_tmp = np.zeros((N, N, 2))
    for i in range(N):
        for j in range(N):
            xref_tmp[i, j, :] = rel_pos[j, :] - rel_pos[i, :]

    errors = np.zeros((timesteps, N))
    for t in range(timesteps):
        mean_cos = np.cos(thetas[t, :]).mean()
        mean_sin = np.sin(thetas[t, :]).mean()
        theta_a = np.arctan2(mean_sin, mean_cos)
        c, s = np.cos(theta_a), np.sin(theta_a)
        R_T = np.array([[ c,  s], [-s,  c]])

        for i in range(N):
            agent_error = 0.0
            for j in range(N):
                if i == j: continue
                p_i = actual_positions[t, i, [0, 2]] if n==4 else actual_positions[t, i, :]
                p_j = actual_positions[t, j, [0, 2]] if n==4 else actual_positions[t, j, :]
                p_ji_a = R_T @ (p_j - p_i)
                position_error = np.linalg.norm(p_ji_a - xref_tmp[i, j, :])

                if n == 4:
                    v_ji_a = R_T @ (actual_positions[t, j, [1, 3]] - actual_positions[t, i, [1, 3]])
                    velocity_error = np.linalg.norm(v_ji_a)
                else:
                    velocity_error = 0.0

                agent_error += position_error + velocity_error
            errors[t, i] = agent_error
    return errors

def calculate_orientation_alignment_errors(df, N):
    timesteps = df.shape[0]
    theta_errors = np.zeros((timesteps, N))
    for t in range(timesteps):
        for i in range(N):
            err_sum = 0
            for j in range(N):
                delta_theta = df[f'Agent{j+1}.theta'].values[t] - df[f'Agent{i+1}.theta'].values[t]
                err_sum += np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
            theta_errors[t, i] = np.linalg.norm(err_sum)
    return theta_errors

def process_folder_with_large_std(folder_path, num_runs_per_scenario, 
                                 num_total_scenarios, num_connectivity_steps, 
                                 connectivity_list, N, n, dist, start_step):
    results = {
        "f_mean": [], "f_std": [], "o_mean": [], "o_std": [],
        "conn": [], "noise": []
    }
    
    if not os.path.exists(folder_path):
        return results

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx") and f.startswith("Test")]
    scenario_file_map = {}
    for filename in all_files:
        match = re.search(r'Test(\d+)\.xlsx', filename)
        if match:
            idx = (int(match.group(1)) - 1) // num_runs_per_scenario
            scenario_file_map.setdefault(idx, []).append(filename)

    for i in tqdm(range(num_total_scenarios), desc=f"처리 중: {os.path.basename(folder_path)}"):
        if i not in scenario_file_map: continue
            
        raw_pool_f, raw_pool_o = [], []
        run_means_f, run_means_o = [], []

        for filename in scenario_file_map[i]:
            try:
                df = pd.read_excel(os.path.join(folder_path, filename))
                f_series = calculate_displacement_errors(df, N, n, dist).mean(axis=1)[start_step:]
                o_series = calculate_orientation_alignment_errors(df, N).mean(axis=1)[start_step:]
                
                raw_pool_f.extend(f_series)
                raw_pool_o.extend(o_series)
                run_means_f.append(np.nanmean(f_series))
                run_means_o.append(np.nanmean(o_series))
            except Exception: continue

        if run_means_f:
            results["f_mean"].append(np.mean(run_means_f))
            results["o_mean"].append(np.mean(run_means_o))
            results["f_std"].append(np.std(raw_pool_f))
            results["o_std"].append(np.std(raw_pool_o))
            results["conn"].append(connectivity_list[i % num_connectivity_steps])
            results["noise"].append((i // num_connectivity_steps) + 1)

    return results

if __name__ == "__main__":
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    colors = ['red', 'blue', 'green', 'purple']
    markers = ['o', 's', 'x', 'D']
    final_data = {}

    for (label, path), color, marker in zip(FOLDER_CONFIG.items(), colors, markers):
        data = process_folder_with_large_std(path, NUM_RUNS_PER_SCENARIO, TOTAL_SCENARIOS, 
                                            NUM_CONNECTIVITY_STEPS, CONNECTIVITY_LIST, 
                                            N_AGENT, N_STATE, DIST, START_STEP)
        if data["f_mean"]:
            ax.scatter(data["f_mean"], data["o_mean"], label=label, color=color, marker=marker, s=80, alpha=0.7)
            for fm, om, c in zip(data["f_mean"], data["o_mean"], data["conn"]):
                ax.text(fm, om, f' {c}', fontsize=9)
            final_data[label] = data

    ax.set_xlabel("Formation Error (RMSE)", fontsize=12)
    ax.set_ylabel("Orientation Error (RMSE)", fontsize=12)
    ax.set_title("Scenario Error Comparison (STD reflects Instability)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("scenario_comparison_large_std.png")
    
    # --- 수치 분석 출력 ---
    base_lbl = list(FOLDER_CONFIG.keys())[0]
    
    def get_map(d): return {(c, n): (fm, fs, om, os) for fm, fs, om, os, c, n in zip(d["f_mean"], d["f_std"], d["o_mean"], d["o_std"], d["conn"], d["noise"])}
    all_maps = {lbl: get_map(final_data[lbl]) for lbl in final_data}

    for conn in sorted(CONNECTIVITY_LIST):
        key = (conn, 1) # Noise Step 1 기준
        if key not in all_maps.get(base_lbl, {}): continue
        
        print(f"\n[연결 {conn}개 시나리오]")
        b_fm, b_fs, b_om, b_os = all_maps[base_lbl][key]
        print(f" * {base_lbl:25s}: Form {b_fm:.3f}±{b_fs:.3f} | Orient {b_om:.3f}±{b_os:.3f}")

        for other in list(FOLDER_CONFIG.keys())[1:]:
            if key in all_maps.get(other, {}):
                o_fm, o_fs, o_om, o_os = all_maps[other][key]
                f_win = "✅" if b_fm < o_fm else "❌"
                print(f" vs {other:25s}: Form {o_fm:.3f}±{o_fs:.3f} {f_win} | Orient {o_om:.3f}±{o_os:.3f}")