import json
import glob
import os
import numpy as np
import argparse

def angular_error_simple(p_long, p_lat, t_long, t_lat):
    """Simple angular error calculation for diagnostic purposes."""
    p_az = np.radians(p_long)
    p_el = np.radians(p_lat)
    t_az = np.radians(t_long)
    t_el = np.radians(t_lat)
    
    cos_d = (np.sin(p_el) * np.sin(t_el) + 
             np.cos(p_el) * np.cos(t_el) * np.cos(p_az - t_az))
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.degrees(np.arccos(cos_d))

def analyze_dataset(name, pattern, max_files=50):
    files = glob.glob(pattern)
    if not files:
        print(f"[{name}] No files found matching: {pattern}")
        return None

    step0_errors = []
    long_biases = []
    lat_bias = []
    linearity_scores = [] # Variance of error velocity (lower = more linear)

    print(f"Analyzing {name} (sampling {min(len(files), max_files)} files)...")
    
    for f in files[:max_files]:
        with open(f) as j:
            data = json.load(j)
        
        # 1. Step 0 Error (Gaze vs HMD at start of window)
        r0 = data[0]
        err0 = angular_error_simple(r0['horz_hmd_deg'], r0['vert_hmd_deg'], r0['giw_long'], r0['giw_lat'])
        step0_errors.append(err0)

        # 2. Per-frame analysis for bias and linearity
        file_errors = []
        for r in data:
            long_biases.append(r['giw_long'] - r['horz_hmd_deg'])
            lat_bias.append(r['giw_lat'] - r['vert_hmd_deg'])
            
            # Error relative to static HMD(0) baseline
            e = angular_error_simple(r0['horz_hmd_deg'], r0['vert_hmd_deg'], r['giw_long'], r['giw_lat'])
            file_errors.append(e)
        
        # 3. Linearity check (variance of error velocity)
        if len(file_errors) > 2:
            velocity = np.diff(file_errors)
            linearity_scores.append(np.var(velocity))

    results = {
        "Avg Step-0 Error": np.mean(step0_errors),
        "Mean Long Bias": np.mean(long_biases),
        "Mean Lat Bias": np.mean(lat_bias),
        "Linearity Metric (Lower is more linear)": np.mean(linearity_scores)
    }
    
    print(f"\n--- Results for {name} ---")
    for k, v in results.items():
        print(f"{k:<40}: {v:.4f}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eht_path", default="../processed_dataset/eht_dataset_prossesed_vid/Video_*/User_*_Task_*/*.json")
    parser.add_argument("--dsav_path", default="../processed_dataset/dsav360_processed_vid/Video_*/User_*_Task_1/*.json")
    args = parser.parse_args()

    eht = analyze_dataset("EHTask", args.eht_path)
    dsav = analyze_dataset("DSAV360", args.dsav_path)

    if eht and dsav:
        print("\n" + "="*60)
        print("DIAGNOSTIC CONCLUSION")
        print("="*60)
        diff_bias = abs(dsav["Mean Long Bias"]) / abs(eht["Mean Long Bias"])
        print(f"1. Systematic Bias: DSAV360 has {diff_bias:.1f}x higher longitude shift than EHT.")
        
        offset_diff = dsav["Avg Step-0 Error"] - eht["Avg Step-0 Error"]
        print(f"2. Initial Offset: DSAV360 starts with {offset_diff:.2f} deg higher error than EHT.")
        
        lin_ratio = eht["Linearity Metric (Lower is more linear)"] / dsav["Linearity Metric (Lower is more linear)"]
        print(f"3. Path Geometry: DSAV360 error growth is {lin_ratio:.1f}x more linear than EHT.")
        print("   (Linearity is caused by sparse sampling/interpolation in DSAV360.)")
        print("="*60)
