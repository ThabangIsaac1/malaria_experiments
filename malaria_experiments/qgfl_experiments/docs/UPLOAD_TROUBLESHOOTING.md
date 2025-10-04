# Upload Troubleshooting

## Issue: Connection Timed Out

**Error:** `ssh: connect to host 147.252.6.20 port 22: Operation timed out`

---

## Solutions

### 1. Check VPN Connection

**Is your VPN still connected?**

```bash
# Test connection
ping 147.252.6.20

# If no response, reconnect VPN and try again
```

---

### 2. Test SSH Connection First

Before uploading, verify you can SSH:

```bash
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.20
```

If this works, you're connected. Type `exit` to return to local.

---

### 3. Alternative: Upload in Chunks

If full upload times out, upload in smaller pieces:

**A. Upload code only (small, fast):**
```bash
# Upload Python files
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/qgfl_experiments/cluster_run_baseline.py" \
  d23125116@147.252.6.20:~/

# Upload configs
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 -r \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/qgfl_experiments/configs" \
  d23125116@147.252.6.20:~/

# Upload src
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 -r \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/qgfl_experiments/src" \
  d23125116@147.252.6.20:~/
```

**B. Upload datasets separately (or check if already there):**

**Important Question:** Are the datasets (D1, D2, D3) already on the cluster from previous work?

If YES: Skip dataset upload, just upload code!

If NO: Compress and upload:
```bash
# Compress datasets first (faster upload)
cd "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments"
tar -czf datasets.tar.gz dataset_d1 dataset_d2 dataset_d3

# Upload compressed file
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 datasets.tar.gz d23125116@147.252.6.20:~/

# SSH to cluster and extract
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.20
cd ~
tar -xzf datasets.tar.gz
rm datasets.tar.gz  # Clean up
```

---

### 4. Work Directly on Cluster (Fastest Option)

Since you were already SSH'd in earlier, you could:

1. **SSH back in:**
   ```bash
   ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.20
   ```

2. **Create project structure on cluster:**
   ```bash
   mkdir -p ~/malaria_qgfl_experiments/{configs/data_yamls,src/{evaluation,training,utils},cluster_scripts,logs}
   ```

3. **Download models directly on cluster:**
   ```bash
   cd ~/malaria_qgfl_experiments
   wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
   ```

4. **Use scp to upload small files only** (from local):
   - cluster_run_baseline.py
   - configs/
   - src/

---

## Recommended Quick Path

**Step 1:** Check VPN and test SSH
```bash
# Test connection
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.20

# If works, exit
exit
```

**Step 2:** Upload code only (skip datasets if already there)
```bash
# Create tar of code only
cd "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/qgfl_experiments"
tar -czf qgfl_code.tar.gz cluster_run_baseline.py configs src

# Upload
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 qgfl_code.tar.gz d23125116@147.252.6.20:~/

# SSH and extract
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.20
cd ~
mkdir -p malaria_qgfl_experiments
tar -xzf qgfl_code.tar.gz -C malaria_qgfl_experiments/
rm qgfl_code.tar.gz
```

**Step 3:** Check if datasets exist on cluster or upload separately

---

## Next Steps After Upload

Continue with CLUSTER_DEPLOYMENT_STEPS.md from Step 3:
- Setup phd_env
- Create SLURM scripts
- Submit jobs

---

**What to do now?**

1. **Check VPN connection** - Reconnect if needed
2. **Try SSH test** - Verify connection works
3. **Choose upload method:**
   - Full upload (if VPN stable)
   - Chunked upload (if timeout issues)
   - Code-only upload (if datasets already on cluster)
