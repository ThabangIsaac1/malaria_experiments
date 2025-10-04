# Alternate Upload Method - Already SSH'd into Cluster

**Issue:** SSH key exchange method outdated - rsync from local fails
**Solution:** Upload using SCP with legacy options OR work directly on cluster

---

## Option 1: Fix SSH Key Exchange (From Local)

Add this to your local `~/.ssh/config`:

```bash
cat >> ~/.ssh/config << 'EOF'

Host adapt-cluster
    HostName 147.252.6.20
    User d23125116
    KexAlgorithms +diffie-hellman-group14-sha1,diffie-hellman-group1-sha1
    HostKeyAlgorithms +ssh-rsa,ssh-dss
    PubkeyAcceptedKeyTypes +ssh-rsa
EOF
```

Then upload:
```bash
rsync -avz --progress \
  -e "ssh -o KexAlgorithms=+diffie-hellman-group14-sha1" \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/" \
  d23125116@147.252.6.20:~/malaria_qgfl_experiments/
```

---

## Option 2: Upload via SCP with Legacy Options (From Local)

```bash
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 -r \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/" \
  d23125116@147.252.6.20:~/malaria_qgfl_experiments/
```

---

## Option 3: Git Clone (If Project is on GitHub)

**On cluster:**
```bash
cd ~
git clone YOUR_GITHUB_REPO_URL malaria_qgfl_experiments
```

---

## Option 4: Download Directly on Cluster (Recommended since you're already SSH'd)

Since you're already on the cluster, let's create everything there directly!

### Step 1: Create project structure on cluster
```bash
cd ~
mkdir -p malaria_qgfl_experiments/{configs/data_yamls,src/{evaluation,training,utils},cluster_scripts,logs,models}
```

### Step 2: Download model weights on cluster
```bash
cd ~/malaria_qgfl_experiments/models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

### Step 3: Upload only essential Python files

From your **local terminal** (in a new window, keep cluster SSH open):

**Upload just the Python code (small files):**
```bash
# Upload cluster_run_baseline.py
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/qgfl_experiments/cluster_run_baseline.py" \
  d23125116@147.252.6.20:~/malaria_qgfl_experiments/

# Upload configs
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 -r \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/qgfl_experiments/configs/" \
  d23125116@147.252.6.20:~/malaria_qgfl_experiments/

# Upload src
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 -r \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/qgfl_experiments/src/" \
  d23125116@147.252.6.20:~/malaria_qgfl_experiments/
```

### Step 4: Where are your datasets?

**Important question:** Are the datasets (D1, D2, D3) already on the cluster somewhere, or do they need to be uploaded?

If they need upload, compress first:
```bash
# On local machine
cd "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/"
tar -czf datasets.tar.gz dataset_d1 dataset_d2 dataset_d3

# Upload compressed (much faster)
scp -o KexAlgorithms=+diffie-hellman-group14-sha1 datasets.tar.gz d23125116@147.252.6.20:~/

# On cluster, extract
cd ~
tar -xzf datasets.tar.gz -C malaria_qgfl_experiments/
```

---

## Step 5: Continue with Deployment (On Cluster)

Once files are uploaded, continue from **Step 3** in CLUSTER_DEPLOYMENT_STEPS.md:
- Setup phd_env
- Create SLURM scripts
- Submit jobs

---

**Recommendation:** Use **Option 4** - work directly on cluster since you're already SSH'd in!
