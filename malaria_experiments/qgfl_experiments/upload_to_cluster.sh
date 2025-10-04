#!/bin/bash
# Upload entire malaria_experiments folder to cluster
# Handles old SSH key exchange methods

echo "========================================="
echo "Uploading malaria_experiments to Cluster"
echo "========================================="

# Use rsync with SSH legacy options and exclude unnecessary files
rsync -avz --progress \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='env/' \
  --exclude='phd_env/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='runs/' \
  -e "ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa" \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/" \
  d23125116@147.252.6.50:~/malaria_qgfl_experiments/

echo "========================================="
echo "Upload complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. SSH into cluster: ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.50"
echo "2. Verify upload: ls -lh ~/malaria_qgfl_experiments/"
echo "3. Continue with Step 3 in CLUSTER_DEPLOYMENT_STEPS.md"
