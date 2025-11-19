# VERSA Installation Guide

This guide provides step-by-step instructions for installing VERSA on a new system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- CUDA (optional, for GPU support)

## Installation Steps

### 1. Install PyTorch (REQUIRED FIRST STEP)

VERSA requires PyTorch >= 2.6.0 for security compliance (CVE-2025-32434). You must install PyTorch **before** installing VERSA.

**Choose the appropriate command based on your system:**

#### For CUDA 12.4 (Recommended)
```bash
pip install torch>=2.6.0 torchaudio>=2.6.0 torchvision>=2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

#### For CUDA 11.8
```bash
pip install torch>=2.6.0 torchaudio>=2.6.0 torchvision>=2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU-only (no GPU)
```bash
pip install torch>=2.6.0 torchaudio>=2.6.0 torchvision>=2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Important:** Make sure `torch`, `torchaudio`, and `torchvision` all have matching CUDA versions. Mismatched versions will cause runtime errors.

### 2. Clone and Install VERSA

```bash
# Clone the repository
git clone https://github.com/mmarkaki/versa.git
cd versa

# Install VERSA
pip install .
```

### 3. Verify Installation

Test that VERSA is installed correctly:

```bash
# Test core functionality
python test/test_pipeline/test_general.py
```

If you see any errors about missing dependencies, see the "Troubleshooting" section below.

## Optional: Install Additional Metrics

Some metrics require additional dependencies. If you need specific metrics, run the installation scripts in the `tools/` directory:

```bash
cd tools
# Install all optional metrics (may take some time)
bash easy_install.sh

# OR install specific metrics individually:
bash install_utmosv2.sh
bash install_scoreq.sh
# ... etc
```

## Troubleshooting

### Error: "torch version X.X.X is too old"
- **Solution:** Upgrade PyTorch to version 2.6.0 or higher (see Step 1 above)

### Error: "PyTorch and TorchAudio were compiled with different CUDA versions"
- **Solution:** Uninstall and reinstall PyTorch, TorchAudio, and TorchVision with matching CUDA versions:
  ```bash
  pip uninstall torch torchaudio torchvision
  pip install torch>=2.6.0 torchaudio>=2.6.0 torchvision>=2.6.0 --index-url https://download.pytorch.org/whl/cu124
  ```

### Error: "module 'torchaudio' has no attribute 'set_audio_backend'"
- **Solution:** This should be automatically handled by VERSA's compatibility workaround. If you still see this error, make sure you're using the latest version from the repository.

### Error: "Could not load this library: libtorchaudio.so: undefined symbol"
- **Solution:** This usually indicates a CUDA version mismatch. Reinstall PyTorch, TorchAudio, and TorchVision with matching versions (see above).

### Some metrics are skipped during import
- **Note:** This is normal! VERSA will gracefully skip metrics that have missing dependencies. Check the log messages for which metrics were skipped and why. Install the missing dependencies if you need those specific metrics.

## Quick Reference

**Minimum Requirements:**
- Python >= 3.8
- PyTorch >= 2.6.0
- Matching torchaudio and torchvision versions

**Installation Command Summary:**
```bash
# 1. Install PyTorch
pip install torch>=2.6.0 torchaudio>=2.6.0 torchvision>=2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install VERSA
git clone https://github.com/mmarkaki/versa.git
cd versa
pip install .

# 3. Test
python test/test_pipeline/test_general.py
```

## Need Help?

If you encounter issues not covered here:
1. Check the main [README.md](README.md) for more details
2. Review error messages carefully - they often contain helpful hints
3. Ensure all PyTorch-related packages have matching versions
4. Try installing in a fresh virtual environment to avoid dependency conflicts

