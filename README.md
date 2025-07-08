# TunEOS
<p align="center">
  <img src="logo.png" width="500" alt="TunEOS logo">
</p>

**TunEOS** is a hyperparameter optimization and automation toolkit for modern equivariant graph neural network models in computational materials science. It streamlines the process of tuning and training three state-of-the-art models—**NequIP**, **MACE**, and **Allegro**—by integrating:

* **Optuna** for efficient hyperparameter search.
* **Slurm** for scalable job scheduling.
* **OmegaConf** & **Hydra** for flexible configuration management.
* **Weights & Biases** (W\&B) for experiment logging and tracking.

## 📂 Repository Structure

```
TunEOS/
├── opt_Allegro/        # Allegro model configs & Slurm template
│   ├── config.yaml     # Default Allegro config with Tuneos(...) entries
│   └── job.sh          # Slurm submission script template
├── opt_Mace/           # MACE model configs & Slurm template
│   ├── config.yaml     # Default MACE config with Tuneos(...) entries
│   └── job.sh          # Slurm submission script template
├── opt_Nequip/         # NequIP model configs & Slurm template
│   ├── config.yaml     # Default NequIP config with Tuneos(...) entries
│   └── job.sh          # Slurm submission script template
├── preprocess_conf.py  # Parse configs to extract tunable parameters
├── tuning.py           # Launch Optuna studies and submit Slurm jobs
├── requirements.txt    # Python dependency list
```

## 🔧 Prerequisites

* **Python** ≥ 3.9
* **Slurm** workload manager installed and configured
* **Weights & Biases** account & CLI (`wandb`)
* **PostgreSQL** server for Optuna storage (update connection string in `tuning.py`)
* CUDA-enabled GPU setup (for `torch`, `mace-torch`, `nequip`, etc.)

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kahassanzadeh/TunEOS.git
   cd TunEOS
   ```
2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Log into W\&B:

   ```bash
   wandb login
   ```

## ⚙️ Configuration

1. In each `opt_<Model>/config.yaml`, set the:

   * **data paths** (e.g., `split_dataset.file_path`, `train`, `val`, `test`).
   * **model parameters** (e.g., `num_channels`, `l_max`)—use `Tuneos(...)` to mark tunable fields.
   * **training settings** (e.g., `batch_size`, `max_epochs`, `trainer`).
2. Customize `job.sh` for your Slurm environment (modules, paths, partitions).
3. In `tuning.py`, update the `storage` URL to point to your PostgreSQL instance.

## 🔍 Discover Tunable Parameters

Run:

```bash
python preprocess_conf.py
```

This will print all parameters marked with `Tuneos(...)` in each `opt_<Model>/config.yaml` and their candidate values.

## 🎯 Running Hyperparameter Tuning

By default, `tuning.py` runs the MACE study. To launch tuning for your chosen model:

```bash
# For MACE (default study name: final_mace_test)
python tuning.py

# To change model or study name, modify the last lines of tuning.py:
#     if __name__ == "__main__":
#         tuning('Nequip', study_name='my_nequip_study')
```

The script will:

1. Parse tunable parameters.
2. For each Optuna trial:

   * Update Slurm script and config.
   * Submit a job via `sbatch`.
   * Monitor the W\&B run URL from Slurm logs.
   * Fetch the final metric (`test0_epoch/weighted_sum`).
3. Record results in the Optuna study (PostgreSQL).

## 📝 Notes & Best Practices

* **Slurm**: Ensure `sacct`, `scancel`, and `sbatch` are in your `PATH`.
* **W\&B**: Set `entity` and `project` fields in your config for proper logging.
* **Resource Cleanup**: Remove stale `opt_<Model>/*` directories between runs if needed.

## 🤝 Contributing

1. Fork the repo and create a new branch.
2. Make your changes (e.g., add new model support, improve CLI).
3. Submit a pull request with a clear description and tests.

*Happy tuning!*
