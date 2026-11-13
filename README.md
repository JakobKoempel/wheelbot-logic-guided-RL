# Wheelbot MJX
This repository contains an implementation of the Mini Wheelbot simulation using [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html), [JAX](https://docs.jax.dev/en/latest/quickstart.html), and [BRAX](https://github.com/google/brax). On top of the simulation it provides three training pipelines, all based on [SAC](https://arxiv.org/abs/1812.05905):

| Pipeline | What is learned | Script |
|---|---|---|
| Classical | A single-setpoint driving policy without any LTL conditioning | `classical_sac_train` |
| Setpoint (torque level) | A DeepLTL-style policy that reaches setpoints, conditioned on a reach-avoid sequence | `setpoint_sac_train` |
| Zone env (planning level) | A DeepLTL-style policy that chooses setpoints in the zone environment and drives the frozen torque-level setpoint policy | `zone_env_sac_train` |

LTL formulas are translated to automata with the [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html) tool (`ltl2ldba`), which is shipped in `rabinizer4/`.

## Installation

### 1. Clone the repository
```bash
git clone git@git.rwth-aachen.de:dsme-projects/wheelbot/wheelbot-mjx.git
cd wheelbot-mjx
```

### 2. Python environment (uv)
Install [UV](https://docs.astral.sh/uv/getting-started/installation/), which is our preferred Python package and project manager. Python **3.10** is required.
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements/requirements.txt
uv pip install -e .
```
`requirements.txt` pins JAX `0.6.0` with the CUDA 12 plugin (`jax-cuda12-*`), so an NVIDIA GPU with a CUDA 12 compatible driver is expected for training.

### 3. Graphviz
Graphviz is used to draw the automata (`graphviz` Python package, which needs the system binaries `dot`):
```bash
# Debian / Ubuntu
sudo apt install graphviz
# macOS
brew install graphviz
# check
dot -V
```
On a cluster without root rights, install it via conda (`conda install -c conda-forge graphviz`) or load the module provided by your cluster and make sure `dot` is on your `PATH`.

### 4. Java 11 (for Rabinizer)
Rabinizer (`rabinizer4/bin/ltl2ldba`, which runs `rabinizer4/lib/owl.jar`) needs **Java 11**. Install a JDK 11 and make sure that it is the `java` that is found:
```bash
# Debian / Ubuntu
sudo apt install openjdk-11-jre
# check, should report version 11.x
java -version
```
If several Java versions are installed, point `JAVA_HOME` and `PATH` to Java 11:
```bash
export JAVA_HOME=/path/to/jdk-11
export PATH=$JAVA_HOME/bin:$PATH
```

### 5. Rabinizer path
The Rabinizer path is a **relative** path, `rabinizer4/bin/ltl2ldba` (see [rabinizer.py](wheelbot_sim_python/automata/ltl/automata/rabinizer.py)). Therefore:
- **Always start all scripts from the repository root** (the folder that contains `rabinizer4/`), using `python -m wheelbot_sim_python.<script>`.
- Make sure the start script is executable:
  ```bash
  chmod +x rabinizer4/bin/ltl2ldba
  ```
- Any stderr output of Rabinizer (except the harmless `Picked up JAVA_TOOL_OPTIONS: -Xmx2048m` message) makes the Python call raise an error. If you see Java warnings, check that `java -version` reports 11 and that `JAVA_TOOL_OPTIONS` is unset or `-Xmx2048m`.

## Configuration
[Hydra](https://hydra.cc/docs/intro/) is used for config management. Configs are in [wheelbot_sim_python/config/](wheelbot_sim_python/config). Any value can be overridden on the command line (`key=value`), and values that are not in the config are added with `+key=value`.

| Config | Used by | Content |
|---|---|---|
| `classical_main.yaml` | classical training | `algorithm: sac_classical`, `env: wheelbot_classical`, experiment name `classical` |
| `main.yaml` | setpoint + zone env training, all evaluations / visualizations | `algorithm` (setpoint SAC), `high_level_algorithm` (zone env SAC), `env: wheelbot`, `deep_ltl` network, `seed`, `experiment`, `wandb_project` |

Every run writes its outputs (Hydra config, checkpoints) to `exp/<experiment>/<date>/<time>/`. Change the run name with `experiment=<name>`.

## Training
Always run from the repository root, with the virtual environment activated.

### 1. Classical setpoint policy
Trains a plain SAC policy that drives to a single setpoint (no LTL).
```bash
python -m wheelbot_sim_python.classical_sac_train
```
Settings: [classical_main.yaml](wheelbot_sim_python/config/classical_main.yaml), [algorithm/sac_classical.yaml](wheelbot_sim_python/config/algorithm/sac_classical.yaml), [env/wheelbot_classical.yaml](wheelbot_sim_python/config/env/wheelbot_classical.yaml).
The final model is saved as `exp/classical/<date>/<time>/final_mjx_brax_policy`, the best checkpoint as `best_policy` in the same folder.

### 2. DeepLTL setpoint policy (torque level)
Trains the DeepLTL SAC policy on reach-avoid sequences over setpoints.
```bash
python -m wheelbot_sim_python.setpoint_sac_train
```
Settings: [main.yaml](wheelbot_sim_python/config/main.yaml), [algorithm/sac.yaml](wheelbot_sim_python/config/algorithm/sac.yaml), [env/wheelbot.yaml](wheelbot_sim_python/config/env/wheelbot.yaml), [deep_ltl/deep_ltl_net.yaml](wheelbot_sim_python/config/deep_ltl/deep_ltl_net.yaml).
The final model is saved as `exp/<experiment>/<date>/<time>/final_policy`, the best checkpoint as `best_policy` in the same folder.

### 3. DeepLTL zone environment (planning level)
Trains the planning-level policy on top of a **trained, frozen setpoint policy**, so run the setpoint training first and pass its checkpoint as `low_level_agent_dir`:
```bash
python -m wheelbot_sim_python.zone_env_sac_train +low_level_agent_dir=saved_agents/consecutive_setpoints
```
`low_level_agent_dir` is mandatory, there is no default. Settings for the high-level agent are in [high_level_algorithm/sac.yaml](wheelbot_sim_python/config/high_level_algorithm/sac.yaml) (`algorithm/sac.yaml` only describes the torque level). The final model is saved as `exp/<experiment>/<date>/<time>/final_policy`, the best checkpoint as `best_policy`.

## Pretrained agents
[saved_agents/](saved_agents) contains trained agents that can be used for evaluation and visualization:

| Folder | Agent |
|---|---|
| `saved_agents/classical` | classical policy |
| `saved_agents/roll_angles/r_03 ... r_09` | setpoint policies trained with different roll thresholds |
| `saved_agents/consecutive_setpoints` | setpoint policy used as the torque level for several consecutive setpoints and the zone env |
| `saved_agents/zone_env/best_policy` | planning-level zone env policy |