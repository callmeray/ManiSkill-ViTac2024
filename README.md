
# [ManiSkill-Vitac Challenge 2024](https://ai-workshops.github.io/maniskill-vitac-challenge-2024/)

**Table of Contents**

- [Installation](#installation)
- [Training Example](#example)
- [Submission](#submission)
- [Leaderboard](#leaderboard)
- [Contact](#contact)
- [Citation](#citation)

## Installation

**Requirements:**

- Python 3.7.x-3.11.x
- Microsoft Visual Studio 2019 upwards (Windows)
- GCC 7.2 upwards (Linux)
- CUDA Toolkit 11.5 or higher
- Git LFS installed (https://git-lfs.github.com/)


Clone this repo with

```bash
git clone https://github.com/callmeray/ManiSkill-ViTac2024.git
```

Run

```bash
conda env create -f environment.yaml
conda activate mani_vitac
```

Then use the following commands to install [sapienIPC](https://github.com/Rabbit-Hu/sapienipc-exp), following the [README](https://github.com/Rabbit-Hu/sapienipc-exp/blob/main/README.md) file in that repo.

## Training Example

To train our example policy, run

```bash
# example policy for peg insertion
python scripts/universal_training_script.py --cfg configs/parameters/peg_insertion.yaml
# example policy for open lock
python scripts/universal_training_script.py --cfg configs/parameters/long_open_lock.yaml
```

## Submission 
For policy evaluation in simulation, run

```bash
# evaluation of peg insertion and lock opening
# replace the key and the policy model
python scripts/peg_insertion_sim_evaluation.py
python scripts/open_lock_sim_evaluation.py
```
Submit the evaluation logs by emailing them to [maniskill.vitac@gmail.com](maniskill.vitac@gmail.com)

## Leaderboard

The leaderboard for this challenge is available at [Google Drive](https://docs.google.com/spreadsheets/d/1ZCNSbctm5eyr4Q59KmVBE0ZMo5mt63emFLihbJn1maw/).

## Contact

Join our [discord](https://discord.gg/B8qEVTav) to contact us. You may also email us at [maniskill.vitac@gmail.com](maniskill.vitac@gmail.com)


## Citation

```
@ARTICLE{chen2024tactilesim2real,
              author={Chen, Weihang and Xu, Jing and Xiang, Fanbo and Yuan, Xiaodi and Su, Hao and Chen, Rui},
              journal={IEEE Transactions on Robotics}, 
              title={General-Purpose Sim2Real Protocol for Learning Contact-Rich Manipulation With Marker-Based Visuotactile Sensors}, 
              year={2024},
              volume={},
              number={},
              pages={1-18},
              doi={10.1109/TRO.2024.3352969}}
```

