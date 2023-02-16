# Title
...

## Publication
The paper is available: [Not Yet] \
Bibtex citation:
```
not yet
```

![Example](doc/cover.png "Example")

## Quick Start
### OpEn
The NMPC formulation is solved using open source implementation of PANOC, namely [OpEn](https://alphaville.github.io/optimization-engine/). Follow the [installation instructions](https://alphaville.github.io/optimization-engine/docs/installation) before proceeding. 

### Install dependencies
```
pip install -r requirements.txt
```

### Generate MPC solver
Go to "test_block.py", change **INIT_BUILD** to true and run
```
python test_block_mpc.py
```
After this, a new directory *mpc_build* will appear and contains the solver. Then, you are good to go :)

## Use Case
Run *main.py* for the warehouse simulation (one robot, two pedestrians) in Python. Several cases are available in *test_block_mpc.py*. Motion prediction test is in *test_block_mmp.py*.

## ROS Simulation
[ROS XXX](https://github.com/) [Not Yet]



