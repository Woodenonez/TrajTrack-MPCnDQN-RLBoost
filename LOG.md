# Create (Dec. 9 2022) [TrajGen_MPC_FlexAMR]
To build a flexiable AMR high-level (v/w level) controller...

# 20221209
Data [Hadi]
# 20220724
Dataset (bookstore), with time offset channel; 
Training trial 1 [Start].
# 20220725
Training trial 1 [Done]. Prepare new data for real training.
# 20220803
Training trial 2 [Start]. New training data (20 start points, 10 trajectories each)

# 20220810
Dataset (warehouse), with time offset channel; 
Training trial 1 [Start].

# 20220816
Define data-type class: Node, Path, Trajectory, NetGraph in "datatype.py"
For graphs: networkx.Graph ⊂ NetGraph ⊂ SceneGraph

# 20220920
Construct Trajectory generation (MPC) + Dynamic obstacle motion prediction (DL) module/code structure 
(Current V1 - 2022 Init)

# 20220927
Construct interface - "doc-interface.pdf"

# 20221027
Merge into ROS!