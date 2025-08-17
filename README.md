# CudaAcceleratedOgameBattleSimulator
To do: thousands of simulations per second for billions of spaceships on multiple GPUs.

In development.

Dependencies:

- CUDA runtime
- curand
- (probably thrust or cuda-unbounded too)
- At least 1 Nvidia GPU

----

Development:

- initialization of random seeds for ships: done
- initialization of hull (hitpoints) for ships: done
- initialization of ship types: done
- copying ship data to simulation ship data (that will be updated many times): done
- randomly picking a target per simulation ship: done
- sorting simulation ships on their target index: to do (this can be done with thrust or cub)
- segmented reduction or histogram of total damage from source simulation ships to target simulation ships and calculating remaining hull: to do (cub or thrust)
- stream-compaction of simulation ships with (hull > 0) condition: to do (thrust's copy_if)
- updating new number of simulation ships on both teams: to do
- repeat if each side has at least 1 ship: to do
