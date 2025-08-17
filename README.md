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
- sorting simulation ships on their target index: done
- segmented reduction or histogram of total damage from source simulation ships to target simulation ships (thrust reduce_by_key): done
- calculate remaining hull points (minimum 0) from the calculated damage: to do
- stream-compaction of simulation ships with (hull > 0) condition: to do (thrust's copy_if)
- updating new number of simulation ships on both teams: to do
- repeat if each side has at least 1 ship: to do
- add simulation queue (1 gpu per consumer) with n number of consumers: to do
