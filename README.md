# DSE-04-Interstellar-Exploration
Repo for the DSE project

The current plan is as follows:

generate a series of ISOs, representative of the 10 year period

for each ISO, calculate the optimal trajectory, optimising for min dV, or min total dV
use this to generate a probability distribution of required dv for either just flyby, or rendezvouz
this should give a dv needed for the 90% criteria

(start without gravity assists, perhaps add that to the optimizer later)

Perhaps change from Nelder-mead to iterated local search, see: [this KSP KOS code](https://github.com/maneatingape/rsvp/tree/main/src)


TODO:
- [ ] read Dorsey to get a detection model
- [ ] incorporate this for FINAL iso generation and detection
- [ ] doublecheck FINAL dv values for icpt & rdvz
- [ ] incoporate Jupiter model
- [ ] charcterize optimal trajectories (time, range, etc.) (make loads of graphs from panda)
- [ ] write about it
- more?





- [ ] make proper dockstrings with input output? 
