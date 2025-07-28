# Purpose of the project

Understanding faults, especially those with high impact, is a key step toward designing efficient fault-tolerant distributed algorithms. In this paper, we present an analysis of consistency violation faults (cvf) encountered in practical large-scale distributed computations. Our analysis highlights the presence of some cvfs that are capable of infinitely delaying the convergence of two self-stabilizing case study programs (graph coloring and maximal matching). An interesting observation from our analysis is that cvfs are quite possible in graph coloring but rare in maximal matching. Furthermore, with inexpensive enhancement, maximal matching can provably converge despite the presence of frequent cvfs. Our results can provide further insight into the interaction of cvfs in the self-stabilizing program and help design efficient algorithms that are resilient to them.


# Folder Structure


1. `cvf-analysis/`

    Base classes for `Full Analysis` code. Full Analysis logic explores all possible state space in the given distributed program. Based on the exploration ranks and rank effects of the states are calculated.

   1.1 `cvf-analysis/graphs`
   
      Graphs, in the form of text file, to run the analysis on.
    
    
    
1. `simulations/`

    All simulations logic and code. Simulations are performed `N` (thousands) times to generate a statistics of rank effect based on the number of steps taken by the random initial state to reach an state where no possible transition or fault (cvf) exist. Faults are introduced every `Fault Interval` steps i.e. if `Fault Interval = 5` then first `4` state transitions are Program Transitions and the `5th` step is a fault from cvf at targetted edges.
    
1. `utils/`

    Helper functions.


# Running the program
  
  1. Running the Simulation

     Eg.1. 
