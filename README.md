# Purpose of the project

Understanding faults, especially those with high impact, is a key step toward designing efficient fault-tolerant distributed algorithms. In this paper, we present an analysis of consistency violation faults (cvf) encountered in practical large-scale distributed computations. Our analysis highlights the presence of some cvfs that are capable of infinitely delaying the convergence of two self-stabilizing case study programs (graph coloring and maximal matching). An interesting observation from our analysis is that cvfs are quite possible in graph coloring but rare in maximal matching. Furthermore, with inexpensive enhancement, maximal matching can provably converge despite the presence of frequent cvfs. Our results can provide further insight into the interaction of cvfs in the self-stabilizing program and help design efficient algorithms that are resilient to them.


# Folder Structure


1. `cvf-analysis/`

    Base classes for `Full Analysis` code. Full Analysis logic explores all possible state space in the given distributed program. Based on the exploration ranks and rank effects of the states are calculated.
    
    
    
1. `simulations/`

    All simulations logic and code.

    
1. `utils/`

    Helper functions.

# Algorith

# Running the program
