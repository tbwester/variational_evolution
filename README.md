# Variational Evolution

A genetic algorithm for finding variational Anzatz solutions that approximately solve quantum mechanics Hamiltonians.

### Usage

To run:

```
python variational_evolution.py
```

The program defaults to run for 10 generations, with 200 trial functions per generation. The target Hamiltonian is the symmetric double well potential.

Written for Python2.7. Untested for Python3, but it will probably work with minimal adjustments.

Requires `numpy`, `scipy`, `matplotlib`.

### Function Parameters:
 
- `COMPONENTS` List of possible functions that can be included in an organism's trial function. Capital `I`s are replaced by integers from [-6,6]. Empty parentheses `()` are replaced by polynomials up to degree 4.
- `OPERATORS` List of possible operators to join terms of function. By default, only multiplaction and division since the double-well potential has symmetric solutions.
- `MAX_TERMS` Number of terms from `COMPONENT` list that will be chosen. By default, the first term generates a symmetric and anti-symmetric component which is then multiplied or divided by a second term. Untested for values other than 1 or 2.
- `H_DIM` Dimensions of Hamiltonian matrix to compute exact solutions numerically.
- `M`, `W`, `L` are constants of the Hamiltonian.



### Genetic algorithm parameters:
 
- `MAX_GENERATIONS` controls the number of cycles the program will run before ending.
- `POPULATION_SIZE` controls how many functions are tested and compared per cycle.
- `MUTATION_RATE` controls random modification chance of new trial function at the start of each cycle.
- `MIGRATION_SIZE` controls the number of reserved spaces for brand new functions at the start of each cycle.
