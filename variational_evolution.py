import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
from multiprocessing import Pool
from numpy.linalg import norm

# special function import
from scipy.special import jv, airy, laguerre, erfc
from scipy.stats import t

# lots of numpy warnings imminent
np.warnings.filterwarnings('ignore')


# list of generic function components
# characters will be replaced, e.g. I->Integer
# additional components will be inserted between ()
COMPONENTS = [
        'np.exp()', '()',                                # generic
        'np.sin()**I', 'np.cos()**I',                    # trig
        'np.cosh()**I', 'np.sinh()**I',                  # hyperbolic trig
        'jv(I, ())**I', 'airy()[0]**I', 'airy()[2]**I',  # special functions
        't.pdf((), np.abs(I))', 'laguerre(np.abs(I))()', # special functions
        'erfc()'                                         # special functions, add your own!
        ]
OPERATORS = ['*', '/']

# evolution parameters
MAX_TERMS = 2          # max number of terms in generated functions. Modify at risk of bugs!
MUTATION_RATE = 0.05   # how frequent do offspring get randomly mutated
POPULATION_SIZE = 200
MIGRATION_SIZE = 25    # this many spaces in population are reserved for random organisms each cycle
MAX_GENERATIONS = 10   # how long to run

# Hamiltonian dimensions
H_DIM = 200

# Potential parameters
M = 1.0
W = 1.0
L = 4.0

def hamiltonian(l=4):
    ''' Generate a Hamiltonian in matrix form, return matrix, eigen values, and x range '''
    x_max = l + 5 * np.sqrt(1./(2 * M * W)) 
    x_min = - x_max

    x, dx = np.linspace(x_min, x_max, H_DIM, retstep=True)

    #v_x = 0.5 * M * W**2 * X**2 + 0.25 * epsilon * X**4   
    v_x = M * W**2 / (8 * l**2) * (x**2 - l**2)**2    

    # Solve Hamiltonian exactly in matrix basis
    v = scipy.sparse.diags(v_x)
    k = 1. / M * scipy.sparse.diags(np.ones(H_DIM), 0)
    k += -(0.5 / M) * scipy.sparse.diags(np.ones(H_DIM - 1), 1)
    k += -(0.5 / M) * scipy.sparse.diags(np.ones(H_DIM - 1), -1)
    # k[-1,0]=-(0.5/m); K[0,-1]=-(0.5/m);  # Uncomment this line for periodic boundary conditions. Ignore the warning.

    h = k / (dx**2) + v

    # Calculate the 2 lowest energy Eigenstates of the system
    eigees = scipy.sparse.linalg.eigsh(h, k=2, which='SA')

    return h, eigees, x

# we don't want to recompute the hamiltonian each time we evaluate a function
# so we'll compute a few first and sample these
HAMILTONIANS = [hamiltonian(x) for x in np.linspace(3, 5, 5)]


class Organism(object):
    ''' Organism class which holds data about trial functions during each cycle '''

    def __init__(self, func_comps=None, func_ops=None, guesses=None, guesses_asym=None):
        # age increments such that this organism will die if it gets too old
        # keeps population dynamic
        self._age = 0

        # initialize this organism's function
        self._function_components = func_comps
        self._function_operators = func_ops
        if func_comps is None or func_ops is None:
            self._function_components, self._function_operators = Organism.random_function()
        else:
            assert len(self._function_operators) == len(self._function_components) - 1

        self._guesses = guesses
        self._guesses_asym = guesses_asym

        self._function_string = ''
        self._function_string_asym = ''
        self._n_params = 0
        self.set_functions()

        # to be filled after computation
        self._fitness = None
        self._best_fit = None
        self._best_fit_asym = None
        self._energy_split = None
        self._fidelity = None
        self._split_error = None
        self._fidelity_error = None

    def set_functions(self):
        ''' Build a function from function component strings '''
        func_string = ''
        func_string_asym = ''
        for i, term in enumerate(self._function_components):
            if i == 0:
                func_string += '(' + term + ' + ' + term.replace('x*', '-x*') + ')'
                func_string_asym += '(' + term + ' - ' + term.replace('x*', '-x*') + ')'
            else:
                func_string += term
                func_string_asym += term
            if i != len(self._function_components) - 1:
                func_string += ' {} '.format(self._function_operators[i - 1])
                func_string_asym += ' {} '.format(self._function_operators[i - 1])

        self._function_string = func_string
        self._function_string_asym = func_string_asym

        # find number of params
        n_params = 0
        for i in range(10):
            if 'p[{}]'.format(str(i)) in func_string:
                n_params += 1

        self._n_params = n_params 


    @property
    def function_components(self):
        return self._function_components

    @property
    def function_operators(self):
        return self._function_operators

    @property
    def function_string(self):
        return self._function_string

    @property
    def function_string_asym(self):
        return self._function_string_asym

    @property
    def age(self):
        return self._age

    @property
    def fitness(self):
        return self._fitness

    @property
    def best_fit(self):
        return self._best_fit

    @property
    def best_fit_asym(self):
        return self._best_fit_asym

    @property
    def energy_split(self):
        return self._energy_split

    @property
    def split_error(self):
        return self._split_error
    
    @property
    def fidelity(self):
        return self._fidelity

    @property
    def fidelity_error(self):
        return self._fidelity_error


    def compute(self, hamiltonian):
        ''' Find best fit of trial function given Hamiltonian, exact energies and x range '''

        # always call this to ensure function strings get updated before computation
        self.set_functions()

        # getting older!
        self._age += 1

        # use user-supplied guesses if available, otherwise pick random ones
        guess = self._guesses
        guess_asym = self._guesses_asym

        if self._guesses is None:
            guess = np.random.normal(loc=0.5, scale=0.5, size=self._n_params)
        if self._guesses_asym is None:
            guess_asym = np.random.normal(loc=0.5, scale=0.5, size=self._n_params)

        h, energy_exact, xs = hamiltonian

        # evaluate wavefunction for given set of parameters
        psi_func = lambda x, p: eval(self._function_string) / norm(eval(self._function_string))
        psi_func_asym = lambda x, p: eval(self._function_string_asym) / norm(eval(self._function_string_asym))

        # function to minimize
        def energy(params, psi):
            psi_eval = psi(xs, params)
            energy = np.dot(np.conj(psi_eval), h.dot(psi_eval))
            return energy

        optimizer = scipy.optimize.minimize(energy, guess, args=(psi_func))
        optimizer_asym = scipy.optimize.minimize(energy, guess_asym, args=(psi_func_asym))
        self._best_fit = optimizer['x']
        self._best_fit_asym = optimizer_asym['x']

        energy = optimizer['fun']
        energy_asym = optimizer_asym['fun']

        self._energy_split = energy_asym - energy
        num_split = energy_exact[0][1] - energy_exact[0][0]

        self._split_error = np.abs(self._energy_split - num_split) / num_split

        self._fidelity = np.abs(
                np.conj(energy_exact[1][:,0]).dot(psi_func(xs, self._best_fit))
                )
        self._fidelity_error = 1.0 - self._fidelity

        # want both small energy splitting and good function agreement
        self._fitness = (1. / self._split_error) * (1. / self._fidelity_error)
        
        # minimum fidelity cut
        if self._fidelity < 0.9:
            self._fitness /= 100

        # minimum energy split cut
        if self._split_error > 0.5:
            self._fitness /= 100

        if np.isnan(self._fidelity_error) or np.isnan(self._split_error):
            self._fitness = 0.0


    @staticmethod
    def random_function(max_terms=MAX_TERMS):
        ''' Generates a random function from bank of components defined at the top '''
        function_components = []
        function_operators = []

        # adds polynomial term with arbitrary prefactors
        def polynomial(nterms=1):
            s = ''
            powers = [1]
            if nterms > 1:
                powers = np.random.choice(np.arange(nterms) + 1, np.random.randint(1, nterms), replace=False) 

            for p in powers:
                if s != '':
                    s += ' + '
                comp = '(x{}p[{}])**{}'.format(np.random.choice(['+','*']), np.random.randint(-1,2), p)
                if 'x-p' in comp or 'x+p' in comp:
                    comp = 'p[{}]*'.format(np.random.randint(-1, 2)) + comp 
                s += comp

            if 'p[1]' in s and 'p[0]' not in s:
                s = s.replace('p[1]', 'p[0]')

            # sometimes get a function with no parameters.
            # causes problems for optimizer, so this should
            # set 1 parameter minimum
            if 'p[1]' in s or 'p[0]' in s:
                s = s.replace('p[-1]','1')
            else:
                s = s.replace('p[-1]', 'p[0]')

            return s

        for i in range(max_terms):
            # random stopping condition
            if np.random.random() > 0.5 and len(function_components) > 0:
                break

            comp = np.random.choice(COMPONENTS)

            # replace placeholders
            comp = comp.replace('I', str(np.random.randint(-6,6)))
            if '()' in comp:
                comp = comp.replace('()', '({})'.format(polynomial(np.random.randint(1,5))))

            # no operator for first component
            if len(function_components) != 0:
                function_operators.append(np.random.choice(OPERATORS))

            # add component and placeholder for parameter
            function_components.append(comp)

        return function_components, function_operators


    def mutate(self):
        ''' make point changes to an organism's function '''
        mutated = False

        new_comp = Organism.random_function(max_terms=1)[0][0]
        new_op = np.random.choice(OPERATORS)

        # try until we pick a valid mutation
        # always succeeds since replacement is always allowed
        while not mutated:
            choice = np.random.randint(0,4)
            if choice == 0:
                # replace one component with another component
                self._function_components[
                        np.random.randint(0, len(self._function_components))
                        ] = new_comp
                mutated = True

            if choice == 1:
                # replace an operator with another random one
                # only valid for >1 term function
                if len(self._function_operators) > 0:
                    self._function_operators[
                            np.random.randint(0, len(self._function_operators))
                            ] = new_op
                    mutated = True

            if choice == 2:
                # add a new term, only valid for < MAX_TERM functions
                if len(self._function_components) < MAX_TERMS:
                    self._function_components.append(new_comp)
                    self._function_operators.append(new_op)
                    mutated = True

            if choice == 3:
                # delete a term, only valid for >1 term functions
                if len(self._function_components) > 1:
                    idx = np.random.randint(0, len(self._function_components))
                    del self._function_components[idx]
                    if idx == 0:
                        del self._function_operators[0]
                    else:
                        del self._function_operators[idx - 1]
                    mutated = True


    def breed(self, org):
        ''' combine features of this organism with another into a new one '''
        comps = list(self._function_components)
        ops = list(self._function_operators)
        comps.extend(org.function_components)
        ops.extend(org.function_operators)

        # calling list on this avoids converting to np array
        new_comps = list(np.random.choice(comps, np.random.randint(1, MAX_TERMS), replace=False))
        new_ops = []
        while len(new_ops) < len(new_comps) - 1:
            if len(ops) > 0:
                # pick one from pile
                new_ops.append(np.random.choice(ops))
            else:
                # pick a random one
                new_ops.append(np.random.choice(OPERATORS))

        child = Organism(new_comps, new_ops)
        if np.random.uniform() > MUTATION_RATE:
            child.mutate()

        return child


# it would be more straightfotward to call organism's compute function directly,
# but we need these wrappers to use python multiprocessing
def compute_lazy(org, h=HAMILTONIANS[0]):
    ''' computes only if organism hasn't been computed yet '''
    if org.fitness is None:
        org.compute(h)
    return org

def compute(org, h=HAMILTONIANS[3]):
    org.compute(h)
    return org


if __name__ == '__main__':
    # initialize
    max_fit = 0
    best_o = None

    generation = 0
    population = [Organism() for x in range(0, POPULATION_SIZE)]

    with open('log.txt', 'a') as f:
        f.write('generation,min_fit,max_fit,avg_Fit,med_fit\n')

    while generation < MAX_GENERATIONS:
        generation += 1
        if generation % 10 == 0:
            print(generation)

        # parallelize!
        pool = Pool(processes=4)
        population = pool.map(compute_lazy, population)
        pool.close()
        fitnesses = [o.fitness for o in population]

        best_index = max(enumerate(fitnesses), key=lambda x: x[1])[0] 
        if population[best_index].fitness > max_fit:
            best_o = population[best_index]
            max_fit = best_o.fitness
            print('{}| new max: {}'.format(generation, max_fit))
            with open('population_best.txt', 'a') as f:
                org_data = {
                        'generation': generation,
                        'fitness': max_fit,
                        'function': best_o.function_string,
                        'function_asym': best_o.function_string_asym,
                        'params': list(best_o.best_fit),
                        'params_asym': list(best_o.best_fit_asym),
                        'energy_split': best_o.energy_split,
                        'split_error': best_o.split_error,
                        'fidelity': best_o.fidelity,
                        'fidelity_error': best_o.fidelity_error
                        }
                f.write(json.dumps(org_data,indent=4, sort_keys=True) + ',\n')

        with open('log.txt', 'a') as f:
            f.write('{},{},{},{}\n'.format(
                generation, 
                np.min(fitnesses), 
                np.max(fitnesses), 
                np.mean(fitnesses), 
                np.median(fitnesses)
            ))
            

        # remove organisms with rate proportional to fitness in population 
        remove_indices = []
        sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])

        for i, o in enumerate(sorted_population):
            survival_prob = float(i) / POPULATION_SIZE
            if np.random.uniform() > survival_prob or o[0].age > 20:
                # sorry, pal
                remove_indices.append(i)

        # remove indices starting from highest to avoid shifting list
        for index in sorted(remove_indices, reverse=True):
            del sorted_population[index]

        # unzip
        population = [_[0] for _ in sorted_population] 
        fitnesses = [_[1] for _ in sorted_population]

        # repopulate
        new_orgs = []
        while len(population) + len(new_orgs) < POPULATION_SIZE - MIGRATION_SIZE:
            org1, org2 = np.random.choice(population, 2, p=fitnesses / sum(fitnesses), replace=False)
            new_orgs.append(org1.breed(org2))

        for i in range(MIGRATION_SIZE):
            new_orgs.append(Organism())

        population.extend(new_orgs)

    # plot the winner, and the default hamiltonian exact solutions
    fig = plt.figure()
    ax = fig.add_subplot(111)

    h, energy_exact, xs = HAMILTONIANS[0]
    sign_even = 1 if energy_exact[1][int(H_DIM/4), 0] > 0 else -1
    sign_odd = 1 if energy_exact[1][int(H_DIM/4), 1] < 0 else -1

    ax.plot(xs, energy_exact[1][:H_DIM,0] * sign_even, 'r', linewidth=2, label='Exact even')
    ax.plot(xs, energy_exact[1][:H_DIM,1] * sign_odd, 'r--', linewidth=2, label='Exact odd')

    best_func = lambda x, p: eval(best_o.function_string) / norm(eval(best_o.function_string))
    best_func_asym = lambda x, p: eval(best_o.function_string_asym) / norm(eval(best_o.function_string_asym))
    ax.plot(xs, best_func(xs, best_o.best_fit), 'b', linewidth=2, label=best_o.function_string)
    ax.plot(xs, best_func_asym(xs, best_o.best_fit_asym), 'b--', linewidth=2, label=best_o.function_string)
    ax.legend()

    plt.savefig('best.pdf')
