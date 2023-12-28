from src.vqe_runner import VQERunner
from src.q_systems import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend, MatrixCacheBackend
from src.utils import LogUtils
from src.cache import *
from src.iter_vqe_utils import *
from src.molecules.molecules import *

import logging
import time
import numpy
import pandas
import datetime
# import qiskit


if __name__ == "__main__":

    # Define molecular system

    # bond distance in angstroms
    r = 0.75
    # frozen electronic orbitals
    frozen_els = None  # {'occupied': [0, 1], 'unoccupied': [6, 7]}
    q_system = H2O()

    # init logging
    LogUtils.log_config()

    # Create a UCCSD ansatz for the specified molecule
    ansatz = UCCSDExcitations(q_system.n_orbitals, q_system.n_electrons, 'f_exc').get_all_elements()

    # choose a backend to calculate expectation values
    backend = MatrixCacheBackend

    # create a cache of precomputed values to accelerate the simulation (optional)
    global_cache = GlobalCache(q_system)
    global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)
    # global_cache.calculate_commutators_sparse_matrices_dict(ansatz)

    # Create a VQE runner, and specify the minimizer
    optimizer = 'BFGS'
    optimizer_options = {'gtol': 10e-6}
    vqe_runner = VQERunner(q_system, backend=backend, print_var_parameters=False, use_ansatz_gradient=True,
                           optimizer=optimizer, optimizer_options=optimizer_options)

    t0 = time.time()
    result = vqe_runner.vqe_run(ansatz=ansatz, cache=global_cache)  # initial_var_parameters=var_parameters)
    t = time.time()

    gate_count = IterVQEQasmUtils.gate_count_from_ansatz(ansatz, q_system.n_orbitals)
    print(f"cnot_count = {gate_count['cnot_count']}")
    # results_data_frame.loc[iter_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e,
    #                                         'error': current_energy - exact_energy, 'n_iters': result['n_iters'],
    #                                         'cnot_count': gate_count['cnot_count'], 'u1_count': gate_count['u1_count'],
    #                                         'cnot_depth': gate_count['cnot_depth'], 'u1_depth': gate_count['u1_depth'],
    #                                         'element': element_to_add.element, 'element_qubits': element_qubits,
    #                                         'var_parameters': 0}
    
    
    logging.critical(result)
    print(result)
    print('Time for running: ', t - t0)

    print('Pizza')


