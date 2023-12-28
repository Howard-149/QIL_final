import logging
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
from cost_hamiltonian import cost_hamiltonian, cc

def mixing_hamiltonian(c, qubits, par):
    for i in range(len(qubits)):
        c += cirq.rx(2 * par).on(qubits[i])
    return c

def make_circuit(p, graph, qs):
    qaoa_circuit = cirq.Circuit()
    num_param = 2 * p 
    qaoa_parameters = sympy.symbols("q0:%d"%num_param)
    for i in qs:
        qaoa_circuit += cirq.H(i)
    for i in range(p):
        qaoa_circuit = cost_hamiltonian(qaoa_circuit, qs, graph, qaoa_parameters[2 * i])
        qaoa_circuit = mixing_hamiltonian(qaoa_circuit, qs, qaoa_parameters[2 * i + 1])
    return qaoa_circuit

def make_qaoa(graph, p):
    qs = [cirq.GridQubit(0, i) for i in range(len(graph.nodes()))]
    qaoa_circuit = make_circuit(p, graph, qs)
    
    depth = len(cirq.Circuit(qaoa_circuit.all_operations()))
    print(depth)
    logging.info(f'iteration: {p-1}, depth: {depth}')

    cost = cc(qs, graph)
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    outs = tfq.layers.PQC(qaoa_circuit, cost, differentiator=tfq.differentiators.Adjoint())(ins)
    qaoa = tf.keras.models.Model(inputs=ins, outputs=outs)
    return qaoa

def train_qaoa(qaoa):
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    tol = 1e-5
    old = np.inf

    initial = cirq.Circuit()
    inputs = tfq.convert_to_tensor([initial])

    while True:
        with tf.GradientTape() as tape:
            error = qaoa(inputs)
        
        grads = tape.gradient(error, qaoa.trainable_variables)
        opt.apply_gradients(zip(grads, qaoa.trainable_variables))
        error = error.numpy()[0][0]
        if abs(old - error) < tol:
            break
        old = error
    
    return error