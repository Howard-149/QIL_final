import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
from itertools import combinations
import tkinter as tk
from ansatz_pool import s_pool, m_pool
from original_qaoa import train_qaoa, make_qaoa
from cost_hamiltonian import cost_hamiltonian, cc
from ui.draw_graph import GraphBuilderApp
from add_element_evolution import generate_circuits

def adapt_qaoa(graph, op_pool, p):
    expectation_layer = tfq.layers.Expectation()
    adapt_iter = 0
    tol = 1e-5

    params = []
    symbols = []
    base_circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(0, i) for i in range(len(graph.nodes()))]
    for q in qubits:
        base_circuit += cirq.H(q)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    track = []
    h = cc(qubits, graph)
    counter = 0
    while adapt_iter < p:
        params.append(0.01)
        symbols.append(sympy.symbols("c%d"%adapt_iter))
        base_circuit = cost_hamiltonian(base_circuit, qubits, graph, symbols[-1])
        # print("base circuit", base_circuit)

        init = None
        inner_counter = 0
        while True:
            params.append(0.01)
            symbols.append(sympy.symbols("m%d"%counter))
            counter += 1
            inner_counter += 1
            circuits = generate_circuits(base_circuit, op_pool, symbols[-1])
            var = tf.Variable(params, dtype=tf.float32, trainable=True)
            grads = []
            for c in circuits:
                with tf.GradientTape() as tape:
                    tape.watch(var)
                    exp_val = expectation_layer(c, symbol_names=[s.name for s in symbols], symbol_values=[var], operators=h)
                #grads.append(tf.math.abs(tape.gradient(exp_val, var)[-counter:])[0])
                grads.append(tf.math.reduce_mean(tf.math.abs((tape.gradient(exp_val, var)[-inner_counter:]))))
                #grads.append(tf.norm(tape.gradient(exp_val, var)[-counter:]))

            if init is None:
                init = max(grads)
            print(adapt_iter, counter, max(grads), init)
            grads_ = [i for i in grads if i >= init]
            if len(grads_) == 0 or counter > (adapt_iter * 2 + 25):
                break
            else:
                base_circuit = circuits[np.argmax(grads)]

            old = np.inf
            while True:
                with tf.GradientTape() as tape:
                    tape.watch(var)
                    guess = expectation_layer(base_circuit, symbol_names=[s.name for s in symbols], symbol_values=[var], operators=h)
                grads = tape.gradient(guess, var)
                opt.apply_gradients(zip([grads], [var]))
                guess = guess.numpy()[0][0]
                if abs(guess - old) < tol:
                    break
                old = guess
            params = var.numpy().tolist()

        track.append(guess)
        adapt_iter += 1
        print(adapt_iter, track[-1])
        # print("base circuit", base_circuit)

    return track

def real_cost(graph):
    sub_lists = []
    for i in range(len(graph.nodes()) + 1):
        temp = [list(x) for x in combinations(graph.nodes(), i)]
        sub_lists.extend(temp)

    cut_classic = []
    for sub_list in sub_lists:
        cut_classic.append(nx.algorithms.cuts.cut_size(graph, sub_list))
    
    sol = sub_lists[np.argmax(cut_classic)]

    c = 0
    for edge in graph.edges():
        s1 = -1 if edge[0] in sol else 1
        s2 = -1 if edge[0] in sol else 1
        c += (-1/2 * s1 * s2 * graph.edges[edge]['weight'])
    return c

# Build the graph
nodes = 3
complete_graph = nx.complete_graph(nodes)
# # # cycle_graph = nx.cycle_graph(nodes)
# # # cycle_graph.add_edge(0, 3)
# # # cycle_graph.add_edge(1, 4)
# # # cycle_graph.add_edge(2, 5)

graph = complete_graph
for (u, v) in graph.edges():
    graph.edges[u,v]['weight'] = np.random.uniform(0.1, 1.0)

# graph = nx.Graph()
# def set_graph(g):
#     global graph
#     graph = g

# root = tk.Tk()
# app = GraphBuilderApp(root, callback=set_graph)
# root.mainloop()

# nx.draw(complete_graph)
# plt.show()
# nx.draw(cycle_graph)
# plt.show()

### show the cost homiltonian
base_circuit = cirq.Circuit()
qubits = [cirq.GridQubit(0, i) for i in range(len(graph.nodes()))]
for q in qubits:
    base_circuit += cirq.H(q)
# base_circuit = cost_hamiltonian(base_circuit, qubits, graph, sympy.symbols("alpha0"))
# print(base_circuit)

h = cc(qubits, graph)
circuits = generate_circuits(base_circuit, [h], sympy.symbols("a0"))
for c in circuits:
    print(c)
###

solution = real_cost(graph)
print("real cost", solution)
max_p = 8

single_qubit_pool = s_pool(graph)
multi_qubit_pool = m_pool(graph)

qaoa_energies = []
ps = []

adapt_single = adapt_qaoa(graph, single_qubit_pool, max_p)
adapt_single = [i - solution for i in adapt_single]
adapt_multi = adapt_qaoa(graph, multi_qubit_pool, max_p)
adapt_multi = [i - solution for i in adapt_multi]

for p in range(1, max_p + 1):
    e = train_qaoa(make_qaoa(graph, p))
    print(p, e)
    qaoa_energies.append(e - solution)
    ps.append(p)

plt.plot(ps, qaoa_energies, label="QAOA")
plt.plot(ps, adapt_single, label="ADAPT-single")
plt.plot(ps, adapt_multi, label="ADAPT-multi")
plt.legend()
plt.yscale('log')
plt.ylabel("Energy")
plt.xlabel("p")
plt.show()
