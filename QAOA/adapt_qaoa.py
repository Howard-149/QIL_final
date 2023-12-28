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
from traditional_qaoa import train_qaoa, make_qaoa
from cost_hamiltonian import cost_hamiltonian, cc
from ui.draw_graph import GraphBuilderApp
from add_element_evolution import generate_circuits
import logging

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
    track_depth = []
    h = cc(qubits, graph)
    counter = 0
    while adapt_iter < p:
        logging.info(f'iteration: {adapt_iter}')

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
                grads.append(tf.math.reduce_mean(tf.math.abs((tape.gradient(exp_val, var)[-inner_counter:]))))

            if init is None:
                init = max(grads)
            print(adapt_iter, counter, max(grads), init)
            logging.info(f'iteration: {adapt_iter}, inner-counter: {inner_counter}, init-grad: {init}, max-grad: {max(grads)}')

            grads_ = [i for i in grads if i >= init]
            if len(grads_) == 0 or counter > (adapt_iter * 2 + 20):
                break
            else:
                base_circuit = circuits[np.argmax(grads)]
                logging.info(f'adding element {op_pool[np.argmax(grads)]}')

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
            logging.info(f'energy {guess}')
            params = var.numpy().tolist()

            # calculate depth
            depth = len(cirq.Circuit(base_circuit.all_operations()))
            logging.info(f'iteration: {adapt_iter}, depth: {depth}')

        track.append(guess)

        # calculate depth
        depth = len(cirq.Circuit(base_circuit.all_operations()))
        print(depth)
        logging.info(f'iteration end: {adapt_iter}, depth: {depth}')
        track_depth.append(depth)

        adapt_iter += 1
        print(adapt_iter, track[-1])
        # print("base circuit", base_circuit)

    return track, track_depth

def real_cost(graph):
    sub_lists = []
    for i in range(len(graph.nodes()) + 1):
        temp = [list(x) for x in combinations(graph.nodes(), i)]
        sub_lists.extend(temp)

    cut_classic = []
    for sub_list in sub_lists:
        cut_classic.append(nx.algorithms.cuts.cut_size(graph, sub_list, weight='weight'))
    
    sol = sub_lists[np.argmax(cut_classic)]

    print("sol: ", sol)

    c = 0
    for edge in graph.edges():
        s1 = -1 if edge[0] in sol else 1
        s2 = -1 if edge[1] in sol else 1
        c += (-1/2 * s1 * s2 * graph.edges[edge]['weight'])
    return -1 * c


logging.basicConfig(filename='./results/example.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### Build the graph -----------------------------------------------------------
# nodes = 5
# complete_graph = nx.complete_graph(nodes)
# cycle_graph = nx.cycle_graph(nodes)
# cycle_graph.add_edge(0, 3)
# cycle_graph.add_edge(1, 4)
# cycle_graph.add_edge(2, 5)

# graph = complete_graph
# for (u, v) in graph.edges():
#     graph.edges[u,v]['weight'] = np.random.uniform(0.1, 1.0)
### ---------------------------------------------------------------------------

graph = nx.Graph()
def set_graph(g):
    global graph
    graph = g

root = tk.Tk()
app = GraphBuilderApp(root, callback=set_graph)
root.mainloop()


print(graph)
logging.info(f'graph: {graph}')
for edge in graph.edges():
    print(edge, graph.edges[edge]['weight'])
    logging.info(f'edge {edge}, weight: {graph.edges[edge]["weight"]}')

### show the cost homiltonian -------------------------------------------------
# bc = cirq.Circuit()
# qubits = [cirq.GridQubit(0, i) for i in range(len(graph.nodes()))]
# print("qubits", qubits)
# for q in qubits:
#     bc += cirq.H(q)
# bc = cost_hamiltonian(bc, qubits, graph, sympy.symbols("alpha0"))
# print(bc)
### ---------------------------------------------------------------------------


solution = real_cost(graph)
print("real cost", solution)
logging.info(f'real cost: {solution}')
max_p = 10

single_qubit_pool = s_pool(graph)
multi_qubit_pool = m_pool(graph)

qaoa_energies = []
ps = []

### Single qubit QAOA
adapt_single, adapt_single_depth = adapt_qaoa(graph, single_qubit_pool, max_p)
adapt_single = [i - solution for i in adapt_single]
logging.info(f"error: {adapt_single}")
logging.info(f"depth: {adapt_single_depth}")
### ---------------------------------------------------------------------------

### Double qubit QAOA
adapt_multi, adapt_multi_depth = adapt_qaoa(graph, multi_qubit_pool, max_p)
adapt_multi = [i - solution for i in adapt_multi]
logging.info(f"error: {adapt_multi}")
logging.info(f"depth: {adapt_multi_depth}")
### ---------------------------------------------------------------------------

### Traditional QAOA 
for p in range(1, max_p + 1):
    logging.info(f'iteration: {p-1}')
    e = train_qaoa(make_qaoa(graph, p))
    print(p, e)
    logging.info(f'iteration: {p-1}, energy: {e}')
    qaoa_energies.append(e - solution)
    ps.append(p)
### ---------------------------------------------------------------------------


### Draw
experments = {
    'QAOA':qaoa_energies,
    'ADAPT-single-QAOA': adapt_single,
    'ADAPT-double-QAOA':adapt_multi
}

for name, y in experments.items():
    plt.semilogy(ps, y, marker='o', linestyle='-', label=name)

plt.xlabel('iteration')
plt.ylabel('QAOA Energy Error(log scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid with dashed lines

# Adjust the appearance of the legend
legend = plt.legend()
legend.get_frame().set_linewidth(0.7)

# Customize the appearance of the plot
plt.tight_layout()  # Adjust layout for better appearance

# Save the figure as an image (optional)
plt.savefig('./results/example.png', dpi=300, bbox_inches='tight')

plt.show()