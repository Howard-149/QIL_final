import cirq
from add_element_evolution import generate_circuits

def cc(qubits, g):
    c = 0
    for edge in g.edges():
        c += cirq.PauliString(1/2 * cirq.Z(qubits[edge[0]]) * cirq.Z(qubits[edge[1]]) * g.edges[edge]['weight'])
    return c

def cost_hamiltonian(c, qubits, graph, ps):
    h = cc(qubits, graph)
    circuits = generate_circuits(c, [h], ps)
    c += circuits[0]
    return c

def cost_hamiltonian_original(c, qubits, g, ps):
    for edge in g.edges():
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
        c += cirq.rz(ps).on(qubits[edge[1]])
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
    return c