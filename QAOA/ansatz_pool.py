import cirq

def s_pool(maxcut_graph):
    pool = []
    cirq_qubits = [cirq.GridQubit(0, i) for i in range(len(maxcut_graph.nodes()))]
    mixing_ham = 0
    for node in maxcut_graph.nodes():
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.X(qubit))
    pool.append(mixing_ham)

    mixing_ham = 0
    for node in maxcut_graph.nodes():
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.Y(qubit))

    pool.append(mixing_ham)

    for node in maxcut_graph.nodes():
        mixing_ham = 0
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.X(qubit))
        pool.append(mixing_ham)

    for node in maxcut_graph.nodes():
        mixing_ham = 0
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.Y(qubit))
        pool.append(mixing_ham)
    return pool

def m_pool(maxcut_graph):
    pool = s_pool(maxcut_graph)
    cirq_qubits = [cirq.GridQubit(0, i) for i in range(len(maxcut_graph.nodes()))]
    for pauli in [cirq.X, cirq.Y, cirq.Z]:
        for node1 in range(len(maxcut_graph.nodes())):
            for node2 in range(node1, len(maxcut_graph.nodes())):
                mixing_ham = 0
                qubit1 = cirq_qubits[node1]
                qubit2 = cirq_qubits[node2]
                mixing_ham += cirq.PauliString(pauli(qubit1) * pauli(qubit2))
                if (mixing_ham not in pool):
                    pool.append(mixing_ham)
    return pool
