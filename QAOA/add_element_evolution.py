import tensorflow_quantum as tfq

def generate_circuits(base_cir, ops, sym):
    circuit = []
    op = [tfq.util.exponential([i], [sym]) for i in ops]
    for c in op:
        circuit.append(base_cir + c)
    return circuit