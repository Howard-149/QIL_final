import cirq
import sympy
import networkx as nx 
from ansatz_pool import s_pool, m_pool
from adapt_qaoa import generate_circuits
from utils.print_color import bcolors

nodes = 2
graph = nx.complete_graph(nodes)

single_qubit_pool = s_pool(graph)
multi_qubit_pool = m_pool(graph)

for element in single_qubit_pool:
    print("single qubit pool", element)

for element in multi_qubit_pool:
    print("multiple qubit pool", element)

print("\n\n")

base_circuit = cirq.Circuit()
circuits = generate_circuits(base_circuit, single_qubit_pool, sympy.symbols("a0"))
for c in circuits:
    print(bcolors.GREEN + "    Single-qubit ansatz element:" + bcolors.ENDC)
    print(c)
    print("")

print("\n\n")

base_circuit = cirq.Circuit()
circuits = generate_circuits(base_circuit, multi_qubit_pool, sympy.symbols("a0"))
for c in circuits:
    print(bcolors.GREEN + "    Double-qubit ansatz element:" + bcolors.ENDC)
    print(c)
    print("")