import pyscf

def calculate_ground_state_energy():
    # Define the molecule
    atom_symbols = ['Be', 'H', 'H']
    atom_coordinates = [[0, 0, 0], [0, 0, 1.3], [0, 0, -1.3]]
    molecule = pyscf.gto.Mole()
    molecule.atom = list(zip(atom_symbols, atom_coordinates))
    molecule.basis = 'sto-3g'
    molecule.build()

    # Run the HF calculation
    mf = pyscf.scf.RHF(molecule)
    mf.verbose = 4  # Set the verbosity level for output
    ground_state_energy = mf.kernel()

    return ground_state_energy

# Call the function to calculate the ground state energy
energy = calculate_ground_state_energy()
print("Ground state energy:", energy)