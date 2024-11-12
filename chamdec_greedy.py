import numpy as np
from chamdec_conversions import stab_2_location, position_2_neighbors, qubit_2_location
from chamdec_clusters_sweep import apply_pauli

# ----- add a preliminary step of a GREEDY DECODER ----------------
# This function identifies the diamond shape associated with single qubit errors
# and eliminates them first as a preliminary step...
def greedy_decoder(total_syndrome, px, numqubits):
    syndrome_loc = [stab_2_location(stab_index, px) for stab_index in np.nonzero(total_syndrome)[0]]
    syndromes_deleted = []
    x_corrections = []
    z_corrections = []
    all_corrections = []
    for q in range(numqubits):
        q_loc = qubit_2_location(q, px)
        stabs = position_2_neighbors(q_loc, px=px)
        x = stabs[2:]
        y = [stabs[i] for i in [0,1,4,5]]
        z = stabs[:4]
        if all(i in syndrome_loc for i in stabs):
            continue
        elif all(i in syndrome_loc for i in x):
            syndromes_deleted = apply_pauli(qubit=q_loc, loc_clust=syndromes_deleted, pauli=0, px=px)
            all_corrections.append(numqubits+q)
            x_corrections.append(q)
        elif all(i in syndrome_loc for i in z):
            syndromes_deleted = apply_pauli(qubit=q_loc, loc_clust=syndromes_deleted, pauli=2, px=px)
            all_corrections.append(q)
            z_corrections.append(q)
        elif all(i in syndrome_loc for i in y):
            syndromes_deleted = apply_pauli(qubit=q_loc, loc_clust=syndromes_deleted, pauli=1, px=px)
            if numqubits+q in all_corrections:
                x_corrections.remove(q)
                all_corrections.remove(numqubits+q)
            else:
                all_corrections.append(numqubits+q)
                x_corrections.append(q)
            if q in all_corrections:
                z_corrections.remove(q)
                all_corrections.remove(q)
            else:
                z_corrections.append(q)
                all_corrections.append(q)

    return all_corrections, x_corrections, z_corrections, syndromes_deleted

def combine_two_flips(noisy_qubits, corrections):
    noise = set(noisy_qubits)
    cor = set(corrections)

    noise_rems = noise.difference(cor)
    temp = cor.difference(noise)
    noise_rems.update(temp)

    return noise_rems