import random

def depolarizing_noise(numqubits, p_errx, p_erry, p_errz, seed_val):
    # IMPORTANT NOTE: notice that here the noise isn't what we would expect from depolarizing noise, but rather for
    # the X, Y, and Z noise parameters being independent.
    # So when analyzing the data the noise rate is converted to the usual depolarizing noise parameters...
    random.seed(seed_val)
    noisy_qubits = []
    for i in range(2*numqubits):
        r = random.random()
        if i < numqubits and r < p_errz:
            noisy_qubits.append(i)
        if i >= numqubits and r < p_errx:
            noisy_qubits.append(i)
    for j in range(numqubits):
        r = random.random()
        if r < p_erry:
            if j in noisy_qubits:
                noisy_qubits.remove(j)
            else:
                noisy_qubits.append(j)
            if j+numqubits in noisy_qubits:
                noisy_qubits.remove(j+numqubits)
            else:
                noisy_qubits.append(j+numqubits)
    return noisy_qubits
