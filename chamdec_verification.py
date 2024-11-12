import numpy as np
from chamdec_conversions import loc_2_number


def convert_correction(correction, c_type, numqubits, px):
    correction_num = []
    if c_type == "z" or c_type == 2:
        extra = 0
    elif c_type == "x" or c_type == 0:
        extra = numqubits
    else:
        raise Exception("type should be x or z")

    for c in correction:
        c_num = loc_2_number(c, px) + extra
        correction_num.append(c_num)

    return correction_num


def all_paulis(noisy_qubits, x_correction, z_correction):
    x_correction.extend(z_correction)
    correction = set(x_correction)
    noise = set(noisy_qubits)
    remains = correction.difference(noise)
    noise_remains = noise.difference(correction)
    remains.update(noise_remains)
    return correction, remains

def remainder_from_indices(noisy_qubits, correction):
    cor = set(correction)
    noise = set(noisy_qubits)
    rem = cor.difference(noise)
    rem2 = noise.difference(cor)
    rem.update(rem2)
    return rem


def chamon_logicals(px, numqubits):
    zlogical1 = [0] * numqubits
    zlogical2 = [0] * numqubits
    zlogical3 = [0] * numqubits
    zlogical4 = [0] * numqubits
    xlogical1 = [0] * numqubits
    xlogical2 = [0] * numqubits
    xlogical3 = [0] * numqubits
    xlogical4 = [0] * numqubits
    numq_layer = 2 * px * px
    for i in range(numqubits):
        if i < numq_layer:
            if (i // px) % 2 == 0:
                zlogical1[i] = 1
            else:
                zlogical2[i] = 1
        elif i < 2*numq_layer:
            if (i // px) % 2 == 0:
                zlogical3[i] = 1
            else:
                zlogical4[i] = 1
        if i % (2*px) == 0 and (i // numq_layer) % 2 == 0:
            xlogical1[i] = 1
        elif i % (2*px) == 0 and (i // numq_layer) % 2 == 1:
            xlogical2[i] = 1
        elif i % (2*px) == px and (i // numq_layer) % 2 == 0:
            xlogical3[i] = 1
        elif i % (2*px) == px and (i // numq_layer) % 2 == 1:
            xlogical4[i] = 1

    return zlogical1, zlogical2, zlogical3, zlogical4, xlogical1, xlogical2, xlogical3, xlogical4


def chamon_full_logicals(px, numqubits):
    logicals = chamon_logicals(px, numqubits)
    zeros = [0]*numqubits
    full_logicals = []
    counter = 0
    for logical in logicals:
        if counter < 4:
            newl = zeros + logical
        else:
            newl = logical + zeros
        full_logicals.append(newl)
        counter += 1
    return full_logicals


def check_commute_logical(logical_op, noise_after_correction):
    logical = np.array(logical_op)
    noise = np.array(noise_after_correction)
    res = np.matmul(logical, noise)
    return int(res)

def check_any_logical_error(logicals, numqubits, rem):
    failure = 0
    for logical in logicals:
        noise_rem = np.zeros(2*numqubits)
        noise_rem[rem] = 1
        com = check_commute_logical(logical, noise_rem)
        if com%2:
            failure += 1
    return failure
