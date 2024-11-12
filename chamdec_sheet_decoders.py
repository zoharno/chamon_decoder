import numpy as np
from pymatching import Matching

# All the functions needed to create the sheets and perform the MWPM decoding on the sheets

def stabs_in_diagonal(first_el, orientation, diag_num, px, num_stlayer):
    # stabilizers in each diagonal on layers starting with stabilizer

    # first_el specifies what is the first element of the layer, 0 - stabilizer, 1 - qubit
    # orientation specifies orientation of diagonal, 0 being right, 1 being left
    # diag_num specifies shift of diagonal, say 0 means the main diagonal

    diag = []  # initialize the variable that will store the index of the stabilizers in the diagonal

    # check that the layer includes such a diagonal number
    # raise an exception if index is out of range
    if (first_el + orientation) % 2 == 0:  # the cases where the diagonal index cant be larger than px-1.
        # see chamon_diagonals.jpg for the diagonal indices
        if abs(diag_num) >= px:  # if diagonal index is too high
            raise Exception("diagonal number out of range")
    if (first_el + orientation) % 2 == 1:  # the cases where the diagonal index cant be larger than px.
        if abs(diag_num) >= px + 1:  # if diagonal index is too high
            raise Exception("diagonal number out of range")

    # we need to define separately depending on the orientation of the diagonal
    # and initial element in layer

    # we are defining the terms for a loop going over the indices in the diagonal
    # start - the initial index in the diagonal
    # stop - an upper bound on the indices in the diagonal
    # step - the difference between a stabilizer in the diagonal and the next-next

    # define the step between the included stabilizers
    if orientation == 0:  # if the diagonal is to the right
        step = 2 * px + 1  # the difference between one stabilizer in the diagonal to the one after the next one
    elif orientation == 1:  # if the diagonal is to the left
        step = 2 * px - 1  # the difference between one stabilizer in the diagonal to the one after the next one
    else:
        raise Exception("orientation variable should be 0 or 1")

    # check value first_el is correct
    if first_el not in [0, 1]:
        raise Exception("first_el should be 0 or 1")

    # right diagonal for layer starting with stabilizer
    if first_el == 0 and orientation == 0:
        if diag_num >= 0:  # non negative diagonal index
            start = diag_num  # seems self explanatory...
            stop = num_stlayer - 2 * px * diag_num
        elif diag_num < 0:
            start = 2 * px * abs(diag_num)
            stop = num_stlayer
        else:
            raise Exception("in this orientation diag_num should be an integer")

    # right diagonal for layer starting with qubit
    if first_el == 1 and orientation == 0:
        if diag_num > 0:
            # we are adding first row separately here, so that difference between first element
            # in the loop to next one will be px
            diag.append(diag_num - 1)
            start = diag_num + px
            stop = num_stlayer - px - 2 * px * (diag_num - 1)
        elif diag_num < 0:
            diag.append(num_stlayer + diag_num)
            start = px + 2 * px * abs(diag_num + 1)
            stop = num_stlayer - px
        else:
            raise Exception("in this orientation diag_num should be a non zero integer")

    # left diagonal for layer starting with qubit
    if first_el == 1 and orientation == 1:
        if diag_num >= 0:
            start = px - 1 - diag_num
            stop = num_stlayer - 2 * px * diag_num - 1
        elif diag_num < 0:
            start = 2 * px * abs(diag_num) - 1 + px
            stop = num_stlayer
        else:
            raise Exception("in this orientation diag_num should be an integer")

    # left diagonal for layer starting with stabilizer
    if first_el == 0 and orientation == 1:
        if diag_num > 0:
            # we are adding first row separately here, so that difference between first element
            # in the loop to next one will be px
            diag.append(px - diag_num)
            start = 2 * px - 1 - diag_num
            stop = num_stlayer + 1 - 2 * px * diag_num
        elif diag_num < 0:
            diag.append(num_stlayer - px - 1 - diag_num)
            start = 2 * px - 1 + 2 * px * abs(diag_num + 1)
            stop = num_stlayer - px
        else:
            raise Exception("in this orientation diag_num should be a non zero integer")

    # a list of stabilizers included in the specified diagonal
    for i in range(start, stop, step):
        diag.append(i)
        diag.append(i + px)
    return diag


def create_sheets(px):
    # next we create the sheets using the function stabs_in_diagonal which gives the
    # stabilizers in a specific diagonal
    py = px  # for cubic case
    pz = px  # for cubic case
    numstab = 4 * px * py * pz
    num_stlayer = 2 * px * py  # number of stabilizers in xy layer
    num_orientation = 4
    num_sheets_per_or = px  # number of sheets per orientation for a CUBIC Chamon code
    num_sheets = num_orientation * num_sheets_per_or  # total number of sheets

    num_stabs_per_sheet = int(numstab / num_sheets_per_or)  # number of stabilizers in a sheet
    # sheets = [[0 for _ in range(num_stabs_per_sheet)] for _ in range(num_sheets)]
    # initialize variable that will be
    # a list of all stabilizers in each sheet, every row being a different sheet
    sheets = np.zeros((num_sheets, num_stabs_per_sheet))

    # I am creating a list of all the pairs of diagonals in a sheet in a layer
    diagonal_indices = np.zeros((2, 2 * pz))  # initialize variable that will be
    # the pairs of diagonal in each layer included in the same sheet
    # notice that the diagonal 0 is alone in a sheet, and the index paired to it here
    # will be ignored afterwards
    for i in range(2 * px):
        first_diag = int(i / 2) + 1
        second_diag = int((i + 1) / 2) - px
        diagonal_indices[0, i] = first_diag
        diagonal_indices[1, i] = second_diag

    # all sheets - creating a matrix, each line is the stabilizers included in the sheet
    for orientation in range(2):  # for an orientation
        for prop_i in range(2):  # for a direction of propagation - meaning if when
            # z is higher by one, does the diagonal index grow or decrease
            propagation_direction = pow(-1, prop_i)  # the propagation direction
            for sheet_index in range(px):  # for a sheet
                current_sheet = []  # initialize variable that will hold stabilizers in sheet
                # print("sheet index ", sheet_index )
                for layer in range(2 * pz):  # for a layer of constant z
                    temp = []  # initialize variable that will hold stabilizers in sheet in layer
                    first_el = (layer + 1) % 2  # the first element in this layer
                    diag_index = (2 * sheet_index + propagation_direction * layer) % (2 * pz)
                    # the index from diagonal_indices variables that are included in the sheet
                    if orientation == 1:  # change for opposite orientation
                        diag_index = (diag_index + 1) % (2 * pz)
                    diag_1 = int(diagonal_indices[0, diag_index])  # the index of one diagonal in sheet
                    diag_2 = int(diagonal_indices[1, diag_index])  # the index of the other diagonal in sheet
                    # print("diag 1 ", diag_1, ", diag 2 ", diag_2)
                    if diag_2 != 0:  # only diag_2 can get value 0. if not:
                        # add stabilizers from diag_1. Otherwise, there is only one
                        # diagonal in the sheet, of index 0
                        temp.extend(stabs_in_diagonal(first_el, orientation, diag_1, px, num_stlayer))
                    temp.extend(stabs_in_diagonal(first_el, orientation, diag_2, px, num_stlayer))
                    new_temp = [s + layer * num_stlayer for s in temp]  # add a value to get
                    # the currect stabilizer index, by the z-layer it is in..
                    current_sheet.extend(new_temp)  # add the stabilizers to variable
                sheets[2 * px * orientation + px * prop_i + sheet_index] = current_sheet
                # this adds all the stabilizers in specific sheet to general parameter

    return sheets, num_sheets, num_stabs_per_sheet, numstab


def find_qubits_in_sheet(parity_mat_reduced_stabs, numqubits):
    # check the sheet is valid and return the qubits included in the sheet
    qubit_appear = np.sum(parity_mat_reduced_stabs, 0).astype(int)  # in how many stabilizers does each qubit appear
    count = 0
    for i in qubit_appear:
        if i != 0 and i != 2:
            count += 1
    if count:  # if some qubit doesn't have a total of identity acting on it
        raise Exception("some qubit doesn't have a total of identity when all stabilizers applied")
    qubits_in_sheet = np.nonzero(qubit_appear)[0]  # should have the list of qubits twice,
    # with numqubits the difference between the repetitions
    half_index = len(qubits_in_sheet) // 2
    first_half = qubits_in_sheet[:half_index]  # the list of the qubits having x stab applied
    second_half = qubits_in_sheet[half_index:] - numqubits  # the list of the qubits having z stab applied
    if sum(first_half - second_half):  # if some qubit doesn't have both
        raise Exception("stabilizers don't add to zero on some of the qubits")
    return qubits_in_sheet


def convert_new2old(newindex, qubit_in_sheet):
    # convert index of qubit/stabilizer on sheet to index of qubit/stabilizer on full code
    oldindex = qubit_in_sheet[newindex]
    return oldindex


def convert_old2new(oldindex, qubit_in_sheet):
    # convert index of qubit/stabilizer on full code to index of qubit/stabilizer on sheet
    newindex = np.where(qubit_in_sheet == oldindex)[0]
    if len(newindex) == 1:
        return newindex[0]
    elif len(newindex) == 0:
        return -1
    else:
        raise Exception("a qubit somehow appears twice")


def modify_boundary_connection(start_val, stop_val, px):
    # modify connections so that they go through the boundaries when they should
    # pretty simplistic..
    if abs(start_val - stop_val) > px:
        if stop_val > start_val:
            stop_val -= 2 * px
        elif start_val > stop_val:
            start_val -= 2 * px
    return start_val, stop_val


def sheet_decoders(sheets, stabs, numqubits, num_sheets, noisy_qubits):
    # run the sheet-decoders:
    # initialize a list to keep all the pairs of stabilizers given by the correction
    stab_connections = []

    for sheet_index in range(num_sheets):

        stabs_in_sheet = sheets[sheet_index].astype(int)  # the stabilizers in the sheet
        parity_mat_reduced_stabs = stabs[stabs_in_sheet]  # parity matrix with reduced stabilizers

        qubits_in_sheet = find_qubits_in_sheet(parity_mat_reduced_stabs, numqubits)  # qubits in sheet
        parity_mat_reduced = parity_mat_reduced_stabs[:, qubits_in_sheet.tolist()]
        # parity matrix only on stabilizers in the sheet and qubits affected by stabilizers
        # in sheet

        num_qubits_per_sheet = len(qubits_in_sheet)
        noise_in_sheet = np.zeros(num_qubits_per_sheet)  # initialize variable that
        # will be only the noisy qubits in the sheet
        for i in range(len(noisy_qubits)):  # for all noisy qubits
            noisy_qubit_index = noisy_qubits[i]  # index of noisy qubits
            noisy_qubit_index_in_sheet = convert_old2new(noisy_qubit_index, qubits_in_sheet)
            # convert the index of the noisy qubit to the index in the sheet
            if noisy_qubit_index_in_sheet > -1:  # if noisy qubit in the sheet
                noise_in_sheet[noisy_qubit_index_in_sheet] = 1  # add noisy qubit

        # correction
        syndrome = np.matmul(parity_mat_reduced, noise_in_sheet) % 2  # syndrome inside the sheet
        matching = Matching(parity_mat_reduced)  # matching inside the sheet
        correction = matching.decode_to_matched_dets_array(syndrome)
        # and this indeed gives us the detection events that are paired!!!
        # https://pymatching.readthedocs.io/en/latest/api.html
        # (instead of the default which is the qubits that are flipped for a correction)

        num_corrections = correction.shape[0]  # the number of corrections

        for index in range(num_corrections):  # for a pair of stabilizers connected by the decoder
            # convert the index to the origianl index
            first_stab = convert_new2old(correction[index][0], stabs_in_sheet)
            second_stab = convert_new2old(correction[index][1], stabs_in_sheet)
            stab_connections.append((first_stab, second_stab))  # add connections
    return stab_connections


def sheet_weighted_decoders(sheets, stabs, numqubits, num_sheets, noisy_qubits, weights):
    # run the sheet-decoders:
    # initialize a list to keep all the pairs of stabilizers given by the correction
    stab_connections = []

    for sheet_index in range(num_sheets):

        stabs_in_sheet = sheets[sheet_index].astype(int)  # the stabilizers in the sheet
        parity_mat_reduced_stabs = stabs[stabs_in_sheet]  # parity matrix with reduced stabilizers

        qubits_in_sheet = find_qubits_in_sheet(parity_mat_reduced_stabs, numqubits)  # qubits in sheet
        parity_mat_reduced = parity_mat_reduced_stabs[:, qubits_in_sheet.tolist()]
        # parity matrix only on stabilizers in the sheet and qubits affected by stabilizers
        # in sheet
        weights_reduced = weights[qubits_in_sheet.tolist()]

        num_qubits_per_sheet = len(qubits_in_sheet)
        noise_in_sheet = np.zeros(num_qubits_per_sheet)  # initialize variable that
        # will be only the noisy qubits in the sheet
        for i in range(len(noisy_qubits)):  # for all noisy qubits
            noisy_qubit_index = noisy_qubits[i]  # index of noisy qubits
            noisy_qubit_index_in_sheet = convert_old2new(noisy_qubit_index, qubits_in_sheet)
            # convert the index of the noisy qubit to the index in the sheet
            if noisy_qubit_index_in_sheet > -1:  # if noisy qubit in the sheet
                noise_in_sheet[noisy_qubit_index_in_sheet] = 1  # add noisy qubit

        # correction
        syndrome = np.matmul(parity_mat_reduced, noise_in_sheet) % 2  # syndrome inside the sheet
        matching = Matching(parity_mat_reduced, weights=weights_reduced)  # matching inside the sheet
        correction = matching.decode_to_matched_dets_array(syndrome)
        # and this indeed gives us the detection events that are paired!!!
        # https://pymatching.readthedocs.io/en/latest/api.html
        # (instead of the default which is the qubits that are flipped for a correction)

        num_corrections = correction.shape[0]  # the number of corrections

        for index in range(num_corrections):  # for a pair of stabilizers connected by the decoder
            # convert the index to the origianl index
            first_stab = convert_new2old(correction[index][0], stabs_in_sheet)
            second_stab = convert_new2old(correction[index][1], stabs_in_sheet)
            stab_connections.append((first_stab, second_stab))  # add connections
    return stab_connections

def modify_weights(weights):
    # find qubits with negative weights:
    temp = weights.copy()
    temp[temp > 0] = 0
    negative_qubits = np.nonzero(temp)

    # check if weights has values other than inf and -inf:
    temp1 = weights.copy()
    temp1 = temp1[temp1 != np.inf]
    temp1 = temp1[temp1 != -np.inf]
    if np.size(temp1) < 2: # if all values are inf, -inf, or maybe one more value..
        weights[weights > 0] = 1000
        weights[weights < 0] = 0
    if max(weights) == np.inf:
        weights[weights == np.inf] = np.unique(weights)[-2]
    if min(weights) == -np.inf:
        weights[weights == -np.inf] = np.unique(weights)[1]
    if min(weights) < 0:
        weights = weights - min(weights)
    return weights, negative_qubits





