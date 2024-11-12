import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib import pyplot as plt
from chamdec_conversions import stab_2_location, position_2_neighbors
from chamdec_verification import remainder_from_indices


def find_connected_components(connections_duplicates):
    # input is the list of connections between syndrome defects that result from
    # the MWPM on the different symmetry sheets
    # output is a list of clusters

    # start by removing duplicates by converting to set
    set_connections = set(connections_duplicates)
    edges = list(set_connections)

    # translate this into the form that is needed for the graph functions
    l0 = [a for (a, b) in edges]
    l1 = [b for (a, b) in edges]
    w = np.ones(len(l0))

    # the default of the connected component function is to look at all vertices,
    # even the ones without edges, so we want to have only vertices with edges
    # so are creating dictionaries from the original vertex number to the new
    all_vals = l0 + l1
    set_unique_vals = set(all_vals)
    unique_vals = list(set_unique_vals)
    dict_old2new = {unique_vals[i]:i for i in range(len(unique_vals))}
    dict_new2old = {i: unique_vals[i] for i in range(len(unique_vals))}
    new_l0 = [dict_old2new[i] for i in l0]
    new_l1 = [dict_old2new[i] for i in l1]
    m = max(max(new_l0), max(new_l1)) + 1

    l_temp = csr_matrix((w,(new_l0,new_l1)),shape=(m,m))

    # apply the connected component function
    n_components, labels = connected_components(csgraph=l_temp, directed=False, return_labels=True)

    # create a list of defects in cluster, and convert back to original labels
    clusters = []
    for j in range(n_components):
        clusters.append([dict_new2old[i] for i, x in enumerate(labels) if x == j])

    return clusters





def cluster_location(px, clusteri):
    # input is the list of stabilizers in the cluster by their number
    # returns a list of locations of stabilizers in the cluster
    loc_clust = []
    xvals = []
    yvals = []
    zvals = []
    for stab in clusteri:
        stab_loc = stab_2_location(stab, px)
        xvals.append(stab_loc[0])
        yvals.append(stab_loc[1])
        zvals.append(stab_loc[2])
        loc_clust.append(stab_loc)
    return loc_clust, xvals, yvals, zvals


def num_layers(aset, px):
    # gives the number of layers in a axis
    # the input is the values in the axis in the form of a set
    max_dist = 0
    layer_min = 0
    layer_max = 0
    for x1 in aset:
        for x2 in aset:
            dist_reg = abs(x1 - x2)  # regular distance
            dist_boundary = abs(abs(x1 - x2) - 2 * px)  # distance through the boundary
            if dist_reg <= dist_boundary:
                dist = dist_reg
                xmin = min(x1, x2)
                xmax = max(x1, x2)
            else:
                dist = dist_boundary
                xmin = max(x1, x2)
                xmax = min(x1,x2)
            if dist > max_dist:  # update if larger than any other dist so far
                max_dist = dist
                layer_min = xmin
                layer_max = xmax
    layers_num = max_dist + 1 # number of layers
    return layers_num, layer_min, layer_max


def num_layers_lst(alist, px):
    # gives the number of layers in a axis
    # the input is the values in the axis in the form of a list
    aset = set(alist)
    (layers_num, layer_min, layer_max) = num_layers(aset, px)
    return layers_num, layer_min, layer_max


def largest_axis(xl, yl, zl):
    # returns the axis with most layers in the cluster
    n_layers = [xl, yl, zl]
    max_val = max(n_layers)
    max_index = n_layers.index(max_val)
    return max_index


# ------- New method - finding qubit with three neighboring syndrome defects -------
def identify_correction(loc_clust,px):
    # finds a qubit with three neighboring syndrome defects and returns the correction
    # that should be done
    # the input is the cluster of syndrome defects in the form of their locations
    for stab in loc_clust:
        qubits = position_2_neighbors(stab, px=px)
        for q in qubits:
            neighboring_stabs = position_2_neighbors(q, px=px)
            neighboring_defects = set(loc_clust).intersection(set(neighboring_stabs))
            if len(neighboring_defects) >= 3:
                # if len(neighboring_defects) >= 5:
                #     # commented this out since this can definitely occur, and don't want to
                #     # give up if it does
                #     raise Exception("5 or more neighboring defects spotted")
                defects_list = list(neighboring_defects)
                num_defects = len(defects_list)
                defects_type = [None]*num_defects
                for d in range(num_defects):
                    defects_type[d] = neighboring_stabs.index(defects_list[d])
                x_defects = [0,1]
                y_defects = [2,3]
                z_defects = [4,5]
                if all(value in defects_type for value in x_defects):
                    if any(value in defects_type for value in y_defects):
                        if any(value in defects_type for value in z_defects):
                            continue
                        else:
                            correction = ("Z", q)
                            return correction
                    elif any(value in defects_type for value in z_defects):
                        correction = ("Y", q)
                        return correction
                if all(value in defects_type for value in y_defects):
                    if any(value in defects_type for value in z_defects):
                        if any(value in defects_type for value in x_defects):
                            continue
                        else:
                            correction = ("X", q)
                            return correction
                    elif any(value in defects_type for value in x_defects):
                        correction = ("Z", q)
                        return correction
                if all(value in defects_type for value in z_defects):
                    if any(value in defects_type for value in x_defects):
                        if any(value in defects_type for value in y_defects):
                            continue
                        else:
                            correction = ("Y", q)
                            return correction
                    elif any(value in defects_type for value in y_defects):
                        correction = ("X", q)
                        return correction
    return -1

# -------- BROOM / SWEEP DECODER - first find a box around the cluster, then sweep down and back..

def find_crossing_wall_half(clusteri_conn_loc, axis, px):
    list_cons = list(clusteri_conn_loc)
    for connection in list_cons:
        a0 = connection[0][axis]
        a1 = connection[1][axis]
        if a0 == a1:
            continue
        else:
            minc = min(a0, a1)
            maxc = max(a0, a1)
            if maxc - minc <= px:
                return (minc + 0.5) % (2*px)
            else:
                return (maxc + 0.5) % (2*px)
    return a0



def cluster_connections(stab_connections, clusteri):
    a = [(conn_a, conn_b) for conn_a, conn_b in stab_connections if conn_a in clusteri or conn_b in clusteri]
    clusteri_connections = set(a)
    return clusteri_connections


def cluster_connections_loc(stab_connections, clusteri, px):
    a = [(stab_2_location(conn_a, px), stab_2_location(conn_b, px)) for conn_a, conn_b in stab_connections if conn_a in clusteri or conn_b in clusteri]
    clusteri_conn_loc = set(a)
    return clusteri_conn_loc


def check_wall_crosses(wall, axis, clusteri_conn_loc, px):
    cross = False
    for connection in clusteri_conn_loc:
        conn_a_x = connection[0][axis]
        conn_b_x = connection[1][axis]
        minc = min(conn_a_x, conn_b_x)
        maxc = max(conn_a_x, conn_b_x)
        if maxc - minc > px:
            if wall >=  maxc or wall <= minc:
                cross = True
                return cross
        else:
            if wall >= minc and wall <= maxc:
                cross = True
                return cross
    return cross


def border_box_half(axis, clusteri_conn_loc, px):
    wall_plus = find_crossing_wall_half(clusteri_conn_loc, axis, px)
    if wall_plus == np.floor(wall_plus):
        return wall_plus, wall_plus
    wall_minus = wall_plus
    wall_plus_final = wall_minus_final = None
    for i in range(2*px):
        wall_plus = (wall_plus + 1) % (2*px)
        cross = check_wall_crosses(wall_plus, axis, clusteri_conn_loc, px)
        if not cross:
            wall_plus_final = wall_plus
            break
    for i in range(2*px):
        wall_minus = (wall_minus - 1) % (2*px)
        cross = check_wall_crosses(wall_minus, axis, clusteri_conn_loc, px)
        if not cross:
            wall_minus_final = wall_minus
            break
    if wall_minus_final is None or wall_plus_final is None:
        return -1
    else:
        return wall_minus_final, wall_plus_final


def find_box_half(clusteri_conn_loc, px):
    borderx = border_box_half(0, clusteri_conn_loc, px)
    bordery = border_box_half(1, clusteri_conn_loc, px)
    borderz = border_box_half(2, clusteri_conn_loc, px)
    return borderx, bordery, borderz



def find_defect_on_border(axis, wall, loc_clust):
    for defect in loc_clust:
        if defect[axis] == wall:
            return defect
    return -1


def find_qubit_2correct(axis, defect, px):
    # pushes the defect down by applying a pauli x
    # saves the correction that was done
    # changes loc_clust accordingly
    defect_change = defect[axis]
    qubit_axis= (defect_change - 1) % (2*px)
    if axis == 0:
        qubit = (qubit_axis, defect[1], defect[2])
    elif axis == 1:
        qubit = (defect[0], qubit_axis, defect[2])
    elif axis == 2:
        qubit = (defect[0], defect[1], qubit_axis)
    else:
        raise Exception("axis should be 0 1 or 2")
    return qubit


def pauli_on_single_stab(stab, loc_clust):
    if stab in loc_clust:
        loc_clust.remove(stab)
    else:
        loc_clust.append(stab)
    return loc_clust


def apply_pauli(qubit, loc_clust, pauli, px):
    neighboring_stabs = position_2_neighbors(qubit, px)
    for i in range(len(neighboring_stabs)):
        if pauli == 0 and i > 1:
            loc_clust = pauli_on_single_stab(neighboring_stabs[i], loc_clust)
        if pauli == 1 and i in [0,1,4,5]:
            loc_clust = pauli_on_single_stab(neighboring_stabs[i], loc_clust)
        if pauli == 2 and i < 4:
            loc_clust = pauli_on_single_stab(neighboring_stabs[i], loc_clust)
    return loc_clust


def sweep_single_direction_half(axis, boxi, loc_clust, px, plot=False):
    # this is assuming a cube
    # note that the y direction may get sweeped out of the box, and so this isn't
    # as good as it should be. Is there a way not to sweep out of the box?
    lims = boxi[axis]
    bottom = (lims[0] + 0.5) % (2*px)
    top = (lims[1] - 0.5) % (2*px)
    current = top
    if axis == 2:
        pauli = 0
    elif axis == 0:
        pauli = 2
    elif axis == 1:
        pauli = 2
    else:
        raise Exception("axis of pushing should be 0 or 1 or 2 ")
    x_corrections = []
    while (current - bottom) % (2*px) > 1:
        defect = find_defect_on_border(axis=axis, wall=current, loc_clust=loc_clust)
        if defect == -1:
            current = (current - 1) % (2*px)
        else:
            qubit = find_qubit_2correct(axis=axis, defect=defect, px=px)
            x_corrections.append(qubit)
            loc_clust = apply_pauli(qubit=qubit, loc_clust=loc_clust, pauli=pauli, px=px)
    if plot:
        plot_single_cluster(px, vis=0.8, single_cluster=loc_clust, cluster_type="loc")
    return loc_clust, x_corrections

def plot_single_cluster(px, vis = 0.8, single_cluster=None, cluster_type = "loc"):
    # Doesn't really work, probably just better to drop this...

    # apply here the default value, since didnt want mutable default argumaent value
    if single_cluster is None:
        single_cluster = []

    # initialize the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot all the stabilizers with some transparency
    num_stabs = 4 * px * px *px
    x = np.zeros(num_stabs)
    y = np.zeros(num_stabs)
    z = np.zeros(num_stabs)
    for i in range(num_stabs):
        loc = stab_2_location(i, px)
        x[i] = loc[0]
        y[i] = loc[1]
        z[i] = loc[2]
    ax.scatter(x, y, z, c=z, marker='o', s=100, alpha=vis/px)

    # plot the syndrome defects by color
    n = len(single_cluster)
    for i in range(n):
        current_cluster = single_cluster
        for current_stab in current_cluster:
            if cluster_type == "loc":
                loc = current_stab
            else:
                loc = stab_2_location(current_stab, px)
            ax.scatter(loc[0], loc[1], loc[2], marker='o', s=100)

    plt.savefig('chamon_pics\single_cluster', dpi=300.0)
    return fig

# the sweep decoder:
def correct_cluster_all(clusters, stab_connections, px, plot,to_print=False, dec_type = " "):
    x_correction = []
    z_correction = []
    success = True
    for clusteri in clusters:
        (loc_clust, xvals, yvals, zvals) = cluster_location(px, clusteri)
        clusteri_conn_loc = cluster_connections_loc(stab_connections, clusteri, px)
        boxi_temp = find_box_half(clusteri_conn_loc, px)
        boxi = [None]*3
        for i in range(3):
            if boxi_temp[i] == -1:
                if to_print:
                    print(boxi_temp)
                boxi[i] = (2*px-0.5,2*px-0.5)
            else:
                boxi[i] = boxi_temp[i]
        if to_print:
            print(boxi)
            print(loc_clust)
        down_clust_correction = sweep_single_direction_half(axis=2, boxi=boxi, loc_clust=loc_clust, px=px, plot=plot)
        x_correction = remainder_from_indices(x_correction, down_clust_correction[1])
        if to_print:
            print(down_clust_correction[0])
        clust_correction = sweep_single_direction_half(axis=0, boxi=boxi, loc_clust=down_clust_correction[0], px=px,
                                                  plot=plot)
        z_correction = remainder_from_indices(z_correction, clust_correction[1])
        if to_print:
            print(clust_correction[0])
        if len(clust_correction[0]) > 0:
            success = False
            print(dec_type + "False, sweep of cluster didn't eliminate syndrome defects")
            print(clusteri)
            print(loc_clust)
            print(clust_correction[0])
            print(down_clust_correction[0])

    return success, x_correction, z_correction
