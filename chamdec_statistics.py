import numpy as np
import time
from ldpc import bp_decoder, bposd_decoder
from datetime import datetime
import os

from chamdec_noise import depolarizing_noise
from chamdec_stabs_syndromes import create_chamon_stabs, find_total_syndrome, find_syndrome_from_noise_nums
from chamdec_sheet_decoders import create_sheets, sheet_decoders, sheet_weighted_decoders, modify_weights
from chamdec_clusters_sweep import find_connected_components, correct_cluster_all
from chamdec_verification import convert_correction, all_paulis, chamon_full_logicals, \
    remainder_from_indices, check_any_logical_error
from chamdec_greedy import greedy_decoder, combine_two_flips

def chamdec_exec(folder, px, p_err, num_rep, seed_num, decoder_type):
    print_non_valids = True # print out when we get a result which is not valid (should only happen for bp)
    print_stat = True # prints progress status to console, prints when finished with 1/10, 1/4, 1/2, 3/4, 9/10 of the iterations
    print_cluster_evolution = False # prints the defects in all clusters for all steps. KEEP FALSE for many iterations

    # bp variables:
    bp_method = 'product_sum'
    bp_iter = 20*px # number of iterations of bp

    # osd variables
    osd_method = 'combination_sweep'
    osd_order = 40

    # print to the console the variables for this run:
    print("px:" + str(px) + ", err:" + str(p_err) + ", dec: " + decoder_type + ", reps:" + str(num_rep) + ", bp iters:" + str(bp_iter) + ", seed:" +str(seed_num))

    # -------- general stuff -------------------------------------------------
    # creating the stabilizers, a list of stabilizers in each sheet, and the logicals
    py = px # since we are working with a cubic chamon code
    pz = px # since we are working with a cubic chamon code
    numqubits = 4 * px * py * pz # total number of data qubits (there are the same number of stabilizer qubits)
    logicals = chamon_full_logicals(px, numqubits)  # create the logical operators
    zlogicals = logicals[:3] # variable for only the z logicals
    if decoder_type == "greedy_sheets" or decoder_type =="bp_sheets" or decoder_type == "sheets":
        (sheets, num_sheets, num_stabs_per_sheet, numstab) = create_sheets(px)  # create sheets
    stabs = create_chamon_stabs(px, py, pz)  # create stabilizers
    failures = np.zeros(num_rep) # initialize variable for errors

    # initialize the variables that keep track of how much time is spent on each step in the bp_sheets decoder
    start_time = time.time()
    time_bp = 0
    time_weighting = 0
    time_matching = 0
    time_sweeping = 0
    time_organizing = 0
    time_converting = 0
    time_verifying = 0

    t_set = 0
    t_merge = 0
    t_merge2 = 0

    # run the decoding algorithm
    for i in range(num_rep):
        if print_stat: # print to console the progress status
            if i == 0:
                print("started, i=" + str(i) + ", " + str(datetime.now()))
            elif i == int(num_rep/10):
                print("done 1/10, i=" + str(i) + ", " + str(datetime.now()))
            elif i == int(num_rep/4):
                print("done 1/4, i=" + str(i) + ", " + str(datetime.now()))
            elif i == int(num_rep/2):
                print("done 1/2, i=" + str(i) + ", " + str(datetime.now()))
            elif i == int(3*num_rep/4):
                print("done 3/4, i=" + str(i) + ", " + str(datetime.now()))
            elif i == int(9*num_rep/10):
                print("done 9/10, i=" + str(i) + ", " + str(datetime.now()))
            elif i == (num_rep-1):
                print("last iteration, i=" + str(i) + ", " + str(datetime.now()))
        seed_run = seed_num + i
        # --------- STEP 1 - sampling noise  and measuring syndrome------------------------
        # sample a noise instance from the depolarizing noise with p_err
        noisy_qubits = depolarizing_noise(numqubits=numqubits, p_errx=p_err / 3, p_erry=p_err / 3, p_errz=p_err / 3,
                                          seed_val=seed_run)
        # calculate the syndrome for this specific noise instance
        (noisy_qubits_01, total_syndrome) = find_syndrome_from_noise_nums(noisy_qubits, numqubits, stabs)

        fail = 0 # initializing failure parameter

        if decoder_type == "bp_osd":
            # obtain the recovery operator given by the BP-OSD decoder
            bpd_osd = bposd_decoder(
                stabs,  # the parity check matrix
                error_rate=p_err,
                max_iter=bp_iter,  # the maximum number of iterations for BP)
                bp_method=bp_method,  # BP method. The other option is `product_sum'
                # i think before we had bp_method='ms'
                ms_scaling_factor=0,
                # min sum scaling factor. If set to zero the variable scaling factor method is used
                # osd_method="osd_cs",  # the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                osd_method=osd_method,  # the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                # osd_order=7  # the osd search depth
                osd_order=osd_order
            )
            corr_bposd_01 = bpd_osd.decode(total_syndrome) # apply the decoder
            corr_bposd = np.nonzero(corr_bposd_01)[0] # organize the result, to obtain the recovery operator in desired format
            rem_bp_osd_01 = (bpd_osd.osdw_decoding + noisy_qubits_01) % 2 # the noise after recovery operator
            rem_bp_osd = list(np.nonzero(rem_bp_osd_01)[0]) # list of qubits which are still flipped after recovery

            # check the validity of the recovery operator
            syndrome_bp_osd = find_total_syndrome(list(corr_bposd), numqubits, stabs) # the syndrome of recovery alone
            if np.array_equal(syndrome_bp_osd, total_syndrome) is False: # verify that the syndrome for recovery and noise are identical
                if print_non_valids:
                    print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(
                        seed_run) + ", correction from bp+osd is not valid")
                fail = 2
            else:
                # verify that remaining flips of noise+recovery commute with all the logicals
                logical_err = check_any_logical_error(logicals, numqubits, list(rem_bp_osd))
                if logical_err > 0:
                    fail = 1

        elif decoder_type == "bp":
            # obtain the recovery operator given by the BP decoder
            bpd = bp_decoder(
                stabs,  # the parity check matrix
                error_rate=p_err,  # the error rate on each bit
                max_iter=bp_iter,  # the maximum iteration depth for BP
                bp_method=bp_method,  # BP method. "product_sum or `minimum_sum'
                # i think before we had bp_method='ms'
                # channel_probs=[None]  # channel probability probabilities. Will overide error rate.
            )
            corr_bp_01 = bpd.decode(total_syndrome) # apply the decoder
            corr_bp = np.nonzero(corr_bp_01)[0] # organize the result, to obtain the recovery operator in desired format
            rem_bp = list(remainder_from_indices(noisy_qubits, corr_bp)) # list of qubits which are still flipped after recovery

            # checking correction validity
            syndrome_bp = find_total_syndrome(list(corr_bp), numqubits, stabs) # the syndrome of recovery alone
            if np.array_equal(syndrome_bp, total_syndrome) is False: # verify that the syndrome for recovery and noise are identical
                if print_non_valids:
                    print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(
                        seed_run) + ", correction from bp is not valid")
                fail = 2
            else:
                # verify that remaining flips of noise+recovery commute with all the logicals
                logical_err = check_any_logical_error(logicals, numqubits, list(rem_bp))
                if logical_err > 0:
                    fail = 1

        elif decoder_type == "bp_sheets":
            # obtain the recovery operator given by the BP-matching decoder
            # notice that here we also compute the time for each of the steps
            beginning_time = time.time()
            error_for_decoder = p_err
            # create the bp decoder for the initial step
            bpd = bp_decoder(
                stabs,  # the parity check matrix
                error_rate=error_for_decoder,  # the error rate on each bit
                max_iter=bp_iter,  # the maximum iteration depth for BP
                bp_method=bp_method,  # BP method. The other option is `minimum_sum'
                # i think before we had bp_method='product_sum'
                # channel_probs=[None]  # channel probability probabilities. Will overide error rate.
            )

            corr_bp_01 = bpd.decode(total_syndrome) # apply the bp decoder
            end_time = time.time()
            time_bp += (end_time - beginning_time)
            beginning_time = time.time()

            # soft decoding - find qubit weights and modify to fit as input for pymatching
            weights = bpd.log_prob_ratios # take the weights - soft decoding from the bp decoder
            (weights, negative_qubits) = modify_weights(weights) # put boundaries on minimal and maximal values of weights

            end_time = time.time()
            time_weighting += (end_time - beginning_time)
            beginning_time = time.time()

            # find clusters using MWPM on the symmetries with the weights for the syndrome defects obtained by BP
            # run the MWPM decoders on the symmetries with the weights:
            stab_before_bp = sheet_weighted_decoders(sheets, stabs, numqubits, num_sheets, noisy_qubits, weights)
            if print_cluster_evolution:
                print("stabilizer connections from mwpm:")
                print(stab_before_bp)
            end_time = time.time()
            time_matching += (end_time - beginning_time)
            beginning_time = time.time()

            clusters_v2 = find_connected_components(stab_before_bp) # separate syndrome defects into connected components

            end_time = time.time()
            time_organizing += (end_time - beginning_time)
            beginning_time = time.time()

            # apply the sweeping decoder on every cluster separately
            corr_sheets_bp_loc = correct_cluster_all(clusters_v2, stab_before_bp, px, plot=False)

            end_time = time.time()
            time_sweeping += (end_time - beginning_time)
            beginning_time = time.time()

            # convert the recovery operator to desired format:
            x_corr_sheets_bp = convert_correction(corr_sheets_bp_loc[1], "x", numqubits, px)
            z_corr_sheets_bp = convert_correction(corr_sheets_bp_loc[2], "z", numqubits, px)
            (corr_sheets_bp, rem_sheets_bp) = all_paulis(noisy_qubits, x_corr_sheets_bp, z_corr_sheets_bp)

            end_time = time.time()
            time_converting += (end_time - beginning_time)
            beginning_time = time.time()

            # check correction validity
            if corr_sheets_bp_loc[0] is False: # if the decoder didnt return a valid correction
                print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(seed_run) + ", bp+sheets wasn't able to decode")
                fail = 2
            else:
                syndrome_sheets_bp = find_total_syndrome(list(corr_sheets_bp), numqubits, stabs)
                if np.array_equal(syndrome_sheets_bp, total_syndrome) is False: # verify that the syndrome for recovery and noise are identical
                    if print_non_valids:
                        print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(
                            seed_run) + ", correction from bp+sheets is not valid")
                    fail = 2
                else:
                    # verify that remaining flips of noise+recovery commute with all the logicals
                    logical_err = check_any_logical_error(logicals, numqubits, list(rem_sheets_bp))
                    if logical_err > 0:
                        fail = 1

            end_time = time.time()
            time_verifying += (end_time - beginning_time)

        elif decoder_type == "greedy_sheets":
            # obtain the recovery operator given by the greedy-matching decoder
            # apply the greedy initial step:
            (qubits_corrected, x_corrected, z_corrected, syndromes_deleted) = greedy_decoder(total_syndrome, px,
                                                                                             numqubits)
            # the new noise - after applying the initial greedy step:
            new_noise = list(combine_two_flips(noisy_qubits, qubits_corrected))

            # run the MWPM decoders on the symmetries:
            stab_connections = sheet_decoders(sheets, stabs, numqubits, num_sheets, new_noise)

            if len(new_noise) > 0: # if there is any noise left after greedy step
                clusters = find_connected_components(stab_connections)  # separate syndrome defects into connected components
            else:
                clusters = []

            # sweeping decoder
            corr_sheets_greedy_loc = correct_cluster_all(clusters, stab_connections, px, plot=False)
            x_corr = convert_correction(corr_sheets_greedy_loc[1], "x", numqubits, px)
            z_corr = convert_correction(corr_sheets_greedy_loc[2], "z", numqubits, px)

            # full correction and remainder - joining the greedy correction to sweep correction
            x_corr_greedy = [x + numqubits for x in x_corrected]
            x_corr_sheets_greedy = list(combine_two_flips(x_corr_greedy, x_corr))
            z_corr_sheets_greedy = list(combine_two_flips(z_corrected, z_corr))
            (corr_sheets_greedy, rem_sheets_greedy) = all_paulis(noisy_qubits, x_corr_sheets_greedy,
                                                                 z_corr_sheets_greedy)

            # check correction validity
            if corr_sheets_greedy_loc[0] is False: # if the decoder didnt return a valid correction
                print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(
                    seed_run) + ", greedy+sheets wasn't able to decode")
                fail = 2
            else:
                syndrome_sheets_greedy = find_total_syndrome(list(corr_sheets_greedy), numqubits, stabs)
                if np.array_equal(syndrome_sheets_greedy, total_syndrome) is False: # verify that the syndrome for recovery and noise are identical
                    if print_non_valids:
                        print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(
                            seed_run) + ", correction from greedy+sheets is not valid")
                    fail = 2
                else:
                    # verify that remaining flips of noise+recovery commute with all the logicals
                    logical_err = check_any_logical_error(logicals, numqubits, list(rem_sheets_greedy))
                    if logical_err > 0:
                        fail = 1

        elif decoder_type == "sheets":
            # run the MWPM decoders on the symmetries:
            stab_connections = sheet_decoders(sheets, stabs, numqubits, num_sheets, noisy_qubits)

            # divide the stabilizers into clusters by overlapping connections
            #stab_connections = set(stab_connections)  # convert to set
            #connections_set = [set(connection) for connection in stab_connections]
            #clusters_n_nums = merge_overlapping_sets(connections_set)
            #clusters = clusters_n_nums[0]

            clusters = find_connected_components(stab_connections) # separate syndrome defects into connected components

            # sweep decoder
            corr_sheets_loc = correct_cluster_all(clusters, stab_connections, px, plot=False)
            x_corr = convert_correction(corr_sheets_loc[1], "x", numqubits, px)
            z_corr = convert_correction(corr_sheets_loc[2], "z", numqubits, px)

            # full correction and remainder
            (corr_sheets, rem_sheets) = all_paulis(noisy_qubits, x_corr, z_corr)

            # check correction validity
            if corr_sheets_loc[0] is False: # if the decoder didnt return a valid correction
                print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(
                    seed_run) + ", sheets wasn't able to decode")
                fail = 2
            else:
                syndrome_sheets = find_total_syndrome(list(corr_sheets), numqubits, stabs)
                if np.array_equal(syndrome_sheets, total_syndrome) is False: # verify that the syndrome for recovery and noise are identical
                    if print_non_valids:
                        print("px" + str(px) + ", p_err" + str(p_err) + ", seed" + str(
                            seed_run) + ", correction from sheets is not valid")
                    fail = 2
                else:
                    # verify that remaining flips of noise+recovery commute with all the logicals
                    logical_err = check_any_logical_error(logicals, numqubits, list(rem_sheets))
                    if logical_err > 0:
                        fail = 1


        else:
            raise Exception("decoder_type not valid, should be sheets greedy_sheets, bp_sheets, bp or bp_osd")
        failures[i] = fail # failure value. 2 - didnt manage to decode, 1 - logical error, 0 - correct decoding

    num_fails = np.count_nonzero(failures) # total number of failures
    end_time = time.time()
    total_time = end_time - start_time

    # ---------------------saving the data --------------------
    fn_desc = 'len' + str(px) + '_perr' + str(p_err) + '_nreps' + str(num_rep) + '_seed' + str(
            seed_num) + '_dec_' + decoder_type
    fn_bp = '_bpmeth_'+ str(bp_method) + '_bpiter' + str(bp_iter)
    fn_osd = '_osdmeth_' + str(osd_method) + '_osd_order' + str(osd_order)
    if decoder_type == 'bp_osd':
        file_name = folder + fn_desc + fn_bp + fn_osd + '.txt'
    elif decoder_type == 'greedy_sheets' or decoder_type == 'sheets':
        file_name = folder + fn_desc + '.txt'
    else:
        file_name = folder + fn_desc + fn_bp + '.txt'
    data_string = 'len:'+str(px)+', p_err:'+str(p_err)+', num reps:'+str(num_rep)+', seed:'+str(seed_num)+', dec:'+decoder_type
    bp_string = 'bp_iterations:'+str(bp_iter)+', bp_method:'+str(bp_method)
    osd_string = 'osd_method:'+str(osd_method)+', osd_order:'+str(osd_order)

    with open(file_name, 'w') as f:
        # write the parameters for this run:
        f.write(data_string + '\n')
        f.write(bp_string + ', ' + osd_string + '\n')
        # write the total time:
        f.write('Total time of execution: %s seconds \n' % total_time)
        # writing the sum of error:
        f. write('The total number of samples with wrong decoding: ' + str(num_fails) + '\n')
        # for bp_sheets write time break-out
        if decoder_type == "bp_sheets":
            f.write("t_bp: %s, t_matching: %s, t_weighting: %s, t_sweeping: %s, t_organizing: %s, t_converting: %s, t_verifying: %s \n" % (time_bp, time_matching, time_weighting, time_sweeping, time_organizing, time_converting, time_verifying) )
        # write the raw results:
        for elem in failures:
            f.write(str(elem))
            f.write('\n')
    # The file is automatically closed when the 'with' block ends
    abspath = os.path.abspath(file_name)
    print("The results were saved to " + abspath)