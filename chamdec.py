import argparse
from chamdec_statistics import chamdec_exec
def main():
    parser = argparse.ArgumentParser(description="gather statistics of success rate of different decoders on the chamon code")
    parser.add_argument("-f", "--folder", help="name of folder to save results in", default="results//")
    parser.add_argument("-l", "--px", help="half the linear length of the chamon cube, the distance is d=2*px",
                        default=9, type=int)
    parser.add_argument("-p", "--p_err", help="probability of error", default=0.05, type=float)
    parser.add_argument("-n", "--num_rep", help="number of iterations", default=100, type=int)
    parser.add_argument("-s", "--seed_num", help="seed (should be integer)", default=0, type=int)
    parser.add_argument("-t", "--decoder_type", help="choose the decoder. The options are:\n"
                                                     "sheets for the basic matching decoder. \n"
                                                     "greedy_sheets for the matching decoder with initial greedy step. \n"
                                                     "bp_sheets for the matching decoder with initial BP step. \n"
                                                     "bp_osd for the BP-OSD decoder. \n"
                                                     "bp for the hard BP decoder.", default="bp_sheets")
    args = parser.parse_args()

    chamdec_exec(args.folder, args.px, args.p_err, args.num_rep, args.seed_num, args.decoder_type)


if __name__ == '__main__':
    main()

