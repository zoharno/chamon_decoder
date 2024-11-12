This project simulates and checks success rates of a few different decoders on the cubic Chamon code.<br>
This code was written as part of the research performed by Zohar Schwartzman-Nowik and Benjamin Brown, 
presented in a paper titled "Generalizing the matching decoder for the Chamon code", http://arxiv.org/abs/2411.03443. <br>
<p>The decoders introduced in this paper are generalizations of matching decoders on subsets of the stabilizers of the code, termed symmetries.
There are three versions of the decoder - the basic implementation of the matching decoder, a version with an initial greedy step and a version with an initial belief propagation step.
The project also simulates, for comparison, a BP-OSD decoder.</p>
<pre>
usage: chamdec.py [-h] [-f FOLDER] [-l PX] [-p P_ERR] [-n NUM_REP] [-s SEED_NUM] [-t DECODER_TYPE]

gather statistics of success rate of different decoders on the chamon code

options:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        name of folder to save results in
  -l PX, --px PX        half the linear length of the chamon cube, the distance is d=2*px
  -p P_ERR, --p_err P_ERR
                        probability of error
  -n NUM_REP, --num_rep NUM_REP
                        number of iterations
  -s SEED_NUM, --seed_num SEED_NUM
                        seed (should be integer)
  -t DECODER_TYPE, --decoder_type DECODER_TYPE
                        choose the decoder. The options are: sheets for the basic matching decoder. greedy_sheets for
                        the matching decoder with initial greedy step. bp_sheets for the matching decoder with initial
                        BP step. bp_osd for the BP-OSD decoder. bp for the hard BP decoder.
</pre>
