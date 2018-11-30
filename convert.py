import numpy as np
import sys, os
import argparse
import pdb

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--caffe_root', type=str, required=True)
	parser.add_argument('--caffemodel', type=str, required=True)
	parser.add_argument('--prototxt', type=str, required=True)
	return parser

args = make_parser().parse_args()

# Edit the paths as needed:
sys.path.insert(0, args.caffe_root + 'python')

import caffe

net = caffe.Net(args.prototxt, args.caffemodel, caffe.TRAIN)


all_names = [n for n in net._layer_names]
pdb.set_trace()
print all_names

# # For each of the pretrained net sides, copy the params to
# # the corresponding layer of the combined net:
# for side, net in nets.items():
#     for layer in layer_names:
#         W = net.params[layer][0].data[...] # Grab the pretrained weights
#         b = net.params[layer][1].data[...] # Grab the pretrained bias
#         comb_net.params['{}_{}'.format(side, layer)][0].data[...] = W # Insert into new combined net
#         comb_net.params['{}_{}'.format(side, layer)][1].data[...] = b 

# # Save the combined model with pretrained weights to a caffemodel file:
# comb_net.save('pretrained.caffemodel')