import sys
import argparse
import pdb
import pickle
import numpy as np

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--caffe_root', type=str, required=True)
	parser.add_argument('--caffemodel', type=str, required=True)
	parser.add_argument('--prototxt', type=str, required=True)
	parser.add_argument('--save_path', type=str, required=True)
	return parser

args = make_parser().parse_args()

# Edit the paths as needed:
sys.path.insert(0, args.caffe_root + 'python')

import caffe

net = caffe.Net(args.prototxt, args.caffemodel, caffe.TRAIN)

segnet_params = dict()

for layer_name in net.params.keys():
	segnet_params[layer_name] = []
	for i in range(len(net.params[layer_name])):
		segnet_params[layer_name].append(net.params[layer_name][i].data)

with open(args.save_path, 'wb') as handle:
    pickle.dump(segnet_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(args.save_path, 'rb') as handle:
    _segnet_params = pickle.load(handle)

if np.array_equal(_segnet_params['conv1_1_bn'][0],segnet_params['conv1_1_bn'][0]):
	print "Fin"

print net.params.keys()
pdb.set_trace()
print "asdf"