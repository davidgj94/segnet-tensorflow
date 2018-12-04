import numpy as np
import os.path
import argparse
from score import compute_hist
import pdb
import caffe; caffe.set_mode_gpu()
from pathlib import Path

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inference_model', type=str, required=True)
	parser.add_argument('--inference_weights', type=str, required=True)
	parser.add_argument('--test_imgs', type=str, required=True)
	parser.add_argument('--num_classes', type=int, required=True)
	return parser


if __name__ == '__main__':

	parser = make_parser()
	args = parser.parse_args()

	num_test_imgs = len(list(Path(args.test_imgs).glob('*')))
	net = caffe.Net(args.inference_model, args.inference_weights, caffe.TEST)
	hist, acc, per_class_acc, per_class_iu = compute_hist(net, num_test_imgs)

	print hist
	print acc
	print per_class_acc
	print per_class_iu
