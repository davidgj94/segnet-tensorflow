import sys
sys.path.insert(0, '../../caffe-segnet-cudnn5/python')
import caffe; caffe.set_mode_gpu()
import numpy as np
import os.path
import argparse
import pdb
from score import compute_hist
from pathlib import Path

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inference_model', type=str, required=True)
	parser.add_argument('--inference_weights', type=str, required=True)
	parser.add_argument('--test_imgs', type=str, required=True)
	parser.add_argument('--num_classes', type=int, default=3)
	return parser


if __name__ == '__main__':

	parser = make_parser()
	args = parser.parse_args()

	num_test_imgs = len(list(Path(args.test_imgs).glob('*')))
	net = caffe.Net(args.inference_model, args.inference_weights, caffe.TEST)
	hist, acc, per_class_acc, per_class_iu = compute_hist(net, num_test_imgs)

	print ">>>>>>>>>>>>>>>> Accuracy total {}".format(acc)
	print ">>>>>>>>>>>>>>>> Per-Class Accuaracy total {}".format(per_class_acc)
	print ">>>>>>>>>>>>>>>> Per-Class IU".format(per_class_iu)
