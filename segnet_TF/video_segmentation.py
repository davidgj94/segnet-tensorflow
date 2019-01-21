import tensorflow as tf
import flags; FLAGS = tf.app.flags.FLAGS
import numpy as np
from create_TFRecords import read_records
from model import segnet_extended
from score import compute_hist, print_hist_summary
import pdb
import cv2
from downscale import _downscale as downscale
from downscale import get_padding
import vis
import os
import os.path

palette = np.array([[255,255,255],[0, 0, 255],[0, 255, 0]])

_FISHESYE_METHOD_DATA = 'datos_calib/fisheye'
_STANDARD_METHOD_DATA = 'datos_calib/standard'

_D = np.load(os.path.join(_FISHESYE_METHOD_DATA,'D.npy'))
_K = np.load(os.path.join(_FISHESYE_METHOD_DATA,'K.npy'))
_dist = np.load(os.path.join(_STANDARD_METHOD_DATA,'dist.npy'))
_mtx = np.load(os.path.join(_STANDARD_METHOD_DATA,'mtx.npy'))


def undistort(img, use_fisheye_method=False, TOTAL=True, is_mask=False):

    h, w = img.shape[:2]

    if use_fisheye_method:

        if TOTAL:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(_K, _D, (w,h), np.eye(3), balance=1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(_K, _D, np.eye(3), new_K, (w,h), cv2.CV_16SC2)
        else:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(_K, _D, np.eye(3), _K, (w,h), cv2.CV_16SC2)

        dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    else:

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(_mtx, _dist,(w,h),0,(w,h))
        dst = cv2.undistort(img, _mtx, _dist, None, newcameramtx)

    return dst

def _video_segmentation(vidcap, out, delay_msec, x_pad, y_pad):

    with tf.Graph().as_default():

        image  = tf.placeholder(tf.float32, shape=[1, 544, 1024, 3], name="input")
        logits = segnet_extended(image)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        with tf.Session() as sess:

            sess.run(init_op)

            success = True
            time_msec = 0
            count = 0

            while success:
    
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(time_msec))
                success, img = vidcap.read()

                if success:

                    img_undist = undistort(img)
                    img_down = downscale(img_undist, FLAGS.max_dim, False)
                    img_down = img_down.astype(np.float32)
                    img_down = img_down[np.newaxis,:]
                    _logits = sess.run(logits, feed_dict={image: img_down})
                    mask = np.argmax(np.squeeze(_logits), axis=-1)

                    mask = mask[y_pad[0]:-y_pad[-1], x_pad[0]:-x_pad[1]]
                    img_down = np.squeeze(img_down)
                    img_down = img_down[y_pad[0]:-y_pad[-1], x_pad[0]:-x_pad[1]]

                    if FLAGS.save_mask:
                        cv2.imwrite(os.path.join(FLAGS.save_mask_dir, 'frame_{}.png'.format(count)), mask)
                        cv2.imwrite(os.path.join(FLAGS.save_img_dir, 'frame_{}.png'.format(count)), img_down)
                    else:
                        vis_img = vis.vis_seg(img_down[...,::-1], mask, palette)
                        out.write(vis_img[...,::-1])

                    print count
                    time_msec += delay_msec
                    count += 1

                else:
                    out.release()

def main(args):

    vidcap = cv2.VideoCapture(FLAGS.video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, img = vidcap.read()
    img_undist = undistort(img)
    img_aux = downscale(img_undist, FLAGS.max_dim, False, pad_img=False)
    height, width = img_aux.shape[:2]
    x_pad = get_padding(width)
    y_pad = get_padding(height)
    out = cv2.VideoWriter(FLAGS.save_path_video, cv2.VideoWriter_fourcc(*'XVID'), FLAGS.fps, (width, height))
    delay_msec = int(1000 * (1 / FLAGS.fps))

    _video_segmentation(vidcap, out, delay_msec, x_pad, y_pad)

if __name__ == "__main__":
    tf.app.run() # wrapper that handles flags parsing.