from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('--model_dir',
                        help='Path to frozen graph file with a trained model.',
                        required=True,
                        type=str)
    parser.add_argument('input_image', help='input image ')
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], args.model_dir)
        graph = tf.get_default_graph()
        img = cv2.imread(args.input_image,cv2.IMREAD_GRAYSCALE)
        assert img.shape == (28,28)
        input = np.expand_dims(img,axis=0)
        x = sess.graph.get_tensor_by_name('Placeholder:0')
        y = sess.graph.get_tensor_by_name('Softmax:0')
        scores = sess.run(y,
                          feed_dict={x: input})
        print("predict num: %d" % (np.argmax(scores, 1),))