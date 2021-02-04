import os
import argparse
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('run_file', help='The path to the file to be run.', type=str)
    parser.add_argument('--gpu', '-g', help='ID of GPU that should be used.', type=str, default="")
    parser.add_argument('--verbose', '-v', help='ID of GPU that should be used.', type=bool, default=False)

    args = parser.parse_args()

    if not args.verbose:
        tf.get_logger().setLevel('ERROR')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    exec(open(args.run_file).read())
