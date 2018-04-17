import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

# go through all files in the folder
import os
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='examples/places2/wooden_input.png', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='examples/places2/wooden_mask.png', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='examples/output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='model_logs/release_places2_256', type=str,
                    help='The directory of tensorflow checkpoint.')
## add more options                    
parser.add_argument('--image_dir', default='', type=str,
                    help='The directory of input images.')
parser.add_argument('--mask_dir', default='', type=str,
                    help='The directory of input images masks.')
parser.add_argument('--output_dir', default='', type=str,
                    help='The directory of output images.')    

## add more options                    
parser.add_argument('--test_dir', default='test_dir/GOPR47', type=str,
                    help='The directory of test images and masks.')   


if __name__ == "__main__":
    ## ng.get_gpus(1)
    args = parser.parse_args()

    model = InpaintCAModel()
    # image = cv2.imread(args.image)
    # mask = cv2.imread(args.mask)

    # assert image.shape == mask.shape

    # h, w, _ = image.shape
    # grid = 8
    # image = image[:h//grid*grid, :w//grid*grid, :]
    # mask = mask[:h//grid*grid, :w//grid*grid, :]
    # print('Shape of image: {}'.format(image.shape))

    # image = np.expand_dims(image, 0)
    # mask = np.expand_dims(mask, 0)
    # input_image = np.concatenate([image, mask], axis=2)


    # prepare folder path
    input_folder = args.test_dir + "/input"
    mask_folder = args.test_dir + "/mask"
    output_folder = args.test_dir + "/output_" + args.checkpoint_dir.split("/")[1] + "_" +datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # start sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    


    dir_files = os.listdir(input_folder)
    dir_files.sort()

    for file_inter in dir_files:
        sess = tf.Session(config=sess_config)
        
        base_file_name = os.path.basename(file_inter)

        image = cv2.imread(input_folder + "/" + base_file_name)
        mask = cv2.imread(mask_folder + "/" + "mask.jpg")

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 1
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image, reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)

        # write to output folder
        cv2.imwrite(output_folder + "/" + base_file_name, result[0][:, :, ::-1])
        sess.close()
        

        
