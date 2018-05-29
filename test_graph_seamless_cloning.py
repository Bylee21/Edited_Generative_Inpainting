import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
# go through all files in the folder
import os
import datetime
import seamless_cloning
import time


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./test_dir/GOPR47/input/raindrop0121.jpg', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='./test_dir/GOPR47/mask/raindrop0121.jpg', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='./test_dir/GOPR47/output_test_raindrop0121.jpg', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='./model_logs/model_40000', type=str,
                    help='The directory of tensorflow checkpoint.')

## add more options                    
parser.add_argument('--test_dir', default='./test_dir/GOPR0076', type=str,
                    help='The directory of test images and masks.')   
parser.add_argument('--image_width', default=640, type=int,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--image_height', default=640, type=int,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--flist', default="flist.flist", type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--is_same_mask', default=1, type=int,
                    help='The directory of tensorflow checkpoint.')



if __name__ == "__main__":
    ## ng.get_gpus(1)
    args = parser.parse_args()

    input_folder = args.test_dir + "/input"
    mask_folder = args.test_dir + "/mask"

    output_folder_main = args.test_dir + "/output_" + args.checkpoint_dir.split("/")[1] + "_" +datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # seamless_folder = args.test_dir + "/seamless_" + args.checkpoint_dir.split("/")[1] + "_" +datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    seamless_folder = output_folder_main

    output_folder = seamless_folder + "/output"
    seamless_folder_mixed = seamless_folder + "/mixed"
    seamless_folder_normal = seamless_folder + "/normal"
    seamless_folder_monochrome = seamless_folder + "/monochrome" 
    
    # make dir if not existed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(seamless_folder):
        os.makedirs(seamless_folder)
    # inside seamless_folder    
    if not os.path.exists(seamless_folder_mixed):
        os.makedirs(seamless_folder_mixed)
    if not os.path.exists(seamless_folder_normal):
        os.makedirs(seamless_folder_normal)
    if not os.path.exists(seamless_folder_monochrome):
        os.makedirs(seamless_folder_monochrome)
    #####################################
    sess_config = tf.ConfigProto()                                                                                                                                                                                                            
    sess_config.gpu_options.allow_growth = True                                                                                                                                                                                               
    sess = tf.Session(config=sess_config)                                                                                                                                                                                                     
                                                                                                                                                                                                                                              
    model = InpaintCAModel()                                                                                                                                                                                                                  
    input_image_ph = tf.placeholder(                                                                                                                                                                                                          
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))                                                                                                                                                                      
    output = model.build_server_graph(input_image_ph)                                                                                                                                                                                         
    output = (output + 1.) * 127.5                                                                                                                                                                                                            
    output = tf.reverse(output, [-1])                                                                                                                                                                                                         
    output = tf.saturate_cast(output, tf.uint8)                                                                                                                                                                                               
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)                                                                                                                                                                              
    assign_ops = []                                                                                                                                                                                                                           
    for var in vars_list:                                                                                                                                                                                                                     
        vname = var.name                                                                                                                                                                                                                      
        from_name = vname                                                                                                                                                                                                                     
        var_value = tf.contrib.framework.load_variable(                                                                                                                                                                                       
            args.checkpoint_dir, from_name)                                                                                                                                                                                                   
        assign_ops.append(tf.assign(var, var_value))                                                                                                                                                                                          
    sess.run(assign_ops)                                                                                                                                                                                                                      
    print('Model loaded.')                                                                                                                                                                                                                    
                                                                                                                                                                                                                                              

    dir_files = os.listdir(input_folder)
    dir_files.sort()

    # t = time.time()                                                                                                                                                                                                                           
    for file_inter in dir_files:   
        
        # count time
        start_time = time.time()

        base_file_name = os.path.basename(file_inter)            

        image = cv2.imread(input_folder + "/" + base_file_name)
        if args.is_same_mask !=1:
            mask = cv2.imread(mask_folder + "/" + base_file_name)
        else:
            mask = cv2.imread(mask_folder + "/" + "mask.jpg")
                                                                                                                                                                                           
                                                                                                                                                                                                                                              
        assert image.shape == mask.shape                                                                                                                                                                                                      
                                                                                                                                                                                                                                              
        h, w, _ = image.shape                                                                                                                                                                                                                 
        grid = 4                                                                                                                                                                                                                              
        image = image[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                        
        mask = mask[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                          
        print('Shape of image: {}'.format(image.shape))                                                                                                                                                                                       
                                                                                                                                                                                                                                              
        image = np.expand_dims(image, 0)                                                                                                                                                                                                      
        mask = np.expand_dims(mask, 0)                                                                                                                                                                                                        
        input_image = np.concatenate([image, mask], axis=2)                                                                                                                                                                                   
                                                                                                                                                                                                                                              
        # load pretrained model                                                                                                                                                                                                               
        result = sess.run(output, feed_dict={input_image_ph: input_image})                                                                                                                                                                    
        print('Processed: ', output_folder + "/" + base_file_name)   


        # write to output folder
        cv2.imwrite(output_folder + "/" + base_file_name, result[0][:, :, ::-1])

        # post-prosessing with seamless cloning
        image_s = cv2.imread(input_folder + "/" + base_file_name) # input

        if args.is_same_mask !=1:
            mask_s = cv2.imread(mask_folder + "/" + base_file_name)
        else:
            mask_s = cv2.imread(mask_folder + "/" + "mask.jpg")

        obj = cv2.imread(output_folder + "/" + base_file_name) # output

        # write
        dest_cloned_mixed = seamless_cloning.seamless_cloning(obj, image_s,  mask_s, 'mixed_clone')
        dest_cloned_normal = seamless_cloning.seamless_cloning( obj,image_s, mask_s, 'normal_clone')
        dest_cloned_monochrome = seamless_cloning.seamless_cloning(obj,image_s,  mask_s, 'monochrome_transfer mode')

        cv2.imwrite(seamless_folder_mixed + "/" + base_file_name, dest_cloned_mixed)
        cv2.imwrite(seamless_folder_normal + "/" + base_file_name, dest_cloned_normal)
        cv2.imwrite(seamless_folder_monochrome + "/" + base_file_name, dest_cloned_monochrome)

        # count time
        print("-- Processing time(seconds) for: ", input_folder + "/" + base_file_name, " is: ", (time.time() - start_time))


                                                                                                                                                                                                                                              
    # print('Time total: {}'.format(time.time() - t)) 