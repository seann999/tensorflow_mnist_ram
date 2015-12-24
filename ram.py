import tensorflow as tf
import tf_mnist_loader
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn_cell
import math

dataset = tf_mnist_loader.read_data_sets("mnist_data")
save_dir = "save-3scales/"
save_prefix = "save"
start_step = 10000
#load_path = None
load_path = save_dir + save_prefix + str(start_step) + ".ckpt"

# to enable visualization, set draw to True
eval_only = False
animate = True
draw = True

minRadius = 4 # zooms -> minRadius * 2**<depth_level>
sensorBandwidth = 8 # fixed resolution of sensor
sensorArea = sensorBandwidth**2
depth = 3 # zooms
channels = 1 # grayscale
totalSensorBandwidth = depth * sensorBandwidth * sensorBandwidth * channels
batch_size = 10

hg_size = 128
hl_size = 128
g_size = 256
cell_size = 256
cell_out_size = cell_size

glimpses = 6
n_classes = 10

lr = 1e-3
max_iters = 1000000

mnist_size = 28

loc_sd = 0.1
mean_locs = []
sampled_locs = [] # ~N(mean_locs[.], loc_sd)
glimpse_images = [] # to show in window

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0/shape[0]) # for now
    return tf.Variable(initial)

def glimpseSensor(img, normLoc):
    loc = ((normLoc + 1) / 2) * mnist_size # normLoc coordinates are between -1 and 1
    loc = tf.cast(loc, tf.int32)

    img = tf.reshape(img, (batch_size, mnist_size, mnist_size, channels))

    zooms = []
    
    # process each image individually
    for k in xrange(batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        max_radius = minRadius * (2 ** (depth - 1)) 
        offset = max_radius
    
        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
            max_radius * 2 + mnist_size, max_radius * 2 + mnist_size)
        
        for i in xrange(depth):
            r = int(minRadius * (2 ** (i - 1)))

            d_raw = 2 * r
            d = tf.constant(d_raw, shape=[1])

            d = tf.tile(d, [2])
            
            loc_k = loc[k,:]
            adjusted_loc = offset + loc_k - r
            
            
            one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value,\
                one_img.get_shape()[1].value))
                
            # crop image to (d x d)
            zoom = tf.slice(one_img2, adjusted_loc, d)
            
            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
            zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
            imgZooms.append(zoom)
    
        zooms.append(tf.pack(imgZooms))
        
    zooms = tf.pack(zooms)
    
    glimpse_images.append(zooms)
    
    return zooms

def get_glimpse(loc):
    glimpse_input = glimpseSensor(inputs_placeholder, loc)
    
    glimpse_input = tf.reshape(glimpse_input, (batch_size, totalSensorBandwidth))
    
    l_hl = weight_variable((2, hl_size))
    glimpse_hg = weight_variable((totalSensorBandwidth, hg_size))
    
    hg = tf.nn.relu(tf.matmul(glimpse_input, glimpse_hg))
    hl = tf.nn.relu(tf.matmul(loc, l_hl))
    
    hg_g = weight_variable((hg_size, g_size))
    hl_g = weight_variable((hl_size, g_size))
    
    g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))
    
    return g
    
def get_next_input(output, i):
    mean_loc = tf.tanh(tf.matmul(output, h_l_out))
    mean_locs.append(mean_loc)
    
    sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)
    
    sampled_locs.append(sample_loc)
    
    return get_glimpse(sample_loc)
    
def model():
    initial_loc = tf.random_uniform((batch_size, 2), minval=-1, maxval=1)

    initial_glimpse = get_glimpse(initial_loc)   
    
    lstm_cell = rnn_cell.LSTMCell(cell_size, g_size, num_proj=cell_out_size)

    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    
    inputs = [initial_glimpse]
    inputs.extend([0] * (glimpses - 1))
    
    outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=get_next_input)
    get_next_input(outputs[-1], 0)
            
    return outputs
    
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  # copied from TensorFlow tutorial
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * n_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
  
# to use for maximum likelihood with glimpse location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * math.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)
    
def calc_reward(outputs):
    outputs = outputs[-1] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (batch_size, cell_out_size))
    h_a_out = weight_variable((cell_out_size, n_classes))

    p_y = tf.nn.softmax(tf.matmul(outputs, h_a_out))
    max_p_y = tf.arg_max(p_y, 1)
    correct_y = tf.cast(labels_placeholder, tf.int64)

    R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32) # reward per example

    reward = tf.reduce_mean(R) # overall reward
    
    p_loc = gaussian_pdf(mean_locs, sampled_locs)
    p_loc = tf.reshape(p_loc, (batch_size, glimpses * 2))

    R = tf.reshape(R, (batch_size, 1))
    J = tf.concat(1, [tf.log(p_y + 1e-5) * onehot_labels_placeholder, tf.log(p_loc + 1e-5) * R])
    J = tf.reduce_sum(J, 1)
    J = tf.reduce_mean(J, 0)
    cost = -J
    
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(cost)

    return cost, reward, max_p_y, correct_y, train_op

with tf.Graph().as_default():
    labels = tf.placeholder("float32", shape=[batch_size, n_classes])
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 28 * 28), name="images")
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name="labels")
    onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 10), name="oneHotLabels")
    
    h_l_out = weight_variable((cell_out_size, 2))
    loc_mean = weight_variable((batch_size, glimpses, 2))
    
    outputs = model()
    
    # convert list of tensors to one big tensor
    sampled_locs = tf.concat(0, sampled_locs)
    sampled_locs = tf.reshape(sampled_locs, (batch_size, glimpses, 2))
    mean_locs = tf.concat(0, mean_locs)
    mean_locs = tf.reshape(mean_locs, (batch_size, glimpses, 2))
    glimpse_images = tf.concat(0, glimpse_images)
    
    cost, reward, predicted_labels, correct_labels, train_op = calc_reward(outputs)
    tf.scalar_summary("reward", reward)
    tf.scalar_summary("cost", cost)
    
    summary_op = tf.merge_all_summaries()
    
    sess = tf.Session()
    saver = tf.train.Saver()
    
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if load_path is not None and ckpt and ckpt.model_checkpoint_path:
        try:
            saver.restore(sess, load_path)
            print("LOADED CHECKPOINT")
        except:
            print("FAILED TO LOAD CHECKPOINT")
            exit()
    else:
        init = tf.initialize_all_variables()
        sess.run(init)
        
    def evaluate():
        data = dataset.test
        batches_in_epoch = len(data._images) // batch_size
        accuracy = 0
            
        for i in xrange(batches_in_epoch):
            nextX, nextY = dataset.test.next_batch(batch_size)
            feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY, onehot_labels_placeholder: dense_to_one_hot(nextY)}          
            r = sess.run(reward, feed_dict=feed_dict)
            accuracy += r

        accuracy /= batches_in_epoch

        print("ACCURACY: " + str(accuracy))  

    if eval_only:        
        evaluate()
    else:
        summary_writer = tf.train.SummaryWriter("summary", graph_def=sess.graph_def)
        
        if draw:
            fig = plt.figure()
            txt = fig.suptitle("-", fontsize=36, fontweight='bold') 
            plt.ion()
            plt.show()   
            plt.subplots_adjust(top=0.7)
        
            plotImgs = []
        
        for step in xrange(start_step + 1, max_iters):
            start_time = time.time()
            
            nextX, nextY = dataset.train.next_batch(batch_size)
            feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY, onehot_labels_placeholder: dense_to_one_hot(nextY)}
            fetches = [train_op, cost, reward, predicted_labels, correct_labels, glimpse_images]
            
            results = sess.run(fetches, feed_dict=feed_dict)
            _, cost_fetched, reward_fetched, prediction_labels_fetched,\
                correct_labels_fetched, f_glimpse_images_fetched = results
            
            duration = time.time() - start_time
            
            if step % 20 == 0:
                if step % 1000 == 0:
                    saver.save(sess, save_dir + save_prefix + str(step) + ".ckpt")
                    if step % 5000 == 0:
                        evaluate()
                    
    
                ##### DRAW WINDOW ################
    
                f_glimpse_images = np.reshape(f_glimpse_images_fetched, (glimpses + 1, batch_size, depth, sensorBandwidth, sensorBandwidth)) #steps, THEN batch
                
                if draw:
                    if animate:
                        fillList = False
                        if len(plotImgs) == 0:
                            fillList = True
                        
                        # display first in mini-batch
                        for y in xrange(glimpses):
                            txt.set_text('FINAL PREDICTION: %i\nTRUTH: %i\nSTEP: %i/%i'
                                % (prediction_labels_fetched[0], correct_labels_fetched[0], (y + 1), glimpses))
                            
                            for x in xrange(depth):
                                plt.subplot(depth, 1, x + 1)
                                if fillList:
                                    plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                                         interpolation="nearest")
                                    plotImg.autoscale()                                
                                    plotImgs.append(plotImg)
                                else:
                                    plotImgs[x].set_data(f_glimpse_images[y, 0, x])
                                    plotImgs[x].autoscale()  
                                    
                            fillList = False
                            
                            fig.canvas.draw()
                            time.sleep(0.1)
                            plt.pause(0.0001) 
                    else:
                        txt.set_text('PREDICTION: %i\nTRUTH: %i' % (prediction_labels_fetched[0], correct_labels_fetched[0]))  
                        for x in xrange(depth):
                            for y in xrange(glimpses):
                                plt.subplot(depth, glimpses, x * glimpses + y + 1)
                                plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                           interpolation="nearest")
                        
                        plt.draw()
                        time.sleep(0.05)
                        plt.pause(0.0001)  
                        
                ################################
                
                print('Step %d: cost = %.5f reward = %.5f (%.3f sec)' % (step, cost_fetched, reward_fetched, duration))
                
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            
sess.close()
