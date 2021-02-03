import os 
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import argparse
import random

from models import fmnet_model

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

flags = tf.app.flags
FLAGS = flags.FLAGS


train_subjects = list(range(0,50)) 

# ====================================================================================== 
#
# Parse Inputs
#
# ======================================================================================
parser = argparse.ArgumentParser(
    description = 'Input the path directory of input data files')
parser.add_argument('--data_path', type=str,
                        default = './Data/OAI_ZIB_FMnet/femoral_cartilage', help='Directory with data input data files.')
args = parser.parse_args()

evecs_dir = args.data_path + '/evecs'
shot_dir = args.data_path + '/shot'
dist_maps_dir = args.data_path + '/distance_maps'
ground_truth_dir = args.data_path + '/ground_truth'
log_dir = args.data_path.replace('Data/OAI_ZIB_FMnet','Results/train_knee_flags')

# training params
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate.')
flags.DEFINE_integer('batch_size', 30, 'batch size.') 

# architecture parameters
flags.DEFINE_integer('num_layers', 7, 'network depth') # 7 originally
flags.DEFINE_integer('num_evecs', 35,
					 'number of eigenvectors used for representation.') # 120 originally
flags.DEFINE_integer('dim_shot', 352, '')

# Run.Session params
flags.DEFINE_integer('max_train_iter', 5000, '') 
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_model_secs', 300, '') # 10 minutes
flags.DEFINE_string('master', '', '') 


# ================================================================================
#
# Training and Validation Set 
#
# ================================================================================


ref_labels_fname = [e for e in os.listdir(ground_truth_dir) if e.endswith('.mat')]
dist_maps_fname = [e for e in os.listdir(dist_maps_dir) if e.endswith('.mat')]
evecs_file = [e for e in os.listdir(evecs_dir) if e.endswith('.mat')]
shot_file = [e for e in os.listdir(shot_dir) if e.endswith('.mat')]

# ====================================================================================== 
#
# Definition of functions 
#
# ======================================================================================


def load_dist_maps():
	'''
	Load the distance maps of each shape in RAM. 
	=====================================
	Returns: (global)
		dist_maps[i] = NxN matrix, where i the ID of the mesh	
	'''

	print('loading dist maps...')	
	global dist_maps 
	dist_maps = {}
		
	for i_subject in train_subjects:	 
				
		# Load .mat file with D (NxN) of each mesh 
		d = sio.loadmat(dist_maps_dir + '/' + dist_maps_fname[i_subject]) 		
		print('Loaded distance map: ' + dist_maps_dir + '/' + dist_maps_fname[i_subject])
		
		# Save to global dist_maps
		dist_maps[i_subject] = d['D']


def load_models_to_ram():
	'''
	Loads the input data (eigenfuntions, descriptor) of each subject 
	in RAM. The input data contain the: 
	 - eigenvectors (Phi), 
	 - transposed eigenvectors (Phi_T) 
	 - transposed eigenvectors with mass matrix correction (Phi_T * M) 
	=================================================================
	Returns: (global)
		model_train[i]  : input data of training set
		model_val[i]  : input data of validation set
	'''

	# Global variables to save input data
	global models_train
	models_train = {}
	global models_val
	models_val = {}
	
	for i_subject in train_subjects:			
		print('Loaded training model: ' + evecs_dir + '/' + evecs_file[i_subject])
		
		# Truncated eigenfunctions (keep only k first) in Training set
		input_data = sio.loadmat(evecs_dir + '/' + evecs_file[i_subject])
		shot_data = sio.loadmat(shot_dir + '/' + shot_file[i_subject])

		input_data['model_evecs'] = input_data['evecs'][:, 0:FLAGS.num_evecs]
		# input_data['model_evals'] = input_data['evals'][0:FLAGS.num_evecs, 0:FLAGS.num_evecs]
		input_data['model_evecs_trans'] = input_data['evecs_trans'][0:FLAGS.num_evecs, :]
		input_data['model_shot'] = shot_data['shot']
		models_train[i_subject] = input_data


def get_pair_from_ram(i_model, i_part, dataset):
	'''
	Gets a pair of meshes loaded in RAM.
		- Part  (source) = X -> Phi evecs
		- Model (target) = Y -> Psi evecs
	============================================
	Returns: 
		- input_data: contains the PART and MODEL input data (evecs, descriptors)
		- dist_maps[model] : Dy 
	'''
	
	input_data = {}

	if dataset == 'train':
		input_data['part_evecs'] = models_train[i_part]['model_evecs']
		# input_data['part_evals'] = models_train[i_part]['model_evals']
		input_data['part_evecs_trans'] = models_train[i_part]['model_evecs_trans']
		input_data['part_shot'] = models_train[i_part]['model_shot']
		input_data.update(models_train[i_model])
	else:
		input_data['part_evecs'] = models_val[i_part]['model_evecs']
		# input_data['part_evals'] = models_val[i_part]['model_evals']
		input_data['part_evecs_trans'] = models_val[i_part]['model_evecs_trans']
		input_data['part_shot'] = models_val[i_part]['model_shot']
		input_data.update(models_val[i_model])

	return input_data, dist_maps[i_model]


def get_input_pair(batch_size=1, num_vertices=500): 
	'''
	Create mini-batches the are fed to th FMnet.
	Minibatches are training pairs (batch_size X 1 pair)
	created from randomly chosen common points. 
		Example: for batch_size = 8 we have 8 pairs of shapes
		(16 total shapes) formed by 1000 randomly chosen points
		common to each pair.
	============================================================
	Returns:
		 - batch_input[i_batch] : input data of each pair in batch_size
		 - batch_dist[i_batch] : dist_map of Y patch in each pair in batch_size
	'''

	dataset = 'train' # -- TODO Add validation in session.run 
	
	# --------------
	# Initialization
	# --------------
	batch_input = {'part_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
                'model_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
                # 'model_evals': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_evecs)),
                # 'part_evals': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_evecs)),
                'part_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
                'model_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
                'part_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),
                'model_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot))
                }
	batch_dist = np.zeros((batch_size, num_vertices, num_vertices))

	# -------------------
	# Create mini-batches 
	# -------------------
	for i_batch in range(batch_size):

		# Select a random pair from the dataset
		i_model = np.random.choice(train_subjects) 
		i_part = np.random.choice(train_subjects) 
		while i_part == i_model:
			i_part = np.random.choice(train_subjects) 
		
		# Fetch X and Y input data and dist_map of Y
		batch_input_, batch_dist_ = get_pair_from_ram(i_model, i_part, dataset) 


		gt_labels = sio.loadmat(ground_truth_dir + '/' + ref_labels_fname[i_model]) 
		batch_input_['model_KAB'] = gt_labels['labels'][ref_labels_fname[i_part].replace('.mat','')][0].tolist()[0]
		batch_input_['part_IB'] = gt_labels['labels'][ref_labels_fname[i_model].replace('.mat','')][0].tolist()[0]


		# N = np.shape(batch_input_['part_evecs'])[0]
		# joint_labels = np.random.permutation(range(N))[:num_vertices]
		idx_part = range(np.shape(batch_input_['part_evecs'])[0]) # 1:N
		idx_model= range(np.shape(batch_input_['model_evecs'])[0]) # 1:M
		joint_labels = np.intersect1d(idx_part, idx_model) 
		joint_labels = np.random.permutation(joint_labels)[:num_vertices]  # type(joint_labels) = array

	
		ind_dict_part = [(ind, value.tolist()[0]-1) for ind, value in enumerate(batch_input_['part_IB'])]
		ind_part = [ind_dict_part[x][1] for x in joint_labels]
		ind_dict_model = [(ind, value.tolist()[0]-1) for ind, value in enumerate(batch_input_['model_KAB'])]
		ind_model = [ind_dict_model[x][1] for x in joint_labels]

		# Distance map of Y patch
		batch_dist[i_batch] = batch_dist_[ind_model, :][:, ind_model] 

		# mini-batch pairs 	
		batch_input['part_evecs'][i_batch] = batch_input_['part_evecs'][ind_part, :]
		batch_input['part_evecs_trans'][i_batch] = batch_input_['part_evecs_trans'][:, ind_part]
		batch_input['part_shot'][i_batch] = batch_input_['part_shot'][ind_part, :]
		# batch_input['part_evals'][i_batch] = batch_input_['part_evals']
		
		batch_input['model_evecs'][i_batch] = batch_input_['model_evecs'][ind_model, :]
		batch_input['model_evecs_trans'][i_batch] = batch_input_['model_evecs_trans'][:, ind_model]
		batch_input['model_shot'][i_batch] = batch_input_['model_shot'][ind_model, :]
		# batch_input['model_evals'][i_batch] = batch_input_['model_evals']

	return batch_input, batch_dist


def run_training():
	'''
	Training phase!	

	'''
	
	# Create log_dir directory 
	print('log_dir=%s' % log_dir)
	if not os.path.isdir(log_dir):
		os.makedirs(log_dir)  

	print('Building graph...')	
	with tf.Graph().as_default():

		# Set placeholders for inputs
		part_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='part_shot')
		model_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='model_shot')
		dist_map = tf.placeholder(tf.float32, shape=(None, None, None), name='dist_map')
		part_evecs = tf.placeholder(tf.float32, shape= (None, None, FLAGS.num_evecs), name='part_evecs')
		# part_evals = tf.placeholder(tf.float32, shape= (None, FLAGS.num_evecs, FLAGS.num_evecs), name='part_evals')
		part_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='part_evecs_trans')
		model_evecs = tf.placeholder(tf.float32, shape= (None, None, FLAGS.num_evecs), name='model_evecs')
		# model_evals = tf.placeholder(tf.float32, shape= (None, FLAGS.num_evecs, FLAGS.num_evecs), name='model_evals')
		model_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='model_evecs_trans')
		phase = tf.placeholder(dtype=tf.bool, name='phase')

		# FM Net Model 
		net_loss, safeguard_inverse, merged, P_norm, average_error, net = fmnet_model(phase, part_shot, model_shot,
																						dist_map, part_evecs,
																						part_evecs_trans, 
																						model_evecs,model_evecs_trans
																						)

		# Summary 																	  
		summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs)) 
		# Global step
		global_step = tf.Variable(0, name='global_step', trainable=False)
		# Adam algorithm
		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,name='Adam')
		# Loss function to minimize 
		train_op = optimizer.minimize(net_loss, global_step=global_step)
		# Save model while training
		saver = tf.train.Saver(max_to_keep=100)
		# Supervisor --TODO Depricated. needs to be replaced. Use tf.train.MonitoredTrainingSession
		sv = tf.train.Supervisor(logdir=log_dir,
									init_op=tf.global_variables_initializer(),
									local_init_op=tf.local_variables_initializer(),
									global_step=global_step,
									save_summaries_secs=FLAGS.save_summaries_secs,
									save_model_secs=FLAGS.save_model_secs,
									summary_op=None,
									saver=saver)

		# Writer
		writer = sv.summary_writer
		
		# START SESSION
		print('starting session...')
		iteration = 0
		with sv.managed_session(master=FLAGS.master) as sess:
			
			# Load data to RAM 	
			print('loading data to ram...')
			load_models_to_ram()
			load_dist_maps()

			# TRAINING LOOP
			print('starting training loop...')
			while not sv.should_stop() and iteration < FLAGS.max_train_iter:
				iteration += 1
				start_time = time.time()

				# Create Batch Input Pairs
				input_data, mstar = get_input_pair(FLAGS.batch_size)

				# feed dictionary	
				feed_dict = {phase: True,
								part_shot: input_data['part_shot'],
								model_shot: input_data['model_shot'],
								dist_map: mstar,
								part_evecs: input_data['part_evecs'],
								part_evecs_trans: input_data['part_evecs_trans'],
								model_evecs: input_data['model_evecs'],
								model_evecs_trans: input_data['model_evecs_trans'],
								}
				
				# SESSION RUN 
				summaries, step, my_loss, safeguard, avg_error, _ = sess.run(
					[merged, global_step, net_loss, safeguard_inverse, average_error, train_op], feed_dict=feed_dict)
				
				# Add summaries to writer
				writer.add_summary(summaries, step)
				summary_ = sess.run(summary)
				writer.add_summary(summary_, step)

				# duration of each step
				duration = time.time() - start_time

				print('train - step %d: loss = %.4f (%.3f sec)  avg_err = %.4f ' % (step, my_loss, duration, avg_error))

			# Save model to log_dir
			saver.save(sess, log_dir + '/model.ckpt', global_step=step)
			# flush writer
			writer.flush()
			
			# Stop Supervisor
			sv.request_stop()
			sv.stop()


def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()
