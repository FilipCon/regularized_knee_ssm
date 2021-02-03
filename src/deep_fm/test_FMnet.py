import os
import time
import tensorflow as tf
import scipy.io as sio
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' from tensorflow.python.client import
# device_lib

flags = tf.app.flags
FLAGS = flags.FLAGS


list_of_organs = ['femur', 'femoral_cart',
                  'tibia', 'tibial_cart_med', 'tibial_cart_lat']
dataset_path = './Data/OAI_ZIB/original/geometries/off_clean/'

N = 4

net_ids = [266, 268, 315, 264, 219]

test_set = [18]
train_set = range(507)

# =============================================================================
# Parse Inputs
# =============================================================================

parser = argparse.ArgumentParser(
    description='Input the path directory of input data files')
parser.add_argument('--part_data_path', type=str,
                    default=dataset_path + list_of_organs[N], help='')
parser.add_argument('--model_data_path', type=str,
                    default=dataset_path + list_of_organs[N], help='')
parser.add_argument('--part_id', type=int, default=None,
                    help='test part shape')
parser.add_argument('--model_id', type=int, default=None,
                    help='test model shape')
parser.add_argument('--log_id', type=int,
                    default=net_ids[N], help='trained net ID in log_dir')
parser.add_argument('--num_evecs', type=int, default=35,
                    help='number of eigenvectors used for representation.')
parser.add_argument('--log_dir', type=str,
                    default='./Results/train_knee_flags/' + list_of_organs[N], help='')
args = parser.parse_args()

model_evecs_dir = args.model_data_path + '/evecs'
model_shot_dir = args.model_data_path + '/shot'
part_evecs_dir = args.part_data_path + '/evecs'
part_shot_dir = args.part_data_path + '/shot'
# log_dir =
# args.part_data_path.replace('Data/ok_geometries','Results/train_knee_flags')
log_dir = args.log_dir

#===============================================================================
#
# Test Create Test Set
#
#===============================================================================

input_part_files = [e for e in os.listdir(
    part_evecs_dir) if e.endswith('.mat')]
input_model_files = [e for e in os.listdir(
    model_evecs_dir) if e.endswith('.mat')]

test_list = []
if args.part_id or args.model_id is not None:
    test_subjects = [args.part_id, args.model_id]  # 1:part, 2:model
    test_list = [[input_part_files[test_subjects[0]],
                  input_model_files[test_subjects[1]]]]
else:
    for model_id in train_set:
        for part_id in test_set:
            test_list.append([input_part_files[part_id],
                              input_model_files[model_id]])

# ==============================================================================
#
# Definition of functions
#
# ==============================================================================

def get_test_pair(part_fname, model_fname):
    '''
    Load a test pair from directory. The function loads .mat files containing
    the evecs and descriptors of the two shapes
    ==============================================================
    Returns: - input_data: contains Eigenvectors and Descriptor of PART and
        MODEL
    '''
    # init
    input_data = {}

    # Get file name
    part_evecs_file = '%s/%s' % (part_evecs_dir, part_fname)
    part_shot_file = '%s/%s' % (part_shot_dir, part_fname)
    model_evecs_file = '%s/%s' % (model_evecs_dir, model_fname)
    model_shot_file = '%s/%s' % (model_shot_dir, model_fname)

    part_shot_data = sio.loadmat(part_shot_file)
    model_shot_data = sio.loadmat(model_shot_file)

    # load part
    input_data.update(sio.loadmat(part_evecs_file))
    input_data['part_evecs'] = input_data['evecs']
    del input_data['evecs']
    input_data['part_evecs_trans'] = input_data['evecs_trans']
    del input_data['evecs_trans']
    input_data['part_shot'] = part_shot_data['shot']

    # load model
    input_data.update(sio.loadmat(model_evecs_file))
    input_data['model_evecs'] = input_data['evecs']
    del input_data['evecs']
    input_data['model_evecs_trans'] = input_data['evecs_trans']
    del input_data['evecs_trans']
    input_data['model_shot'] = model_shot_data['shot']

    return input_data


def run_test():
    '''
    Test Phase! 

    '''

    # start session
    sess = tf.Session()

    print('restoring graph...')
    saver = tf.train.import_meta_graph(
        '%s/model.ckpt-%d.meta' % (log_dir, args.log_id))
    saver.restore(sess, tf.train.latest_checkpoint('%s' % log_dir))
    graph = tf.get_default_graph()

    # retrieve placeholder variables
    part_shot = graph.get_tensor_by_name('part_shot:0')
    model_shot = graph.get_tensor_by_name('model_shot:0')
    dist_map = graph.get_tensor_by_name('dist_map:0')
    part_evecs = graph.get_tensor_by_name('part_evecs:0')
    part_evecs_trans = graph.get_tensor_by_name('part_evecs_trans:0')
    model_evecs = graph.get_tensor_by_name('model_evecs:0')
    model_evecs_trans = graph.get_tensor_by_name('model_evecs_trans:0')
    phase = graph.get_tensor_by_name('phase:0')
    # Ct_est =
    # graph.get_tensor_by_name('Ct_est/cholesky_solve/MatrixTriangularSolve_1:0')
    Ct_est = graph.get_tensor_by_name('Ct_est:0')
    softCorr = graph.get_tensor_by_name(
        'pointwise_corr_loss/soft_correspondences:0')

    functional_maps = {}
    soft_corr = {}

    # For each pair in test_list
    for test_pair in test_list:
        input_data = get_test_pair(test_pair[0], test_pair[1])

        # feed dict for sess.run
        feed_dict = {phase: True,
                     part_shot: [input_data['part_shot']],
                     model_shot: [input_data['model_shot']],
                     dist_map: [[[None]]],
                     part_evecs: [input_data['part_evecs'][:, 0:args.num_evecs]],
                     part_evecs_trans: [input_data['part_evecs_trans'][0:args.num_evecs, :]],
                     model_evecs: [input_data['model_evecs'][:, 0:args.num_evecs]],
                     model_evecs_trans: [input_data['model_evecs_trans'][0:args.num_evecs, :]],
                     }

        t = time.time()

        # Calculate C, P
        Ct_est_, softCorr_ = sess.run([Ct_est, softCorr], feed_dict=feed_dict)
        print('Computed correspondences for pair: Part = %s, Model = %s. Took %f seconds' % (
            test_pair[0], test_pair[1], time.time() - t))

        functional_maps['C_est_P%s_M%s' % (test_pair[0].replace(
            '.mat', ''), test_pair[1].replace('.mat', ''))] = Ct_est_.transpose()
        # soft_corr['softCorr_P%sM%s'%(test_pair[0].replace('.mat',''),
        # test_pair[1].replace('.mat',''))] = softCorr_

    # Save Results
    sio.savemat('./Results/' + 'functional_maps_' +
                list_of_organs[N] + '.mat', functional_maps)
    # sio.savemat('./Results/' + 'soft_corr.mat', soft_corr)


def main(_):
  run_test()


if __name__ == '__main__':
  tf.app.run()
