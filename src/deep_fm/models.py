import tensorflow as tf
import autograd.numpy as anp

from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient


flags = tf.app.flags
FLAGS = flags.FLAGS

     
def solve_ls(A, B):
    """functional maps layer.

    Args:
        A: part descriptors projected onto part shape eigenvectors.
        B: model descriptors projected onto model shape eigenvectors.

    Returns:
        Ct_est: estimated C (transposed), such that CA ~= B
        safeguard_inverse:
    """

    # transpose input matrices
    At = tf.transpose(A, perm = [0, 2, 1])
    Bt = tf.transpose(B, perm = [0, 2, 1])
	
    # solve C via least-squares
    
    # Ct_est = tf.matrix_solve_ls(At, Bt, l2_regularizer=0.0, fast=True,name= 'Ct_est')
    # C_est = tf.transpose(Ct_est, perm = [0,2,1], name= 'C_est') 
    
    S, U, V = tf.linalg.svd(tf.matmul(B,At), full_matrices=True)
    C_est = tf.matmul(U,tf.transpose(V, perm = [0,2,1]))
    Ct_est = tf.transpose(C_est, perm = [0,2,1], name= 'Ct_est')   
   

#   # calculate error for safeguarding
    safeguard_inverse = tf.nn.l2_loss(tf.matmul(At,Ct_est) - Bt) / tf.to_float(tf.reduce_prod(tf.shape(A)))

    return C_est, safeguard_inverse

def pointwise_corr_layer(C_est, model_evecs, part_evecs_trans, dist_map):
    """Point-wise correlation between learned descriptors.

     Args:
        C_est: estimated C matrix from previous layer.
        model_evecs: eigen vectors of model shape.
        part_evecs_trans: eigen vectors of part shape, transposed with area preservation factor.
        dist_map: matrix of geodesic distances on model.
     ============================================================================================
     Returns:
        - average_error: average distance error
        - P_norm: soft correspondace matrix
        - loss: value of cost function 
    """

	# Calculate soft correspondence matrix P
    P = tf.matmul(tf.matmul(model_evecs, C_est), part_evecs_trans) # Correspondance matrix T = Psi * C * Phi_T * M
    P = tf.abs(P) # absolute value of each element in x
    P_norm = tf.nn.l2_normalize(P, axis=1, name='soft_correspondences') # normalize the columns
	
	# Loss function
    loss = tf.nn.l2_loss(tf.multiply(dist_map, P_norm)) # changed
    loss /= tf.to_float(tf.shape(P)[1] * tf.shape(P)[0])

	# Hard-Correspondence
    one_hot = tf.one_hot(indices=tf.argmax(P_norm, 1),
                         depth=tf.shape(P_norm)[1],
                         on_value=True,
                         off_value=False,
                         axis=-1,
                         name='hard_correspondences')
    one_hot = tf.boolean_mask(dist_map, one_hot)  
    # average distance error

    average_error = tf.reduce_mean(one_hot)

    return average_error, P_norm, loss


def res_layer(x_in, dims_out, scope, phase):
    """A residual layer implementation.

    Args:
        x_in: input descriptor per point (dims = batch_size X #pts X #channels)
        dims_out: num channles in output. Usually the same as input for a standard resnet layer.
        scope: scope name for variable sharing.
        phase: train\test.

    """
    with tf.variable_scope(scope):
        x = tf.contrib.layers.fully_connected(x_in, dims_out, activation_fn=None, scope='dense_1')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn_1')
        x = tf.nn.elu(x, 'elu') # changed
        x = tf.contrib.layers.fully_connected(x, dims_out, activation_fn=None, scope='dense_2')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn_2')

        # if dims_out change, modify input via linear projection (as suggested in resNet)
        if not x_in.get_shape().as_list()[-1] == dims_out:
            x_in = tf.contrib.layers.fully_connected(x_in, dims_out, activation_fn=None, scope='projection')

        x += x_in

        return tf.nn.elu(x) # changed



def fmnet_model(phase, part_shot, model_shot, dist_map, part_evecs,	part_evecs_trans, model_evecs, model_evecs_trans):
	"""Build FM-net model.

	Args:
		phase: train\test.
		part_shot: SHOT descriptor of source shape (part).
		model_shot: SHOT descriptor of target shape (model).
		dist_map: distance map on target shape to evaluate geodesic error
		part_evecs: eigenvectors on source shape
		part_evecs_trans: transposed part_evecs with mass matrix correction
		model_evecs: eigenvectors on target shape
		model_evecs_trans: transposed model_evecs with mass matrix correction
	=========================================================================
	Returns:
		- net_loss: net loss
		- average_error: average distance error 
		- merged: loss and average error summaries 
		- safeguard_inverse: value of min||AC-B||
		- P_norm: soft corr matrix 		
		- net: contains net layers, A = Fi_hat, B = Psi_hat, C_est
	"""

	net = {}

	for i_layer in range(FLAGS.num_layers):
		with tf.variable_scope("layer_%d" % i_layer) as scope:
			if i_layer == 0:
				net['layer_%d_part' % i_layer] = res_layer(part_shot, dims_out=int(part_shot.shape[-1]), scope=scope,
														   phase=phase)
				scope.reuse_variables()
				net['layer_%d_model' % i_layer] = res_layer(model_shot, dims_out=int(model_shot.shape[-1]), scope=scope,
															phase=phase)
			else:
				net['layer_%d_part' % i_layer] = res_layer(net['layer_%d_part' % (i_layer - 1)],
														   dims_out=int(part_shot.shape[-1]),
														   scope=scope, phase=phase)
				scope.reuse_variables()
				net['layer_%d_model' % i_layer] = res_layer(net['layer_%d_model' % (i_layer - 1)],
															dims_out=int(part_shot.shape[-1]),
															scope=scope, phase=phase)

	#  project output features on the shape Laplacian eigen functions
	layer_C_est = i_layer + 1  # grab current layer index
	A = tf.matmul(part_evecs_trans, net['layer_%d_part' % (layer_C_est - 1)]) # Fi_hat eigenvector
	net['A'] = A
	B = tf.matmul(model_evecs_trans, net['layer_%d_model' % (layer_C_est - 1)]) # Psi_hat eigenvector
	net['B'] = B

	#  FM-layer: evaluate C_est
	net['C_est'], safeguard_inverse = solve_ls(A, B) 
	# net['C_est'], safeguard_inverse = solve_ls(A, B, part_evals, model_evals, alpha=0.1)

	#  Evaluate loss via soft-correspondence error
	with tf.variable_scope("pointwise_corr_loss"):
		average_error, P_norm, net_loss = pointwise_corr_layer(net['C_est'], model_evecs, part_evecs_trans, dist_map)

	tf.summary.scalar('net_loss', net_loss)
	tf.summary.scalar('average_error', average_error)
	merged = tf.summary.merge_all()

	return net_loss, safeguard_inverse, merged, P_norm, average_error, net