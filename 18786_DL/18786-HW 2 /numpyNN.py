import random
import numpy as np
import matplotlib.pyplot as plt
import os

def linearData(n_sample=400):
	theta = np.random.rand() * 2 * np.pi
	w_star = np.array([[np.cos(theta), np.sin(theta)]])
	margin = 0.1
	noise = 0.1
	#  create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (X @ w_star.T) > 0
	label = label.astype(float)
	# create margin
	idx = (label * (X @ w_star.T)) < margin
	X = X + margin * ((idx * label) @ w_star)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def XORData(n_sample=400):
	margin = 0.1
	noise = 0.1
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (X[:, 0] * X[:, 1]) > 0
	label = label.astype(float).reshape((-1, 1))
	# create margin
	pos_flag = X >= 0
	X = X + 0.5 * margin * pos_flag
	X = X - 0.5 * margin * (~pos_flag)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def circleData(n_sample=400):
	noise = 0.05
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	dist = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
	label = dist <= 0.5
	label = label.astype(float).reshape((-1, 1))
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def sinusoidData(n_sample=400):
	noise = 0.05
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (np.sin(np.sum(X, axis=- 1) * 2 * np.pi) > 0)
	label = label.astype(float).reshape((-1, 1))
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def swissrollData(n_sample=400):
	noise = 0.05
	nHalf = int(n_sample / 2)
	# create data
	t = np.random.rand(nHalf, 1)
	x1 = t * np.cos(2 * np.pi * t * 2)
	y1 = t * np.sin(2 * np.pi * t * 2)
	t = np.random.rand(n_sample - nHalf, 1)
	x2 = (-t) * np.cos(2 * np.pi * t * 2)
	y2 = (-t) * np.sin(2 * np.pi * t * 2)
	xy1 = np.concatenate([x1, y1], axis=1)
	xy2 = np.concatenate([x2, y2], axis=1)
	X = np.concatenate([xy1, xy2], axis=0)
	label = np.concatenate([np.ones((nHalf, 1)), np.zeros((n_sample - nHalf, 1))], axis=0)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def sample_data(data_name='circle', nTrain=200, nTest=200, random_seed=0,):
	"""
	Data generation function
	:param data_name: linear-separable, XOR, circle, sinusoid, swiss-roll
	:return:
	"""
	print(f"Sample Data from:{data_name}")
	random.seed(random_seed)
	np.random.seed(random_seed)
	n_sample = nTrain + nTest
	if data_name == 'linear-separable':
		X, label = linearData(n_sample)
	elif data_name == 'XOR':
		X, label = XORData(n_sample)
	elif data_name == 'circle':
		X, label = circleData(n_sample)
	elif data_name == 'sinusoid':
		X, label = sinusoidData(n_sample)
	elif data_name == 'swiss-roll':
		X, label = swissrollData(n_sample)
	else:
		raise NotImplementedError


	indices = np.random.permutation(n_sample)
	train_idx, test_idx = indices[:nTrain], indices[nTrain:]
	x_train = X[train_idx]
	y_train = label[train_idx]
	x_test = X[test_idx]
	y_test = label[test_idx]
	return x_train, y_train, x_test, y_test


def plot_loss(logs):
	"""
	Function to plot training and validation/test loss curves
	:param logs: dict with keys 'train_loss','test_loss' and 'epochs', where train_loss and test_loss are lists with 
				the training and test/validation loss for each epoch
	"""
	plt.figure(figsize=(20, 8))
	plt.subplot(1, 2, 1)
	t = np.arange(len(logs['train_loss']))
	plt.plot(t, logs['train_loss'], label='train_loss', lw=3)
	plt.plot(t, logs['test_loss'], label='test_loss', lw=3)
	plt.grid(1)
	plt.xlabel('epochs',fontsize=15)
	plt.ylabel('loss value',fontsize=15)
	plt.legend(fontsize=15)

def plot_decision_boundary(X, y, pred_fn, boundry_level=None):
    """
    Plots the decision boundary for the model prediction
    :param X: input data
    :param y: true labels
    :param pred_fn: prediction function,  which use the current model to predict。. i.e. y_pred = pred_fn(X)
    :boundry_level: Determines the number and positions of the contour lines / regions.
    :return:
    """
    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = pred_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.7, levels=boundry_level, cmap='viridis_r')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.7,s=50, cmap='viridis_r',)

'''
Deliverable 1: Codebase that trains a generic MLP

Parameters: 
For initialize_mlp: 
- num_layers = num layers w params (hidden + output)
- num_width = each layer width (array w val for each layer) (Ex: [1,2,3] for mlp w 1 hidden)
- opt_act = list of activation for each layer (ex: [sigmoid, relu, tanh]). act could be any of relu, sigmoid, tanh, linear.
- opt_init = init type (xavier or he)

For train_mlp: 
- training_data = data used for param learning
- num_epoch = number of training data passes 
- opt_loss = loss func used ("L2" or "CE")
- opt_optim = optimization algo used for param update ("gd", "gd_momentum", or "Adam")

For test_mlp:
-  test_data = eval data
-  opt_loss = loss func ("L2" or "CE")

Variables / Needed Components (& Dimensions):
- batch size = N 
- input dim = in_dim
- out_dim = 1
- layer dim for layer l = dim(l)
- X (inputs) -> dim: (N, in_dim)
- Weights for layers W[l] = (dim(l), dim(l+1))
- Bias for layer b[l] = (1, dim(l+1))
- training_data.x -> (N, in_dim)
- trianing_data.y -> (N, out_dim) // classifier

Forward Pass: forward(input data) 
- param: training_data.x -> (N, in_dim)
- For each layer l,
	-> W[l] =  (dim(l), dim(l+1)), l = 0 to num_layers-1
	-> b[l] = (1, dim(l+1))
-> Return each layer output value & before activation values for each layer: 
	for layers 1 to l: 
		u^(l) -> (N, dim(l+1)) // before activation
		y(l) -> (N, dim(l+1))
	NOTE: y[num_layers - 1] => output 
-> Overall: y -> array of Length num_layers. 

Backwards Propogation: backward(mlp, training_data.x, y, z, loss_div)
- param: training_data.x -> (N, in_dim)
- for each layer, delta[l] -> (N, dim(l+1))
- With 
- Gradients: For each layer 
	grad_W[l] -> (dim(l), dim(l+1))
	grad_b[l] -> (1, dim(l+1))

Loss: loss(y_L, training_data.y, opt_loss)
- param: 
	y_L -> (N, dim(last layer))
	T -> (N, dim(last layer))
- output: train_loss value, loss_div -> (N, dim(last layer))

update_mlp(mlp, dW, db, opt_optim) -> updates weights and biases in mlp class


Note: 
- num_layers = num of W, b, y, z
- output -> y[num_layers -1 ]


Implementation: 

Feed Forward: 
- Each layer l needs to do: 
	-> before activation: u^(l) = y^(l-1)W^(l)
		where y^(l-1) = row vector outputs of layer y-1
	-> activation: y^(l)=g(u^(l)) where y^(l) is layer l output
- Layer 0 is represented as X (batched inputs), which is y^(0) 
- Note: We shoudl save u^(l) and y^(l-1) for back propogation when doing feed forward. 

Back Propogation: 

General Definition: delta^(l) = dE / du^(l) for gradient of loss 
-> Recursion backwards: delta^(l) = (delta^(l+1)(W^(l+1))^T) dot g'(u^(l))

- Calculate Gradients: 
-> dE / dW^(l) = (y^(l-1))^T * delta^(l)
-> dE / db^(l) = sum of delta^(l) for all 

Format: 
1. For output layer: get output delta^(L) using loss 
-> L2 Loss: E = 1/2N * sum(||y_i - t_i||^2) for all i 
	The derivative with respect to output: dE / dy = 1/N (y -t)
	Output delta = delta^L = (1/N (y^L - t)) dot g'^(L)(u^(L))
-> Cross Entropy Loss: E = -1/n sum_1_to_N([t_i log(y_i) + (1-t_i)log(1-y_i)]) (need sigmoid as final activation)
	The derivative with respect to output: dE / dy = -1/N (t/y - (1-t)/(1-y))
	output delta = delta^L = (-1/N (t/y^L - (1-t)/(1-y^L))) dot g'^(L)(u^(L))
2. For hidden layers, we use backprop formula: delta^(l) = (delta^(l+1)(W^(l+1))^T) dot g'(u^(l))
	Gradients: 
		dE / dW^(l) = (y^(l-1))^T * delta^(l)
		dE / db^(l) = sum of delta^(l) for all 
'''

'''
MLP Class Immplementation 
'''
class MLP_implementation: 


	def __init__(self, layer_num, layer_width_num, layer_activation, opt_init,opt_loss = "he", opt_optimizer = "gd", learning_rate = 1e-3, adam_beta_1=0.9, adam_beta_2=0.99, gd_momentum_coeff=0.9):

		self.layer_num = layer_num 
		self.layer_width_num = layer_width_num # [dim(0),...,dim(layer_num)]
		self.layer_activation = layer_activation # [layer_num elements]
		self.opt_init = opt_init # either he or xavier
		self.learning_rate = learning_rate

		# optimizer & hyperparam
		self.opt_optimizer = opt_optimizer
		self.opt_loss = opt_loss

		# TODO: initialize everything needed for optimizer

		# Adam init moments: https://ml-explained.com/blog/adam-explained
		if (self.opt_optimizer == "Adam"):

			self.adam_beta_1 = adam_beta_1
			self.adam_beta_2 = adam_beta_2

			# first moment weight & bias 
			self.fm_grad_W = None
			self.fm_grad_b = None

			# second moment weight and bias 
			self.sm_grad_W = None
			self.sm_grad_b = None

			self.adam_tick = 0
			self.adam_first_time_init = False

			# TODO: add things needed for adam

		elif (self.opt_optimizer == "gd_momentum"):

			self.gd_momentum_coeff = gd_momentum_coeff
			self.gd_momentum_velocity_W = [] * layer_num
			self.gd_momentum_velocity_b = [] * layer_num
			self.gd_first_time_init = False

			# TODO: add things needed for gd momentum

		elif (self.opt_optimizer == "gd"):
			# nothing to do
			pass

		else:
			print("Invalid optimizer.")


		# set up weights and bases 
		# torch.nn.init.xavier_normal_ Init: var(w_i) = 2 / (fan_in + fan_out)
		# torch.nn.init.kaiming_normal_ (he) Init: var(w_i) = 2 / (fan_in) -> for ReLU
		self.W = []
		self.b = []

		for layer in range(layer_num - 1):

			next_layer = layer + 1

			# Source: https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
			if self.opt_init == "he":
				self.W.append(np.random.randn(self.layer_width_num[layer], self.layer_width_num[layer+1]) * np.sqrt(2 / self.layer_width_num[layer]))
			elif self.opt_init == "xavier":
				self.W.append(np.random.randn(self.layer_width_num[layer], self.layer_width_num[layer+1]) * np.sqrt(2 / (self.layer_width_num[layer] + self.layer_width_num[layer+1])) )

			# add in bias for this layer
			self.b.append(np.zeros((1, self.layer_width_num[next_layer])))

'''
Initialization of MLP
'''
def initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss,opt_optimizer = "gd", learning_rate = 1e-3, adam_beta_1=0.9, adam_beta_2=0.9, gd_momentum_coeff=0.9):

	mlp = MLP_implementation(layer_num=num_layers, layer_width_num=num_width, layer_activation=opt_act, opt_init=opt_init, 
						   opt_optimizer = opt_optimizer, learning_rate= learning_rate, adam_beta_1=adam_beta_1, adam_beta_2=adam_beta_2,
						   gd_momentum_coeff=gd_momentum_coeff, opt_loss=opt_loss)
	
	return mlp


'''
Forward Propogation 
Note: changed to use np.ndarray instead of list for matrix mult, etc

Needs to return:
-> u: vals before activation -> (N, next layer dim) 
-> y: storing vals after activation -> (N, next layer dim) 
'''
def forward(mlp: MLP_implementation, training_data_x: np.ndarray): 

	u = []
	y = []

	# start with inputs as first layer input
	layer_input = training_data_x

	# iterating through all layers
	for layer in range(len(mlp.W)):

		u_layer = layer_input @ mlp.W[layer] + mlp.b[layer]

		# determine the y value for layer based on associated activations
		if (mlp.layer_activation[layer] == "sigmoid"):
			y_layer= 1 / (1 + np.exp(-1 * u_layer))

		elif (mlp.layer_activation[layer] == "tanh"):
			y_layer= np.tanh(u_layer)

		elif (mlp.layer_activation[layer] == "relu"):
			y_layer = np.maximum(0.0, u_layer)

		else: # assume linear
			y_layer = u_layer

		# update the layer value to return 
		u.append(u_layer)
		y.append(y_layer)

		# update for next iter
		layer_input = y_layer

	return u, y
			 
'''
Backward Propogation 

Needs to return:
-> grad_W (length of num hidden layer)
-> grad_b (length of num hidden layer)
'''
def backward(mlp: MLP_implementation, input_data: np.ndarray, u: list, y:list, loss_div: np.ndarray):

	'''
	Activation Layer Derivative
	'''
	def d_activation(layer_activation, u_layer):
		if (layer_activation == "sigmoid"):
			act_der= 1 / (1 + np.exp(-1 * u_layer))
			act_der *= (1 - act_der)
		elif (layer_activation == "tanh"):
			act_der= np.tanh(u_layer)
			act_der = 1 - pow(act_der,2)
		elif (layer_activation == "relu"):
			y_layer = np.maximum(0.0, u_layer)
			act_der = (y_layer > 0).astype(u_layer.dtype)
		else: # assume linear
			act_der = np.ones_like(u_layer)

		return act_der

	num_total_layers = mlp.layer_num - 1

	grad_W = [None] * num_total_layers
	grad_b = [None] * num_total_layers
	

	delta = [0] * num_total_layers
	# output 
	delta[num_total_layers-1] = loss_div * d_activation(mlp.layer_activation[num_total_layers-1], u[num_total_layers-1])

	# back iterate through all layers hidden layer
	starting_hidden_layer = num_total_layers-2
	curr_layer = starting_hidden_layer
	while (curr_layer >= 0):
		delta[curr_layer] = (delta[curr_layer+1] @ mlp.W[curr_layer+1].T) * d_activation(mlp.layer_activation[curr_layer], u[curr_layer])

		curr_layer -= 1 # move pointer one layer back 

	# for layer in range(num_total_layers-2, -1, -1):
	# 	delta[layer] = (delta[layer+1] @ mlp.W[layer+1].T) * d_activation(mlp.layer_activation[layer], u[layer])

	for layer in range(num_total_layers):
		if (layer == 0):
			this_layer_input = input_data
		else:
			this_layer_input = y[layer-1]
		
		grad_W[layer] = this_layer_input.T @ delta[layer]
		grad_b[layer] = np.sum(delta[layer], keepdims=True, axis=0)

	return grad_W, grad_b
	

'''
Update weights given the optimization algo selected.
'''
def update_mlp(mlp: MLP_implementation, grad_W, grad_b):
	
	selected_optim = mlp.opt_optimizer
	selected_learning_rate = mlp.learning_rate

	# basic gd
	if (selected_optim == 'gd'):

		for layer in range(len(mlp.W)):
			# https://www.geeksforgeeks.org/machine-learning/ml-stochastic-gradient-descent-sgd/
			mlp.W[layer] = mlp.W[layer] - grad_W[layer] * selected_learning_rate
			mlp.b[layer] = mlp.b[layer] - grad_b[layer] * selected_learning_rate
		return
	
	elif (selected_optim == 'gd_momentum'):

		# if first time, instantiate it
		if (mlp.gd_first_time_init == False):

			# changed to zeros_ink to fix dims error. 
			mlp.gd_momentum_velocity_W = [np.zeros_like(w) for w in mlp.W]
			mlp.gd_momentum_velocity_b = [np.zeros_like(b) for b in mlp.b]

			mlp.gd_first_time_init = True # set it to true for later

		for layer in range(len(mlp.W)):
			# https://www.geeksforgeeks.org/machine-learning/ml-momentum-based-gradient-optimizer-introduction/
			mlp.gd_momentum_velocity_W[layer] = mlp.gd_momentum_coeff * mlp.gd_momentum_velocity_W[layer] + grad_W[layer]
			mlp.gd_momentum_velocity_b[layer] = mlp.gd_momentum_coeff * mlp.gd_momentum_velocity_b[layer] + grad_b[layer]

			mlp.W[layer] = mlp.W[layer] - mlp.gd_momentum_velocity_W[layer] * selected_learning_rate
			mlp.b[layer] = mlp.b[layer] - mlp.gd_momentum_velocity_b[layer] * selected_learning_rate

		return

	else: # assume Adam

		mlp.adam_tick += 1

		# if first time, instantiate it
		if (mlp.adam_first_time_init == False):
			mlp.fm_grad_W = [np.zeros_like(w) for w in mlp.W]
			mlp.fm_grad_b = [np.zeros_like(b) for b in mlp.b]
			mlp.sm_grad_W = [np.zeros_like(w) for w in mlp.W]
			mlp.sm_grad_b = [np.zeros_like(b) for b in mlp.b]

			mlp.adam_first_time_init = True # set it to true for later

		for layer in range(len(mlp.W)):


			# TODO: figure out if this does it right

			#fm update: momentum term update -> https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
			mlp.fm_grad_W[layer] = mlp.adam_beta_1 * mlp.fm_grad_W[layer] + (1 - mlp.adam_beta_1) * grad_W[layer]
			mlp.fm_grad_b[layer] = mlp.adam_beta_1 * mlp.fm_grad_b[layer] + (1 - mlp.adam_beta_1) * grad_b[layer]

			#sm update
			mlp.sm_grad_W[layer] = mlp.adam_beta_2 * mlp.sm_grad_W[layer] + (1 - mlp.adam_beta_2) * (grad_W[layer] * grad_W[layer]) #squared for second moment
			mlp.sm_grad_b[layer] = mlp.adam_beta_2 * mlp.sm_grad_b[layer] + (1 - mlp.adam_beta_2) * (grad_b[layer] * grad_b[layer])

			# do the bias correction -> m_t_hat = m_t / (1 - beta_1 ^ adam-tick)
			mom_W_hat = mlp.fm_grad_W[layer] / (1 - mlp.adam_beta_1 ** mlp.adam_tick)
			vel_W_hat = mlp.sm_grad_W[layer] / (1 - mlp.adam_beta_2 ** mlp.adam_tick)
			mom_b_hat = mlp.fm_grad_b[layer] / (1 - mlp.adam_beta_1 ** mlp.adam_tick)
			vel_b_hat = mlp.sm_grad_b[layer] / (1 - mlp.adam_beta_2 ** mlp.adam_tick)

			# final weight update -> w_t+1 = w_t - m_t_hat / (sqrt(v_t_hat) + error) * learning rate
			mlp.W[layer] = mlp.W[layer] - (mom_W_hat / (np.sqrt(vel_W_hat) + 1e-8)) * mlp.learning_rate # got better pref when 1e-20 to 1e-8
			mlp.b[layer] = mlp.b[layer] - (mom_b_hat / (np.sqrt(vel_b_hat) + 1e-8)) * mlp.learning_rate 

'''
Loss function
Return: loss value
FIX: ADD dE/dy_pred -> list (for back prop.)
'''
def loss(prediction_y, correct_y, opt_loss):

	if opt_loss == "L2":
		delta = prediction_y - correct_y

		# loss_delta -> dL/dpred_y = 1/N(y_pred - y_correct)
		loss_val = (0.5 * np.sum(delta * delta))
		loss_val = loss_val / correct_y.shape[0] # FIX: scale value back -> easier to log

		# loss_delta -> dL/dpred_y = 1/N(y_pred - y_correct)
		loss_delta = delta / correct_y.shape[0] # gradient of loss for back prop.
		return loss_val, loss_delta
	
	# assume cross entropy otherwise

	# FIX: force the y within error range 
	y_within_range = np.clip(prediction_y, 1e-8, 1 - 1e-8)

	# loss_val= -mean(y*log(p) + (1-y)*log(1-p))
	loss_val = -1 * (np.sum(correct_y * np.log(y_within_range) + (1 - correct_y) * np.log(1 - y_within_range)) / correct_y.shape[0])
	loss_delta = - ((correct_y / y_within_range) - ((1 - correct_y) / (1 - y_within_range))) / correct_y.shape[0]

	return loss_val, loss_delta

'''
Train MLP
'''
def train_mlp(mlp: MLP_implementation, training_data_x, training_data_y, test_x, test_y, num_epoch, opt_loss, opt_optim):

	train_loss = []
	train_accuracy = []
	validation_loss = []
	validation_accuracy = []
	
	for epoch in range(num_epoch):

		u, y = forward(mlp, training_data_x)
		y_L = y[-1] # get the last layer
		
		loss_val, loss_div = loss(y_L, training_data_y, opt_loss)
		train_loss.append(loss_val)
		
		grad_W, grad_b = backward(mlp, training_data_x, u, y, loss_div)
		update_mlp(mlp, grad_W, grad_b)
		   
		# report train & test accuracy
		prediction = (y_L >= 0.5)
		curr_train_accuracy = np.mean(prediction == training_data_y)
		train_accuracy.append(curr_train_accuracy)

		# validation loss
		u_val, y_val = forward(mlp, test_x)
		y_val_l = y_val[-1]
		validation_loss_i, validation_accuracy_i = loss(y_val_l, test_y, opt_loss)
		validation_loss.append(validation_loss_i)

		prediction = (y_val_l >= 0.5)
		curr_val_accuracy = np.mean(prediction == test_y)
		validation_accuracy.append(curr_val_accuracy)

	return train_loss, train_accuracy, validation_loss, validation_accuracy


def test_mlp(mlp: MLP_implementation, test_data_x, test_data_y, opt_loss: str):


	u, y = forward(mlp, test_data_x)
	y_L = y[-1] # get last layer results

	test_loss, test_loss_delta = loss(y_L, test_data_y, opt_loss)
	test_prediction = (y_L >= 0.5)
	test_accuracy = np.mean(test_prediction == test_data_y)

	return test_loss, test_accuracy


def generate_plots(num_epochs, train_loss, test_loss, deliverable_num, mlp, graph_x, graph_y):

	# PLOT PATH #
	# getting the path so I can save these images to my folder.
	os.makedirs(".", exist_ok=True)
	loss_path = os.path.join(".", f"deliverable_{deliverable_num}_loss_plot.png")
	decision_boundary_path = os.path.join(".", f"deliverable_{deliverable_num}_decision_boundary.png")

	# LOSS PLOT #
	# generate epoch array 
	epochs = [i for i in range(num_epochs)]

	# plot training loss
	loss_fig = plt.figure(figsize=(9,5))

	plt.plot(epochs, train_loss, label="train_loss") # plot train loss
	plt.plot(epochs, test_loss, label="test_loss") # plot test loss

	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.grid(True)
	plt.legend()
	plt.title("loss curve")

	loss_fig.savefig(loss_path)
	plt.close(loss_fig)

	# BOUNDARY DECISION PLOT: Using multilinear plot
	# -> https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot 
	# -> https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07#

	# make grid of points to plot in. 
	min_x = graph_x[:,0].min() - 0.1 # find bounds of plot
	max_x = graph_x[:,0].max() + 0.1
	min_y = graph_x[:,1].min() - 0.1
	max_y = graph_x[:,1].max() + 0.1
	x1_grid = np.arange(min_x, max_x, 0.01)
	x2_grid = np.arange(min_y, max_y, 0.01)
	x_plot, y_plot = np.meshgrid(x1_grid, x2_grid) # make each line and row of grid
	grid = np.c_[x_plot.ravel(), y_plot.ravel()] # flatten each grid into vector 

	preactivation_layers, layers_output = forward(mlp, grid)
	y_grid = layers_output[-1].reshape(-1) # flatten into 1D vector
	grad_map_to_prediction = y_grid.reshape(x_plot.shape) # makes each grid coordinate have a pred value

	boundary_fig = plt.figure(figsize=(9,9))
	plt.contourf(x_plot, y_plot, grad_map_to_prediction, alpha=0.5, levels=30, cmap='viridis_r') # viridis_r-> reversed version (pred vals to colors)
	plt.contour(x_plot, y_plot, grad_map_to_prediction, levels=[0.5])

	# visuals
	plt.xlim(x_plot.min(), x_plot.max())
	plt.ylim(y_plot.min(), y_plot.max())
	plt.title(f"iterations:{num_epochs} | train loss:{np.round(train_loss[-1],4)} | val loss:{np.round(test_loss[-1],4)}")
	plt.scatter(graph_x[:,0], graph_x[:,1], c=graph_y.reshape(-1), alpha=0.8, s=60, cmap='viridis_r')
	boundary_fig.savefig(decision_boundary_path, dpi=200) #dpi -> dot per inch

	plt.close(boundary_fig)

	print("graphs complete")

'''
Code testing: Deliverable 1
'''
def deliverable_1():
	# Set Code Variables
	num_layers = 3
	num_width = [2, 32, 1]
	opt_act = ['relu', 'sigmoid'] # for each layer + output
	opt_init = 'he'
	opt_loss = 'CE'
	opt_optim = 'Adam'
	num_epoch = 2000

	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="circle")

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch, opt_loss, opt_optim)
	# print("Training Loss:", training_loss)
	# print("Training Accuracy:", training_acc)


	testing_loss, testing_acc = test_mlp(mlp, test_data_x, test_data_y, opt_loss)
	# print("Testing Loss:", testing_loss)
	# print("Testing Accuracy:", testing_acc)

	generate_plots(num_epoch, training_loss, validation_loss, 1, mlp, training_data_x, training_data_y)

'''
Deliverable 2: Linear-Seperable (200 train, 200 val), Basic GD, L2 loss, 1 hidden layer, 1 perceptron
'''

def deliverable_2():
	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="linear-separable", nTrain=200, nTest=200, random_seed=0)

	# Set Code Variables
	num_layers = 3
	num_width = [2, 1, 1]
	opt_act = ['linear', 'sigmoid'] # for each layer + output

	opt_init = 'he'
	opt_loss = 'L2'
	opt_optim = 'gd'
	learning_rate = 0.1
	num_epoch = 2000

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 2] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 2, mlp, training_data_x, training_data_y)


'''
Deliverable 3: XOR (200 train, 200 val), Basic GD, L2 loss
'''

def deliverable_3():
	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="XOR", nTrain=200, nTest=200, random_seed=0)
	training_data_x = training_data_x * 5 
	test_data_x = test_data_x * 5

	# Set Code Variables
	num_layers = 3
	num_width = [2, 16, 1]
	opt_act = ['relu', 'sigmoid'] # for each layer + output

	opt_init = 'xavier'
	opt_loss = 'L2'
	opt_optim = 'gd_momentum'
	learning_rate = 0.01
	num_epoch = 2000

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 3] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 3, mlp, training_data_x, training_data_y)

'''
Deliverable 4: Circle (200 train, 200 val), L2 loss & CE Loss, GD
'''

def deliverable_4():

	# L2 FUNCTION MODEL

	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="circle", nTrain=200, nTest=200, random_seed=0)

	# Set Code Variables
	num_layers = 3
	num_width = [2, 8, 1]
	opt_act = ['relu', 'linear'] # for each layer + output

	opt_init = 'he'
	opt_loss = 'L2'
	opt_optim = 'Adam'
	learning_rate = 0.001
	num_epoch = 3000

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 4 - L2 Loss] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 41, mlp, training_data_x, training_data_y)

	# CE FUNCTION MODEL

	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="circle", nTrain=200, nTest=200, random_seed=0)

	# Set Code Variables
	num_layers = 3
	num_width = [2, 8, 1]
	opt_act = ['relu', 'sigmoid'] # for each layer + output

	opt_init = 'he'
	opt_loss = 'CE'
	opt_optim = 'Adam'
	learning_rate = 0.001
	num_epoch = 3000

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 4 - CE Loss] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 42, mlp, training_data_x, training_data_y)

'''
Deliverable 5: Sin (200 train, 200 val), Optimizer Playground. 
'''

def deliverable_5():

	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="sinusoid", nTrain=200, nTest=200, random_seed=0)

	# Set Code Variables
	num_layers = 5
	num_width = [2, 32, 32, 32, 1]
	opt_act = ['tanh','tanh', 'tanh', 'sigmoid'] # for each layer + output

	opt_init = 'xavier'
	opt_loss = 'CE'
	learning_rate = 0.001
	num_epoch = 6000

	# OPTIMIZATION STRATEGY #1: ADAM 
	opt_optim = 'Adam'
	adam_beta_1 = 0.9
	adam_beta_2 = 0.999

	# learning_rate = 0.001 # TRYING W DIFFERENT LEARNING RATES

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate, adam_beta_1=adam_beta_1, adam_beta_2=adam_beta_2)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 5 - Adam] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 51, mlp, training_data_x, training_data_y)

	# OPTIMIZATION STRATEGY #2: GD 
	opt_optim = 'gd'
	learning_rate = 0.05 # TRYING W DIFFERENT LEARNING RATES
	# num_epoch = 6000

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 5 - GD] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 52, mlp, training_data_x, training_data_y)

	# OPTIMIZATION STRATEGY #3: GD MOMENTUM
	opt_optim = 'gd_momentum'
	gd_momentum_coeff = 0.8

	learning_rate = 0.02 # TRYING W DIFFERENT LEARNING RATES
	# num_epoch=6000

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate, gd_momentum_coeff=gd_momentum_coeff)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 5 - GD Momentum] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 53, mlp, training_data_x, training_data_y)

'''
Deliverable 6: Swiss Roll (200 train, 200 val), L2 loss & CE Loss, GD
'''
def deliverable_6():

	# L2 FUNCTION MODEL

	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="swiss-roll", nTrain=200, nTest=200, random_seed=0)

	# Set Code Variables
	num_layers = 5
	num_width = [2, 64, 64, 64, 1]
	opt_act = ['tanh','tanh', 'tanh', 'sigmoid'] # for each layer + output

	opt_init = 'xavier'
	opt_loss = 'CE'
	learning_rate = 0.001
	num_epoch = 6000

	opt_optim = 'Adam'
	adam_beta_1 = 0.9
	adam_beta_2 = 0.99

	# learning_rate = 0.001 # TRYING W DIFFERENT LEARNING RATES

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate, adam_beta_1=adam_beta_1, adam_beta_2=adam_beta_2)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, training_data_x, training_data_y, test_data_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 6] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots(num_epoch, training_loss, validation_loss, 6, mlp, training_data_x, training_data_y)

'''
todo Deliverable 7: Non-Linear Embeddings 
'''
# XOR Embedded function -> linearly seperable 
def embedded_xor(x_data):
	x_times_y = (x_data[:,0:1] * x_data[:,1:2])
	embedded_xor = np.concatenate([x_data,x_times_y],axis=1)
	return embedded_xor

# XOR Embedded function -> linearly seperable by maybe just capturing radius of roll
# Staff Hints: 
	# consider using square root
	# phase shift 
	# anything thats coeff and bias is useless -> only consider power, sin since linear coeff doesn't make sense
'''
Archimedean spiral -> x = - t cos(2 * pi * t * 2), y = -t * sin (2 * pi * t * 2)
'''
def embedded_swiss_roll(x_data):
	x, y = (x_data[:,0:1], x_data[:,1:2])

	t = np.sqrt(x * x + y * y) # radius
	theta = np.arctan2(y,x) # should be about 4*pi*radius

	cosA = np.cos(theta)
	shifted_theta = theta - 2 * np.pi * t * 2 # shift from swissrollData generation

	embedded_swiss_roll_output = np.concatenate([x,y,-t * np.cos(shifted_theta), -t * np.sin(shifted_theta)],axis=1)

	return embedded_swiss_roll_output

def generate_plots_embedded_xor(num_epochs, train_loss, test_loss, deliverable_num, mlp, graph_x, graph_y):

	# PLOT PATH #
	# getting the path so I can save these images to my folder.
	os.makedirs(".", exist_ok=True)
	loss_path = os.path.join(".", f"deliverable_{deliverable_num}_loss_plot.png")
	decision_boundary_path = os.path.join(".", f"deliverable_{deliverable_num}_decision_boundary.png")

	# LOSS PLOT #
	# generate epoch array 
	epochs = [i for i in range(num_epochs)]

	# plot training loss
	loss_fig = plt.figure(figsize=(9,5))

	plt.plot(epochs, train_loss, label="train_loss") # plot train loss
	plt.plot(epochs, test_loss, label="test_loss") # plot test loss

	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.grid(True)
	plt.legend()
	plt.title("loss curve")

	loss_fig.savefig(loss_path)
	plt.close(loss_fig)

	# BOUNDARY DECISION PLOT #
	min_x = graph_x[:,0].min() - 0.1
	max_x = graph_x[:,0].max() + 0.1
	min_y = graph_x[:,1].min() - 0.1
	max_y = graph_x[:,1].max() + 0.1

	# make grid of points 
	x_plot, y_plot = np.meshgrid(np.arange(min_x, max_x, 0.01), np.arange(min_y, max_y, 0.01))
	grid = np.c_[x_plot.ravel(), y_plot.ravel()] # flatten
	embedded_grid = embedded_xor(grid)

	preactivation_layers, layers_output = forward(mlp, embedded_grid)
	y_grid = layers_output[-1].reshape(-1)
	grad_map_to_prediction = y_grid.reshape(x_plot.shape) # prediction list

	boundary_fig = plt.figure(figsize=(9,9))
	plt.contourf(x_plot, y_plot, grad_map_to_prediction, alpha=0.7, levels=30, cmap='viridis_r')
	plt.contour(x_plot, y_plot, grad_map_to_prediction, levels=[0.5])

	plt.xlim(x_plot.min(), x_plot.max())
	plt.ylim(y_plot.min(), y_plot.max())
	plt.title(f"iterations:{num_epochs} | train loss:{np.round(train_loss[-1],4)} | val loss:{np.round(test_loss[-1],4)}")

	plt.scatter(graph_x[:,0], graph_x[:,1], c=graph_y.reshape(-1), alpha=0.8, s=50, cmap='viridis_r')

	boundary_fig.savefig(decision_boundary_path, dpi=200)
	plt.close(boundary_fig)

	print("graphs complete")

def generate_plots_embedded_swiss_roll(num_epochs, train_loss, test_loss, deliverable_num, mlp, graph_x, graph_y):

	# PLOT PATH #
	# getting the path so I can save these images to my folder.
	os.makedirs(".", exist_ok=True)
	loss_path = os.path.join(".", f"deliverable_{deliverable_num}_loss_plot.png")
	decision_boundary_path = os.path.join(".", f"deliverable_{deliverable_num}_decision_boundary.png")

	# LOSS PLOT #
	# generate epoch array 
	epochs = [i for i in range(num_epochs)]

	# plot training loss
	loss_fig = plt.figure(figsize=(9,5))

	plt.plot(epochs, train_loss, label="train_loss") # plot train loss
	plt.plot(epochs, test_loss, label="test_loss") # plot test loss

	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.grid(True)
	plt.legend()
	plt.title("loss curve")

	loss_fig.savefig(loss_path)
	plt.close(loss_fig)

	# BOUNDARY DECISION PLOT #
	min_x = graph_x[:,0].min() - 0.1
	max_x = graph_x[:,0].max() + 0.1
	min_y = graph_x[:,1].min() - 0.1
	max_y = graph_x[:,1].max() + 0.1

	# make grid of points 
	x_plot, y_plot = np.meshgrid(np.arange(min_x, max_x, 0.01), np.arange(min_y, max_y, 0.01))
	grid = np.c_[x_plot.ravel(), y_plot.ravel()] # flatten
	embedded_grid = embedded_swiss_roll(grid)

	preactivation_layers, layers_output = forward(mlp, embedded_grid)
	y_grid = layers_output[-1].reshape(-1)
	grad_map_to_prediction = y_grid.reshape(x_plot.shape) # prediction list

	boundary_fig = plt.figure(figsize=(9,9))
	plt.contourf(x_plot, y_plot, grad_map_to_prediction, alpha=0.7, levels=30, cmap='viridis_r')
	plt.contour(x_plot, y_plot, grad_map_to_prediction, levels=[0.5])

	plt.xlim(x_plot.min(), x_plot.max())
	plt.ylim(y_plot.min(), y_plot.max())
	plt.title(f"iterations:{num_epochs} | train loss:{np.round(train_loss[-1],4)} | val loss:{np.round(test_loss[-1],4)}")

	plt.scatter(graph_x[:,0], graph_x[:,1], c=graph_y.reshape(-1), alpha=0.8, s=50, cmap='viridis_r')

	boundary_fig.savefig(decision_boundary_path, dpi=200)
	plt.close(boundary_fig)

	print("graphs complete")

def deliverable_7_xor():
	# XOR implementation 

	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="XOR", nTrain=200, nTest=200, random_seed=0)
	non_linear_training_x = embedded_xor(training_data_x)
	non_linear_testing_x = embedded_xor(test_data_x)

	# Set Code Variables
	num_layers = 2
	num_width = [non_linear_training_x.shape[1], 1]
	opt_act = ['sigmoid'] # for each layer + output
	opt_init = 'xavier'
	opt_loss = 'CE'
	learning_rate = 0.01
	num_epoch = 3000

	opt_optim = 'Adam'
	adam_beta_1 = 0.9
	adam_beta_2 = 0.99

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate, adam_beta_1=adam_beta_1, adam_beta_2=adam_beta_2)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, non_linear_training_x, training_data_y, non_linear_testing_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 7 - XOR] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots_embedded_xor(num_epoch, training_loss, validation_loss, 71, mlp, training_data_x, training_data_y)


def deliverable_7_swiss_roll():

	# Swiss-roll embedded implementation

	# Create data
	training_data_x, training_data_y, test_data_x, test_data_y = sample_data(data_name="swiss-roll", nTrain=200, nTest=200, random_seed=0)
	non_linear_training_x = embedded_swiss_roll(training_data_x)
	non_linear_testing_x = embedded_swiss_roll(test_data_x)

	# Set Code Variables
	num_layers = 2
	num_width = [non_linear_training_x.shape[1], 1]
	opt_act = ['sigmoid'] # for each layer + output
	opt_init = 'xavier'
	opt_loss = 'CE'
	learning_rate = 0.001
	num_epoch = 6000

	opt_optim = 'Adam'
	adam_beta_1 = 0.9
	adam_beta_2 = 0.99

	mlp = initialize_mlp(num_layers, num_width, opt_act, opt_init, opt_loss=opt_loss, opt_optimizer=opt_optim, learning_rate=learning_rate, adam_beta_1=adam_beta_1, adam_beta_2=adam_beta_2)

	training_loss, training_acc, validation_loss, validation_accuracy = train_mlp(mlp, non_linear_training_x, training_data_y, non_linear_testing_x, test_data_y, num_epoch=num_epoch, opt_loss=opt_loss, opt_optim=opt_optim)

	best_validation_accuracy = float(np.max(validation_accuracy))
	final_validation_accuracy = float(validation_accuracy[-1])
	print(f"[Deliverable 7 - Swiss Roll] Best Validation Accuracy: {best_validation_accuracy} | Final Validation Accuracy: {final_validation_accuracy}")

	generate_plots_embedded_swiss_roll(num_epoch, training_loss, validation_loss, 72, mlp, training_data_x, training_data_y)



'''
Run all components of the project.
'''
# run deliverable 1
# deliverable_1()

# run deliverable 2
deliverable_2()

# run deliverable 3
deliverable_3()

# run deliverable 4
deliverable_4()

# run deliverable 5
deliverable_5()

# run deliverable_6
deliverable_6()

#run deliverable_7
deliverable_7_xor()
deliverable_7_swiss_roll()