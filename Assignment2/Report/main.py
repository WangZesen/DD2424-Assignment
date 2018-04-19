import random, math
import numpy as np
import copy as cp
import scipy.io as sio

import matplotlib.pyplot as plt

def activateRelu(input_data):
	# input_data: d_in * N
	output_data = cp.deepcopy(input_data)
	output_data[output_data <= 0] *= 0.00 # change to 0.01 if it's leaky ReLU
	return output_data

def fullyConnect(input_data, W, b):
	# input_data: d_in * N
	# W: d_out * d_in
	# b: d_out * 1
	assert input_data.shape[0] == W.shape[1]
	assert W.shape[0] == b.shape[0]
	output_data = np.dot(W, input_data) + b
	return output_data

def softmax(input_data):
	# input_data: K * N
	output_data = np.exp(input_data)
	for i in range(input_data.shape[1]):
		output_data[:, i] = output_data[:, i] / sum(output_data[:, i])
	return output_data

def crossEntropyLoss(output_data, label):
	# input_data: K * N
	# label: one-hot
	assert output_data.shape == label.shape
	out = - np.log(output_data)
	out = np.multiply(out, label)
	out = np.sum(out)
	return out / output_data.shape[1]

def regularisationLoss(W, lambda_):
	# W: d_out * d_in
	loss = sum([np.sum(np.square(w)) for w in W]) * lambda_
	return loss

def evaluateClassifierVerbose(X, W, b):
	fc = []
	act = []
	last = X
	fc.append(fullyConnect(X, W[0], b[0]))
	act.append(activateRelu(fc[0]))
	fc.append(fullyConnect(act[0], W[1], b[1]))
	p = softmax(fc[1])
	return fc, act, p

def evaluateClassifier(X, W, b):
	fc = []
	act = []
	last = X
	
	fc.append(fullyConnect(X, W[0], b[0]))
	act.append(activateRelu(fc[0]))
	fc.append(fullyConnect(act[0], W[1], b[1]))
	p = softmax(fc[1])
	return p

def computeLoss(X, Y, W, b, lambda_):
	p = evaluateClassifier(X, W, b)
	loss = crossEntropyLoss(p, Y) + regularisationLoss(W, lambda_)
	return loss

def regularisationLossGradient(W, lambda_):
	grad_W = []
	for i in range(len(W)):
		grad_W.append(2 * lambda_ * W[i])
	return grad_W

def softmaxCrossEntropyLossGradient(p, Y):
	return p - Y

def activationReluGradient(lastGrad, fc):
	grad = cp.deepcopy(lastGrad)
	grad[fc <= 0] *= 0.00 # change to 0.01 if it's leaky ReLU
	return grad

def fullyConnectGradient(lastGrad, W):
	return np.dot(W.T, lastGrad)

def computeGradient(X, Y, W, b, lambda_):
	d = X.shape[0]
	K = Y.shape[0]
	m = 50
	
	grad_W = [np.zeros((m, d)), np.zeros((K, m))]
	grad_b = [np.zeros((m, 1)), np.zeros((K, 1))]
	
	for i in range(X.shape[1]):
		fc, act, p = evaluateClassifierVerbose(X[:, i : i+1], W, b)	
		grad = softmaxCrossEntropyLossGradient(p, Y[:, i : i+1])
		# grad = activationReluGradient(grad, fc[1])
		grad_W[1] = grad_W[1] + np.dot(grad, act[0].T)
		grad_b[1] = grad_b[1] + grad
		grad = fullyConnectGradient(grad, W[1])
		grad = activationReluGradient(grad, fc[0])
		grad_W[0] = grad_W[0] + np.dot(grad, X[:, i : i+1].T)
		grad_b[0] = grad_b[0] + grad
	
	grad_W[0] = grad_W[0] / X.shape[1]
	grad_W[1] = grad_W[1] / X.shape[1]
	grad_b[0] = grad_b[0] / X.shape[1]
	grad_b[1] = grad_b[1] / X.shape[1]	
	
	grad_RW = regularisationLossGradient(W, lambda_)
	grad_W[0] = grad_W[0] + grad_RW[0]
	grad_W[1] = grad_W[1] + grad_RW[1]

	return grad_W, grad_b

def computeGradsNumSlow(X, Y, W, b, lambda_, h):
	grad_W = [np.zeros(W[i].shape) for i in range(len(W))]
	grad_b = [np.zeros(b[i].shape) for i in range(len(b))]
	
	for k in range(len(W)):
		for i in range(W[k].shape[0]):
			for j in range(W[k].shape[1]):
				W[k][i][j] -= h
				c1 = computeLoss(X, Y, W, b, lambda_)
				W[k][i][j] += h + h
				c2 = computeLoss(X, Y, W, b, lambda_)
				W[k][i][j] -= h
				grad_W[k][i][j] = (c2 - c1) / (2 * h)
		for i in range(b[k].shape[0]):
			for j in range(b[k].shape[1]):
				b[k][i][j] -= h
				c1 = computeLoss(X, Y, W, b, lambda_)
				b[k][i][j] += h + h
				c2 = computeLoss(X, Y, W, b, lambda_)
				b[k][i][j] -= h
				grad_b[k][i][j] = (c2 - c1) / (2 * h)				
	return grad_W, grad_b

def computeAccuracy(X, y, W, b):
	p = evaluateClassifier(X, W, b)
	count = 0
	for i in range(X.shape[1]):
		if np.argmax(p[:, i]) == y[i]:
			count = count + 1
	return count / X.shape[1]

def miniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, lambda_, params, verbose = False, early_stop = False):
	N = train_X.shape[1]
	last_grad_W = [np.zeros(W[i].shape) for i in range(len(W))]
	last_grad_b = [np.zeros(b[i].shape) for i in range(len(b))]
	Wstar = cp.deepcopy(W)
	bstar = cp.deepcopy(b)
	
	Wbest = cp.deepcopy(W)
	bbset = cp.deepcopy(b)

	best_acc = 0
	best_epoch = 0
	
	eta = params['eta']
	train_loss = []
	val_loss = []
	for i in range(params['n_epochs']):
		for j in range(N // params['n_batch']):
			batch_X = train_X[:, j * params['n_batch'] : (j + 1) * params['n_batch']]
			batch_Y = train_Y[:, j * params['n_batch'] : (j + 1) * params['n_batch']]
			grad_W, grad_b = computeGradient(batch_X, batch_Y, Wstar, bstar, lambda_)

			for k in range(len(W)):
				grad_W[k] = eta * grad_W[k] + params['momentum'] * last_grad_W[k]
				grad_b[k] = eta * grad_b[k] + params['momentum'] * last_grad_b[k]
				Wstar[k] = Wstar[k] - grad_W[k]
				bstar[k] = bstar[k] - grad_b[k]
			last_grad_W = cp.deepcopy(grad_W)
			last_grad_b = cp.deepcopy(grad_b)

		if (i + 1) % params['decay_gap'] == 0:
			eta = eta * params['decay']

		if verbose:
			train_loss.append(computeLoss(train_X, train_Y, Wstar, bstar, lambda_))
			val_loss.append(computeLoss(val_X, val_Y, Wstar, bstar, lambda_))
			val_acc = computeAccuracy(val_X, val_y, Wstar, bstar)
			if val_acc > best_acc:
				Wbest = cp.deepcopy(Wstar)
				bbest = cp.deepcopy(bstar)
				best_epoch = i
				best_acc = val_acc
				print ("Current Best Validation Accuracy at Epoch {}: {}".format(i + 1, best_acc))
			elif (i - best_epoch > 10) and early_stop:
				print ("Early stopping at epoch {}".format(i + 1))
				return Wstar, bstar, train_loss, val_loss, Wbest, bbest

			print ("Epoch {} Finished, Train Loss: {}, Validation Loss: {}".format(i + 1, train_loss[-1], val_loss[-1]))
	if verbose:
		return Wstar, bstar, train_loss, val_loss, Wbest, bbest
	else:
		return Wstar, bstar

def computeRelativeError(p1, p2):
	eps = 1e-12
	error = 0
	for i in range(len(p1)):
		absolute_error = np.abs(p1[i] - p2[i])
		denominator = np.maximum(eps, np.abs(p1[i]) + np.abs(p2[i]))
		error += np.sum(np.divide(absolute_error, denominator)) / p1[i].size
	return error

def loadBatch(filename):
	# Load mat file
	content = sio.loadmat("Datasets/cifar-10-batches-mat/{}".format(filename))
	X = content['data'].T / 255
	mean = np.mean(X, axis = 1)
	# X = (X.T - mean).T
	y = content['labels']
	y = np.reshape(y, (y.shape[0],))
	Y = []
	for i in range(X.shape[1]):
		Y.append([0 for col in range(10)])
		Y[i][y[i]] = 1
	Y = np.array(Y).T
	return X, Y, y, mean

def normalize(X, mean):
	X = (X.T - mean).T
	return X

def initial(K, d, t):
	# Initialize paramters
	m = 50
	if t == "Gaussian":
		W = [np.random.normal(0, 0.001, (m, d)), np.random.normal(0, 0.001, (K, m))]
		b = [np.random.normal(0, 0.001, (m, 1)), np.random.normal(0, 0.001, (K, 1))]
	elif t == "Xavier":
		W = [np.random.normal(0, (2 / (m + d)) ** 0.5, (m, d)), np.random.normal(0, (2 / (K + m)) ** 0.5, (K, m))]
		b = [np.random.normal(0.001, (2 / (m + d)) ** 0.5, (m, 1)), np.random.normal(0.001, (2 / (K + m)) ** 0.5, (K, 1))]		
		# b = [np.ones((m, 1)) * 0.01, np.ones((K, 1)) * 0.01]
	elif t == "He":
		W = [np.random.normal(0, (2 / d) ** 0.5, (m, d)), np.random.normal(0, (2 / m) ** 0.5, (K, m))]
		b = [np.random.normal(0.001, (2 / d) ** 0.5, (m, 1)), np.random.normal(0.001, (2 / m) ** 0.5, (K, 1))]
	else:
		print ("Initialization Type Error!")
	return W, b

if __name__ == "__main__":
	
	np.random.seed(1)
	
	train_X, train_Y, train_y, mean = loadBatch("data_batch_1.mat")
	val_X, val_Y, val_y, mean_ = loadBatch("data_batch_2.mat")
	test_X, test_Y, test_y, mean_ = loadBatch("test_batch.mat")
	
	train_X = normalize(train_X, mean)
	val_X = normalize(val_X, mean)
	test_X = normalize(test_X, mean)

	tasks = ["Task 1: Compute Relative Error",
	"Task 2: Check Overfit",
	"Task 3: Find the Best Momentum",
	"Task 4: Find Reasonable Range for Eta",
	"Task 5: Find the Best Eta and Lambda",
	"Task 6: Train the Network",
	"Task 7 (Optional): Optimize the performance"]
	
	task_label = input("\n".join(tasks) + "\nTask #: ")
	
	if task_label == "1":
	
		train_X = train_X[1:400, :]
		
		d = train_X.shape[0]
		K = train_Y.shape[0]
		W, b = initial(K, d, "Gaussian")
		
		lambda_ = 0.1
		grad_W, grad_b = computeGradient(train_X[:, 0:10], train_Y[:, 0:10], W, b, lambda_)
		grad_W1, grad_b1 = computeGradsNumSlow(train_X[:, 0:10], train_Y[:, 0:10], W, b, lambda_, 1e-6)
		
		print ("Relative Error for W (lambda = 0.1): ", computeRelativeError([grad_W[1]], [grad_W1[1]]))
		print ("Relative Error for b (lambda = 0.1): ", computeRelativeError(grad_b, grad_b1))
		
		lambda_ = 0
		grad_W, grad_b = computeGradient(train_X[:, 0:10], train_Y[:, 0:10], W, b, lambda_)
		grad_W1, grad_b1 = computeGradsNumSlow(train_X[:, 0:10], train_Y[:, 0:10], W, b, lambda_, 1e-6)
		
		print ("Relative Error for W (lambda = 0): ", computeRelativeError([grad_W[1]], [grad_W1[1]]))
		print ("Relative Error for b (lambda = 0): ", computeRelativeError(grad_b, grad_b1))		
	
	if task_label == "2":
		d = train_X.shape[0]
		K = train_Y.shape[0]
		W, b = initial(K, d, "Gaussian")
		lambda_ = 0

		train_X = train_X[:, 0:100]
		train_Y = train_Y[:, 0:100]
		train_y = train_y[0:100]

		params = {
			'n_batch': 100,
			'n_epochs': 200,
			'eta': 5e-2,
			'momentum': 0,
			'decay': 1,
			'decay_gap': 1
		}

		x = [i + 1 for i in range(params['n_epochs'])]
		Wstar, bstar, train_loss, val_loss, Wbest, bbest = miniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, lambda_, params, verbose = True)
		plt.plot(x, train_loss, label = "train")
		plt.plot(x, val_loss, label = "validation")
		plt.legend()
		plt.show()

	if task_label == "3":

		d = train_X.shape[0]
		K = train_Y.shape[0]
		W, b = initial(K, d, "Gaussian")
		lambda_ = 1e-6

		params = {
			'n_batch': 100,
			'n_epochs': 10,
			'eta': 1e-2,
			'momentum': 0.9,
			'decay': 0.95,
			'decay_gap': 1
		}

		x = [i + 1 for i in range(params['n_epochs'])]
		for m in [0, 0.5, 0.9, 0.95, 0.99]:
			params['momentum'] = m
			Wstar, bstar, train_loss, val_loss, Wbest, bbest = miniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, lambda_, params, verbose = True)
			plt.plot(x, train_loss, label = 'rho = {} (train)'.format(m))
			print ("Momentum = {}".format(m))
			print ("Accuracy on Test Set: {}".format(computeAccuracy(test_X, test_y, Wstar, bstar)))
		plt.legend()
		plt.show()

	if task_label == "4":
		d = train_X.shape[0]
		K = train_Y.shape[0]
		W, b = initial(K, d, "Gaussian")
		lambda_ = 1e-6
		
		params = {
			'n_batch': 100,
			'n_epochs': 5,
			'eta': 1e-2,
			'momentum': 0.95,
			'decay': 0.95,
			'decay_gap': 1
		}

		x = [i + 1 for i in range(params['n_epochs'])]
		for m in range(5):
			params['eta'] = 5e-3 + 2e-2 * m
			Wstar, bstar, train_loss, val_loss, Wbest, bbest = miniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, lambda_, params, verbose = True)
			plt.plot(x, train_loss, label = 'eta = {} (train)'.format(params['eta']))
			print ("Learning Rate = {}".format(params['eta']))
			print ("Accuracy on Test Set: {}".format(computeAccuracy(test_X, test_y, Wstar, bstar)))
		plt.legend()
		plt.show()
		
		pass

	if task_label == "5":
		
		d = train_X.shape[0]
		K = train_Y.shape[0]
		W, b = initial(K, d, "Gaussian")

		lambda_e_min = -8
		lambda_e_max = -2
		eta_e_min = math.log(0.001) / math.log(10)
		eta_e_max = math.log(0.040) / math.log(10)
		params = {
			'n_batch': 100,
			'n_epochs': 10,
			'eta': 0,
			'momentum': 0.95,
			'decay': 0.95,
			'decay_gap': 1
		}


		lambdas = []
		etas = []
		results = []
		exp_time = 160

		f = open("lambda_eta_select.txt", "w")

		for i in range(exp_time):
			lambda_ = 10 ** (lambda_e_min + random.uniform(0, 1) * (lambda_e_max - lambda_e_min))
			params['eta'] = 10 ** (eta_e_min + random.uniform(0, 1) * (eta_e_max - eta_e_min))
			Wstar, bstar = miniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, lambda_, params)

			results.append(computeAccuracy(val_X, val_y, Wstar, bstar))
			lambdas.append(lambda_)
			etas.append(params['eta'])
			print ("Lambda = {}, Eta = {}, Accuracy = {}".format(lambda_, params['eta'], results[-1]))

		results = list(zip(results, lambdas, etas))
		results.sort(key = lambda x: -x[0])
		for i in range(min(exp_time, 500)):
			f.write("Accuracy: {}, lambda: {}, eta: {}\n".format(results[i][0], results[i][1], results[i][2]))

		f.close()

	if task_label == "6":
		train_X, train_Y, train_y, mean_ = loadBatch("data_batch_1.mat")
		test_X, test_Y, test_y, mean_ = loadBatch("test_batch.mat")

		for i in range(1, 5):
			tem_X, tem_Y, tem_y, mean_ = loadBatch("data_batch_{}.mat".format(i + 1))
			train_X = np.concatenate((train_X, tem_X), axis = 1)
			train_Y = np.concatenate((train_Y, tem_Y), axis = 1)
			train_y = np.concatenate((train_y, tem_y))

		val_X = train_X[:, 0:1000]
		val_Y = train_Y[:, 0:1000]
		val_y = train_y[0:1000]

		print (val_X.shape, val_Y.shape, val_y.shape)

		train_X = train_X[:, 1000:]
		train_Y = train_Y[:, 1000:]
		train_y = train_y[1000:]

		mean = np.mean(train_X, axis = 1)
		train_X = normalize(train_X, mean)
		val_X = normalize(val_X, mean)
		test_X = normalize(test_X, mean)

		d = train_X.shape[0]
		K = train_Y.shape[0]
		W, b = initial(K, d, "Gaussian")


		params = {
			'n_batch': 100,
			'n_epochs': 30,
			'eta': 0.017453577972249945, # 0.010800662290914505,
			'momentum': 0.95,
			'decay': 0.95,
			'decay_gap': 1
		}
		lambda_ = 0.0023292248102687557 # 0.002963774526491722
		
		Wstar, bstar, train_loss, val_loss, Wbest, bbest = miniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, lambda_, params, verbose = True)
		x = [i + 1 for i in range(params['n_epochs'])]
		plt.plot(x, train_loss, label = 'train')
		plt.plot(x, val_loss, label = 'val')

		print ("Accuracy on test set (final): {}".format(computeAccuracy(test_X, test_y, Wstar, bstar)))
		print ("Accuracy on test set (best): {}".format(computeAccuracy(test_X, test_y, Wbest, bbest)))
		plt.legend()
		plt.show()
		
	if task_label == "7":
		train_X, train_Y, train_y, mean_ = loadBatch("data_batch_1.mat")
		test_X, test_Y, test_y, mean_ = loadBatch("test_batch.mat")

		for i in range(1, 5):
			tem_X, tem_Y, tem_y, mean_ = loadBatch("data_batch_{}.mat".format(i + 1))
			train_X = np.concatenate((train_X, tem_X), axis = 1)
			train_Y = np.concatenate((train_Y, tem_Y), axis = 1)
			train_y = np.concatenate((train_y, tem_y))

		val_X = train_X[:, 0:1000]
		val_Y = train_Y[:, 0:1000]
		val_y = train_y[0:1000]

		print (val_X.shape, val_Y.shape, val_y.shape)

		train_X = train_X[:, 1000:]
		train_Y = train_Y[:, 1000:]
		train_y = train_y[1000:]

		mean = np.mean(train_X, axis = 1)
		train_X = normalize(train_X, mean)
		val_X = normalize(val_X, mean)
		test_X = normalize(test_X, mean)

		d = train_X.shape[0]
		K = train_Y.shape[0]
		W, b = initial(K, d, "He")

		params = {
			'n_batch': 100,
			'n_epochs': 50,
			'eta': 0.017453577972249945, # 0.010800662290914505,
			'momentum': 0.95,
			'decay': 0.1,
			'decay_gap': 8,
		}
		lambda_ = 0.0023292248102687557 # 0.002963774526491722
		
		Wstar, bstar, train_loss, val_loss, Wbest, bbest = miniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, 
																		lambda_, params, verbose = True, early_stop = True)
		x = [i + 1 for i in range(len(train_loss))]
		plt.plot(x, train_loss, label = 'train')
		plt.plot(x, val_loss, label = 'val')

		print ("Accuracy on test set (final): {}".format(computeAccuracy(test_X, test_y, Wstar, bstar)))
		print ("Accuracy on test set (best): {}".format(computeAccuracy(test_X, test_y, Wbest, bbest)))
		plt.legend()
		plt.show()

