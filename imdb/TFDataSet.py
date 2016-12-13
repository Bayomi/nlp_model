import numpy as np

class TFDataSet():
	"""
	Input:
		x_train_vals
		x_train_labels
		x_test_vals
		x_test_labels
	"""
	def __init__(self, train_vals, train_labels, test_vals):
		self.train = self.getTrain(train_vals, train_labels)
		self.test = self.getTest(test_vals)

	def getTrain(self, train_vals, train_labels):
		train = TrainSet(train_vals, train_labels)
		return train

	def getTest(self, test_vals):
		test = TestSet(test_vals)
		return test

	def getValidation(self):
		print 'hey'

class TestSet():
	def __init__(self, test_vals):
		self._vals = test_vals


class TrainSet():
	def __init__(self, train_vals, train_labels):
		self._vals = train_vals
		self._labels = train_labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._num_examples = self._vals.shape[0]

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
	    
	   	start = self._index_in_epoch
	   	self._index_in_epoch += batch_size

	   	if self._index_in_epoch > self._num_examples:
	   		self._epochs_completed += 1
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._vals = self._vals[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch

		return self._vals[start:end], self._labels[start:end]

"""
#Format the training images to [0, 1) size
train_images, train_labels = getSample(60000)
train_images = train_images/255.0

#Format the testing images to [0, 1) size
test_images, test_labels = getTestingSample(10000)
test_images = test_images/255.0



myDS = TFDataSet(train_images, train_labels, test_images)
"""

