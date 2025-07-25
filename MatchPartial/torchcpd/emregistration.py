from __future__ import division
import numpy as np
import numbers
from warnings import warn
import scipy.spatial.distance
import torch


def initialize_sigma2(X, Y):
	"""
	Initialize the variance (sigma2).

	Attributes
	----------
	X: numpy array
		NxD array of points for target.

	Y: numpy array
		MxD array of points for source.

	Returns
	-------
	sigma2: float
		Initial variance.
	"""
	(N, D) = X.shape
	(M, _) = Y.shape
	diff = X[None, :, :] - Y[:, None, :]
	err = diff ** 2
	return err.sum() / (D * M * N)


def lowrankQS(G, beta, num_eig, eig_fgt=False):
	"""
	Calculate eigenvectors and eigenvalues of gaussian matrix G.

	!!!
	This function is a placeholder for implementing the fast
	gauss transform. It is not yet implemented.
	!!!

	Attributes
	----------
	G: numpy array
		Gaussian kernel matrix.

	beta: float
		Width of the Gaussian kernel.

	num_eig: int
		Number of eigenvectors to use in lowrank calculation of G

	eig_fgt: bool
		If True, use fast gauss transform method to speed up.
	"""

	# if we do not use FGT we construct affinity matrix G and find the
	# first eigenvectors/values directly

	if eig_fgt is False:
		S, Q = np.linalg.eigh(G)
		eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
		Q = Q[:, eig_indices]  # eigenvectors
		S = S[eig_indices]  # eigenvalues.

		return Q, S

	elif eig_fgt is True:
		raise Exception('Fast Gauss Transform Not Implemented!')


class EMRegistration(object):
	"""
	Expectation maximization point cloud registration.

	Attributes
	----------
	X: numpy array
		NxD array of target points.

	Y: numpy array
		MxD array of source points.

	TY: numpy array
		MxD array of transformed source points.

	sigma2: float (positive)
		Initial variance of the Gaussian mixture model.

	N: int
		Number of target points.

	M: int
		Number of source points.

	D: int
		Dimensionality of source and target points

	iteration: int
		The current iteration throughout registration.

	max_iterations: int
		Registration will terminate once the algorithm has taken this
		many iterations.

	tolerance: float (positive)
		Registration will terminate once the difference between
		consecutive objective function values falls within this tolerance.

	w: float (between 0 and 1)
		Contribution of the uniform distribution to account for outliers.
		Valid values span 0 (inclusive) and 1 (exclusive).

	q: float
		The objective function value that represents the misalignment between source
		and target point clouds.

	diff: float (positive)
		The absolute difference between the current and previous objective function values.

	P: numpy array
		MxN array of probabilities.
		P[m, n] represents the probability that the m-th source point
		corresponds to the n-th target point.

	Pt1: numpy array
		Nx1 column array.
		Multiplication result between the transpose of P and a column vector of all 1s.

	P1: numpy array
		Mx1 column array.
		Multiplication result between P and a column vector of all 1s.

	Np: float (positive)
		The sum of all elements in P.

	"""

	def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None, w=None, device='cuda', dtype=torch.float32, *args, **kwargs):
		if X.ndim != 2:
			raise ValueError(
				"The target point cloud (X) must be at a 2D numpy array.")

		if Y.ndim != 2:
			raise ValueError(
				"The source point cloud (Y) must be a 2D numpy array.")

		if X.shape[1] != Y.shape[1]:
			raise ValueError(
				"Both point clouds need to have the same number of dimensions.")

		if sigma2 is not None and (not isinstance(sigma2, torch.Tensor) or sigma2 <= 0):
			raise ValueError(
				"Expected a positive value for sigma2 instead got: {}".format(sigma2))

		if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
			raise ValueError(
				"Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
		elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
			warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
			max_iterations = int(max_iterations)

		if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
			raise ValueError(
				"Expected a positive float for tolerance instead got: {}".format(tolerance))

		if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
			raise ValueError(
				"Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

		# self.X = torch.from_numpy(X).type(dtype).to(device)
		# self.Y = torch.from_numpy(Y).type(dtype).to(device)
		if isinstance(X, np.ndarray):
			self.X = torch.from_numpy(X).type(dtype).to(device)
		else:
			self.X = X.type(dtype).to(device)

		if isinstance(Y, np.ndarray):
			self.Y = torch.from_numpy(Y).type(dtype).to(device)
		else:
			self.Y = Y.type(dtype).to(device)


		self.TY = self.Y.clone()
		self.sigma2 = (initialize_sigma2(self.X, self.Y) if sigma2 is None else sigma2)
		(self.N, self.D) = self.X.shape
		(self.M, _) = self.Y.shape
		self.tolerance = 0.001 if tolerance is None else tolerance
		self.w = 0.0 if w is None else w
		self.max_iterations = 100 if max_iterations is None else max_iterations
		self.iteration = 0
		self.diff = np.inf
		self.q = np.inf
		self.P = torch.zeros((self.M, self.N), device=device)
		self.Pt1 = torch.zeros((self.N,), device=device)
		self.P1 = torch.zeros((self.M,), device=device)
		self.PX = torch.zeros((self.M, self.D), device=device)
		self.Np = 0
		self.device = device
		self.dtype = dtype

	def register(self, callback=lambda **kwargs: None):
		"""
		Perform the EM registration.

		Attributes
		----------
		callback: function
			A function that will be called after each iteration.
			Can be used to visualize the registration process.

		Returns
		-------
		self.TY: numpy array
			MxD array of transformed source points.

		registration_parameters:
			Returned params dependent on registration method used.
		"""
		self.transform_point_cloud()
		while self.iteration < self.max_iterations and self.diff > self.tolerance:
			self.iterate()
			if callable(callback):
				kwargs = {'iteration': self.iteration,
				          'error': self.q, 'X': self.X, 'Y': self.TY}
				callback(**kwargs)

		return self.TY, self.get_registration_parameters()

	def get_registration_parameters(self):
		"""
		Placeholder for child classes.
		"""
		raise NotImplementedError(
			"Registration parameters should be defined in child classes.")

	def update_transform(self):
		"""
		Placeholder for child classes.
		"""
		raise NotImplementedError(
			"Updating transform parameters should be defined in child classes.")

	def transform_point_cloud(self):
		"""
		Placeholder for child classes.
		"""
		raise NotImplementedError(
			"Updating the source point cloud should be defined in child classes.")

	def update_variance(self):
		"""
		Placeholder for child classes.
		"""
		raise NotImplementedError(
			"Updating the Gaussian variance for the mixture model should be defined in child classes.")

	def iterate(self):
		"""
		Perform one iteration of the EM algorithm.
		"""
		self.P, self.Pt1, self.P1, self.Np, self.PX = expectation(self.X, self.TY, self.sigma2, self.D, self.w, self.M, self.N)
		self.maximization()
		self.iteration += 1

	def maximization(self):
		"""
		Compute the maximization step of the EM algorithm.
		"""
		self.update_transform()
		self.transform_point_cloud()
		self.update_variance()


def expectation(X, TY, sigma2, D, w, M, N):
	"""
	Compute the expectation step of the EM algorithm.
	"""
	dist = (X[None, :, :] - TY[:, None, :]) ** 2
	P = dist.sum(axis=2)  # (M, N)
	P = torch.exp(-P / (2 * sigma2))
	c = (2 * torch.pi * sigma2) ** (D / 2) * w / (1. - w) * M / N

	den = P.sum(axis=0, keepdims=True)
	den = torch.clamp(den, min=1e-7) + c

	P = P / den
	Pt1 = P.sum(axis=0)
	P1 = P.sum(axis=1)
	Np = P1.sum()
	PX = P @ X
	return P, Pt1, P1, Np, PX

