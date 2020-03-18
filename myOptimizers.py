"""Some standard gradient-based stochastic optimizers.

These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.

These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays."""
from __future__ import absolute_import
from builtins import range

import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps

def unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(grad, x0, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
        if callback:
            _callback = lambda x, i, g: callback(unflatten(x), i, unflatten(g))
        else:
            _callback = None
        return unflatten(optimize(_grad, _x0, _callback, *args, **kwargs))

    return _optimize

@unflatten_optimizer
def sgd(grad, x, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        x = x + step_size * velocity
    return x

@unflatten_optimizer
def rmsprop(grad, x, callback=None, num_iters=100,
            step_size=0.1, gamma=0.9, eps=10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return x

@unflatten_optimizer
def adam(grad, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x

@unflatten_optimizer
def myAdam (grad, x, callback=None,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, diffeqList=None, probList=None, diffeqDiffList = None):
	diffeqDiffList.append(0.0)

	m = np.zeros(len(x))
	v = np.zeros(len(x))
	i = 0

	brkCount2 = 0
	brkCount2_iter = -1
	brkCount2_prob = -1

	brkCount3 = 0
	brkCount3_iter = -1
	brkCount3_prob = -1
	brk3Lim = -1

	brkCount4 = 0
	brkCount4_iter = -1
	brkCount4_prob = -1
	brk4Lim = -1
	brk4LimFrac = 3/2

	endLim = -1
	endLimFrac = 3/2

	k2 = 2 
	k3 = 4
	k4 = 8

	lessTenThou = False
	trainThresh = 20000


	while True :
		g = grad(x, i)
		i = i + 1
		if callback: 
			callback(x, i, g)
            
		m = (1 - b1) * g      + b1 * m  # First  moment estimate
		v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
		mhat = m / (1 - b1**(i + 1))    # Bias correction.
		vhat = v / (1 - b2**(i + 1))
		x = x - step_size*mhat/(np.sqrt(vhat) + eps)

		if i % 100 == 0 :
			if diffeqList and probList and len(diffeqList) > 1 :

				diffeqDiff = diffeqList[-1] - diffeqList[-2]
				diffeqDiffList.append (diffeqDiff)
				print ("diffeqDiff = " + str(diffeqDiff))

				if diffeqDiff < 0 :
					if np.abs(diffeqDiff) < 10**-4 :
						if brkCount4_iter == -1 :
							print ("Reached 10**-4 at " + str(i))
							brkCount4_iter = i
							brkCount4_prob = probList[-1]

							endLim = brkCount4_iter + k4 * (brkCount4_iter - brkCount3_iter)
							endLim3b2 = int(endLimFrac * brkCount4_iter)
							if endLim3b2 % 100 != 0 :
								endLim3b2 = endLim3b2 - 50

							if endLim > 2*brkCount4_iter :
								endLim = 2*brkCount4_iter
							elif endLim < endLim3b2 :
								endLim = endLim3b2

							if endLim < brk4Lim :
								endLim = 2*brk4Lim - endLim

							print ("endLim = " + str(endLim))


						brkCount4 = brkCount4 + 1 
						print ("brkCount4 = " + str(brkCount4))

					elif np.abs(diffeqDiff) < 10**-3 :
						if brkCount3_iter == -1 :
							print ("Reached 10**-3 at " + str(i))
							brkCount3_iter = i
							brkCount3_prob = probList[-1]

							brk4Lim = brkCount3_iter + k3 * (brkCount3_iter - brkCount2_iter)
							brk4Lim3b2 = int(brk4LimFrac * brkCount3_iter)
							if brk4Lim3b2 % 100 != 0 :
								brk4Lim3b2 = brk4Lim3b2 - 50

							if brk4Lim > 2*brkCount3_iter :
								brk4Lim = 2*brkCount3_iter
							elif brk4Lim < brk4Lim3b2 :
								brk4Lim = brk4Lim3b2

							if brk4Lim < brk3Lim :
								brk4Lim = 2*brk3Lim - brk4Lim

							print ("brk4Lim = " + str(brk4Lim))

						brkCount3 = brkCount3 + 1
						print ("brkCount3 = " + str(brkCount3))

					elif np.abs(diffeqDiff) < 10**-2 :
						if brkCount2_iter == -1 :
							print ("Reached 10**-2 at " + str(i))
							brkCount2_iter = i
							brkCount2_prob = probList[-1]

							brk3Lim = k2*brkCount2_iter

							print ("brk3Lim = " + str(brk3Lim))
							if brkCount2_iter < 10000 :
								lessTenThou = True
								print ("lessTenThou = " + str(lessTenThou))
							else :
								print ("lessTenThou = " + str(lessTenThou))

						brkCount2 = brkCount2 + 1 
						print ("brkCount2 = " + str(brkCount2))


			if not lessTenThou :
				if brkCount2_iter != -1 and brkCount3_iter == -1 and i > brk3Lim :
					break

				if brkCount3_iter != -1 and brkCount4_iter == -1 and i > brk4Lim :
					break 

				if brkCount4_iter != -1 : 
					if brkCount4 < 5 and i > endLim :
						break
					elif brkCount4 >= 5 :
						break
			else :
				if i >= 10000 :
					if i == 10000 :
						print ("At 10,000th iteration")
					else :
						print ("Beyond 10,000th iteration")
					if brkCount4_iter != -1 :
						print ("Hit 10**-4 already")
						print ("endLim = " + str(endLim))
						if endLim > trainThresh :
							print ("Going back to main logic")
							lessTenThou = False
						else :
							print ("Will eventually hit 10 of 10**-4 (or has hit)")
							if brkCount4 >= 10 :
								break
					elif brkCount3_iter != -1 :
						print ("Hit 10**-3 already")
						print ("brk4Lim = " + str(brk4Lim))
						if brk4Lim > trainThresh :
							print ("Going back to main logic")
							lessTenThou = False
						else :
							# It will eventually cause (brkCount4_iter != -1) to become True
							print ("It will eventually cause (brkCount4_iter != -1) to become True")
							pass
					else :
						# It will eventually cause (brkCount3_iter != -1) to become True
						print ("It will eventually cause (brkCount3_iter != -1) to become True")
						pass

			#if brkCount4 == 5 :
			#	break 

			print ("---------------------------------")


	print ("========================================")
	print ("brkCount2_iter = " + str(brkCount2_iter) + " brkCount2 = " + str(brkCount2) + " brkCount2_prob = " + str(brkCount2_prob))
	print ("brk3Lim = " + str(brk3Lim) + " brkCount3_iter = " + str(brkCount3_iter) + " brkCount3 = " + str(brkCount3) + " brkCount3_prob = " + str(brkCount3_prob))
	print ("brk4Lim = " + str(brk4Lim) + " brkCount4_iter = " + str(brkCount4_iter) + " brkCount4 = " + str(brkCount4) + " brkCount4_prob = " + str(brkCount4_prob))
	print ("endLim = " + str(endLim))

	return x