# noinspection PyInterpreter
import sys
from time import sleep
import warnings

import pylab as p
import scipy.optimize

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
# print("No Warning Shown")

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point
from itertools import product
import statistics

import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd import hessian
from torchdiffeq import odeint

import time
import datetime
from datetime import datetime as dtime

import webcolors

# Using of MLP's with classes for learning of dynamical system - Modified equation - Variable steps of time

# Maths parameters [adjust]

dyn_syst = "Pendulum"       # Dynamical system studied (choice between "Rigid Body", "SIR", "Stable", "Lotka-Volterra" and "Pendulum")
num_meth = "RK2"       # Choice of the numerical method ("Forward Euler", "MidPoint" and "RK2")
step_h = [0.1, 2.5]        # Interval where steps of time are selected for training
T_simul = 20                # Time for ODE's simulation
h_simul = 0.25              # Time step used for ODE's simulation
sigma = 0.0                 # Amplitude of noise for data's perturbation (Gaussian noise - standard deviation)

# AI parameters [adjust]

K_data = 10000         # Quantity of data
N_h = 1                # Steps of time selected for an initial datum
R = 2                  # Amplitude of data in space (i.e. space data will be selected in the box [-R,R]^d)
p_train = 0.8          # Proportion of data for training
N_terms = 1            # Number of terms in the perturbation (MLP's)
HL = 2                 # Hidden layers per MLP
zeta = 250             # Neurons per hidden layer
alpha = 2e-3           # Learning rate for gradient descent
Lambda = 1e-9          # Weight decay
BS = 100               # Batch size (for mini-batching)
N_epochs = 200         # Epochs
N_epochs_print = 20    # Epochs between two prints of the Loss value

print(150 * "-")
print("Using of MLP's with classes for learning of dynamical system - Modified equation - Variable steps of time")
print(150 * "-")

print("   ")
print(150 * "-")
print("Parameters:")
print(150 * "-")
print('    # Maths parameters:')
print('        - Dynamical system studied:', dyn_syst)
print('        - Numerical method:', num_meth)
print("        - Interval where steps of time are selected for training:", step_h)
print("        - Time for ODE's simulation:", T_simul)
print("        - Step time used for ODE's simulation:", h_simul)
print("        - Amplitude of noise for data's perturbation", sigma)
print("    # AI parameters:")
print("        - Data's number:", K_data)
print("        - Number of step times selected for an initial datum:", N_h)
print("        - Amplitude of data in space:", R)
print("        - Proportion of data for training:", format(p_train, '.1%'))
print("        - Numbers of terms in the perturbation (MLP's):", N_terms)
print("        - Hidden layers per MLP:", HL)
print("        - Neurons on each hidden layer:", zeta)
print("        - Learning rate:", format(alpha, '.2e'))
print("        - Weight decay:", format(Lambda, '.2e'))
print("        - Batch size (mini-batching for training):", BS)
print("        - Epochs:", N_epochs)
print("        - Epochs between two prints of the Loss value:", N_epochs_print)

# Computed parameters

# Dimension of the problem

if dyn_syst == "Rigid Body" or dyn_syst == "SIR" or dyn_syst == "Stable":
    d = 3
else:
    d = 2

# ODE's parameters

if dyn_syst == "Rigid Body":
    I1, I2, I3 = 1, 2, 3  # Moments of inertia
if dyn_syst == "SIR":
    R0, Tr = 3, 10  # R0: Basic reproduction rate, Tr: Recovering time
if dyn_syst == "Lotka-Volterra":
    beta11, beta12, beta21, beta22 = 1, 1, 1, 1  # Rate of reproduction/death of preys/predators


# Initial data for ODE's integration and study of trajectories

def y0_start(dyn_syst):
    """Gives the initial data (vector) for ODE's integration and study of trajectories
    Imputs:
    - dyn_syst: character string - Dynamical system studied"""
    if d == 2:
        if dyn_syst == "Lotka-Volterra":
            Y0_start = np.array([0.5, 1.5])
        else:
            Y0_start = np.array([0.1, 1.5])
    if d == 3:
        if dyn_syst == "Rigid Body":
            Y0_start = np.array([0.3, 0.3, 0.9])
        else:
            Y0_start = np.array([0.8, 0.1, 0.1])
    return Y0_start


# Measure of the domains (space & time) and computation of the density of data

if dyn_syst == "Rigid Body":
    meas_space = (4 / 3) * np.pi * (1.02 ** 3 - 0.98 ** 3)
else:
    meas_space = (2 * R) ** d

meas_time = np.log(step_h[1] / step_h[0])

data_density = (K_data * p_train) / (meas_space * meas_time)

print("        - Data density (data per volume unit - computed):", format(data_density, '.4E'))
print(150 * " ")


# Introduction of classes


class DynamicSyst:
    """Gives the vector field associated to the corresponding dynamical system"""

    def __init__(self):
        pass

    def f_array(self, t, y):
        """Returns the vector field associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable"""

        y = np.array(y).reshape(d, 1)
        z = np.zeros_like(y)

        if dyn_syst == "Rigid Body":
            z = np.array([(1 / I3 - 1 / I2) * y[1, 0] * y[2, 0], (1 / I1 - 1 / I3) * y[0, 0] * y[2, 0],
                          (1 / I2 - 1 / I1) * y[0, 0] * y[1, 0]])

        if dyn_syst == "SIR":
            z = np.array([-(R0 / Tr) * y[0, 0] * y[1, 0], (R0 / Tr) * y[0, 0] * y[1, 0] - (1 / Tr) * y[1, 0],
                          (1 / Tr) * y[1, 0]])

        if dyn_syst == "Stable":
            z = np.array([y[1, 0], -y[0, 0], -y[2, 0]])

        if dyn_syst == "Lotka-Volterra":
            z = np.array([y[0, 0] * (beta11 - beta12 * y[1, 0]), y[1, 0] * (beta21 * y[0, 0] - beta22)])

        if dyn_syst == "Pendulum":
            z = np.array([-np.sin(y[1, 0]), y[0, 0]])

        return z

    def f(self, t, y):
        """Returns the vector field associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable"""

        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)

        if dyn_syst == "Rigid Body":
            z[0, :] = (1 / I3 - 1 / I2) * y[1, :] * y[2, :]
            z[1, :] = (1 / I1 - 1 / I3) * y[0, :] * y[2, :]
            z[2, :] = (1 / I2 - 1 / I1) * y[0, :] * y[1, :]

        if dyn_syst == "SIR":
            z[0, :] = -(R0 / Tr) * y[0, :] * y[1, :]
            z[1, :] = (R0 / Tr) * y[0, :] * y[1, :] - (1 / Tr) * y[1, :]
            z[2, :] = (1 / Tr) * y[1, :]

        if dyn_syst == "Stable":
            z[0, :] = y[1, :]
            z[1, :] = -y[0, :]
            z[2, :] = -y[2, :]

        if dyn_syst == "Lotka-Volterra":
            z[0, :] = y[0, :] * (beta11 - beta12 * y[1, :])
            z[1, :] = y[1, :] * (beta21 * y[0, :] - beta22)

        if dyn_syst == "Pendulum":
            z[0, :] = -np.sin(y[1, :])
            z[1, :] = y[0, :]

        return z


class DynamicSystEDO(DynamicSyst):
    """Gives the vector field associated to a dynamical system so as to be used for ODE solving"""

    def fEDO(self, t, y):
        """Vector field of an ODE
        Inputs:
        - y: Array of shape (d,) - Space variable
        - t: Float - Time
        => Return an array of shape (d,)"""

        if type(y) != torch.Tensor and type(y) != np.ndarray:
            y = y._value
        y = torch.tensor(y)
        y = y.reshape(d, 1)
        y.requires_grad = True
        z = self.f(t, y)
        return (np.array(z, dtype=np.float64).reshape(d, ))

    def solvefEDO(self, y0, T=T_simul, h=h_simul, rel_tol=1e-9, abs_tol=1e-9):
        """Solving of the ODE y'=f(t,y) over an time interval [0,T] with step time h by using RK45 method
        Inputs:
        - y0: Array of shape (d,) - Initial data for ODE solving
        - T: Float - Duration of solving (default: T_simul)
        - h: Float - Step time (default: h_simul)
        - rel_tol: Float - Relative error tolerance (default: 1e-6)
        - abs_tol: Float - Absolute error tolerance (default: 1e-6)
        => Returns an array of shape (d,n) where n is the number of steps for numerical integration"""
        return solve_ivp(self.fEDO, (0, T), y0, method='RK45', t_eval=np.arange(0, T, h), rtol=rel_tol, atol=abs_tol).y

    def solvefEDOData(self, K=K_data, p=p_train, h_data=step_h):
        """production of a set of initial data y0 and final data y1 = flow-h(y0) associated to the dynamical system y'=f(t,y)
        Inputs:
        - K: Data's number (default: K_data)
        - p: Proportion of data for training (default: p_train)
        - h_data: List - Interval where steps of time are chosen for training (default: step_h)
        Denote K0 := int(p*K) the number of data for training
        => Returns the tuple (Y0_train,Y0_test,Y1_train,Y1_test,h_train,h_test) where:
            - Y0_train is a tensor of shape (d,K0) associated to the initial data for training
            - Y0_test is a tensor of shape (d,K-K0) associated to the initial data for test
            - Y1_train is a tensor of shape (d,K0) associated to the final data for training
            - Y1_test is a tensor of shape (d,K-K0) associated to the final data for test
            - h_train is a tensor of shape (1,K0) associated to data of steps of time for training
            - h_train is a tensor of shape (1,K-K0) associated to data of steps of time for test
            Each column of the tensor Y1_* correspopnds to the flow at h_* of the same column of Y0_*
            Initial data are uniformly chosen in [-R,R]^d (excepted for Rigid Body, in a spherical crown)"""

        start_time_data = time.time()

        print(" ")
        print(150 * "-")
        print("Data creation...")
        print(150 * "-")

        K0 = int(p * K)
        YY0 = np.random.uniform(low=-R, high=R, size=(d, K))
        # if num_meth == "RK2":
        #    beta = 0.4
        #    N = np.linalg.norm(YY0, ord=np.infty, axis=0)
        #    P = np.random.uniform(0, 1, size=(1, K))
        #    YY0 = R * YY0 / N * P ** beta
        YY1 = np.zeros((d, K))
        hh = np.exp(np.random.uniform(low=np.log(h_data[0]), high=np.log(h_data[1]), size=(1, K)))

        e1, e2, e3 = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])  # For SIR model (data selection)
        for k in range(K):
            YY0[:, k] = YY0[:, (k // N_h) * N_h]
            if dyn_syst == "Rigid Body":
                YY0[:, k] = YY0[:, k] / np.linalg.norm(YY0[:, k]) * np.random.uniform(0.98, 1.08)
            if dyn_syst == "SIR":
                v1 = np.random.uniform(0, 1)
                v2 = np.random.uniform(0, 1)
                q = np.abs(v1 - v2)
                s = q
                t = 0.5 * (v1 + v2 - q)
                u = 1 - 0.5 * (v1 + v2 + q)
                xi = 0.02 * np.random.uniform(low=-1, high=1, size=(3,))
                YY0[:, k] = s * e1 + t * e2 + u * e3 + xi
            print("  {} % \r".format(str(int(1000 * (k + 1) / K) / 10).rjust(3)), end="")
            YY1[:, k] = self.solvefEDO(y0=YY0[:, k], T=2 * hh[0, k], h=hh[0, k])[:, 1] + np.random.normal(loc=0,
                                                                                                          scale=sigma,
                                                                                                          size=(d,))

        Y0_train = torch.tensor(YY0[:, 0:K0])
        Y0_test = torch.tensor(YY0[:, K0:K])
        Y1_train = torch.tensor(YY1[:, 0:K0])
        Y1_test = torch.tensor(YY1[:, K0:K])
        h_train = torch.tensor(hh[:, 0:K0])
        h_test = torch.tensor(hh[:, K0:K])

        print("Computation time for data creation (h:min:s):",
              str(datetime.timedelta(seconds=int(time.time() - start_time_data))))
        return (Y0_train, Y0_test, Y1_train, Y1_test, h_train, h_test)


class EDONum:
    def solveEDO_num(self, f, y0, T=T_simul, h=h_simul, meth=num_meth):
        """Numerical integration of the ODE y'=f(t,y) over an time interval [0,T] with a step time h by using a specified numerical method
        Inputs:
        - f: Function - Vector field, has to be used by the function solve_ivp
        - y0: Array of shape (d,) - Starting point for the intergation of the ODE
        - T: Float - Upper bound of the time interval of integration (default: T_simul)
        - h: Float - Step time for the integration (default: h_simul)
        - meth: Character string - Numerical method used
        => Returns an array of shape (d,n) where n is the number of iterations with numerical integration"""
        Y = y0.reshape(d, 1)
        y = y0
        TT = np.arange(0, T, h)

        start_time_solveEDO_Num = time.time()

        for t in TT[1:]:
            if meth == "Forward Euler":
                y = y + h * f(t - h, y)
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)

            if meth == "MidPoint":
                def func_iter(x):
                    x = x.reshape(d, )
                    z = y + h * f(t - h, (x + y) / 2)
                    z = z.reshape(d,)
                    return z

                Niter = 15
                x = y
                for k in range(Niter):
                    x = func_iter(x)
                y = x
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)

            if meth == "RK2":
                y = y + h * f(t - h, y + (h / 2) * f(t - h / 2, y))
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)

        return Y

    def solveEDO_DOPRI5(self, f, y0, T=T_simul, h=h_simul):
        """Numerical integration of the ODE y'=f(t,y) over an time interval [0,T] with a step time h by using DOPRI5 method
        Inputs:
        - f: Function - Vector field, has to be used by the function solve_ivp
        - y0: Array of shape (d,) - Starting point for the intergation of the ODE
        - T: Float - Upper bound of the time interval of integration (default: T_simul)
        - h: Float - Step time for the integration (default: h_simul)
        => Returns an array of shape (d,n) where n is the number of iterations with numerical integration"""

        Y = y0.reshape(d, 1)
        y = y0
        TT = np.arange(0, T, h)

        # a = np.array([[1 / 4, 0, 0, 0, 0], [3 / 32, 9 / 32, 0, 0, 0], [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
        #               [439 / 216, -8, 3680 / 513, -845 / 4104, 0], [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]])
        # b = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
        a = np.array([[1 / 5, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0], [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
                      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
                      [9017 / 3168, - 355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
                      [35 / 384, 0, 500 / 1113, 125 / 192, - 2187 / 6784, 11 / 84]])
        b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, - 2187 / 6784, 11 / 84, 0])

        for t in TT[1:]:
            # f is an autonomous vector field
            k1 = f(t - h, y)
            k2 = f(t - h, y + h * a[0, 0] * k1)
            k3 = f(t - h, y + h * (a[1, 0] * k1 + a[1, 1] * k2))
            k4 = f(t - h, y + h * (a[2, 0] * k1 + a[2, 1] * k2 + a[2, 2] * k3))
            k5 = f(t - h, y + h * (a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3 + a[3, 3] * k4))
            k6 = f(t - h, y + h * (a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4 + a[4, 4] * k5))
            y = y + h * (b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6)
            Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)

        return Y


class NN(nn.Module, DynamicSyst):
    def __init__(self):
        super().__init__()
        #zeta_bis = int(zeta / N_terms)
        zeta_bis = int(zeta)
        if N_terms == 1:
            self.R = nn.ModuleList(
                [nn.Linear(d + 1, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
        if N_terms == 2:
            self.f1 = nn.ModuleList(
                [nn.Linear(d, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
            self.R = nn.ModuleList(
                [nn.Linear(d + 1, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
        if N_terms == 3:
            self.f1 = nn.ModuleList(
                [nn.Linear(d, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
            self.f2 = nn.ModuleList(
                [nn.Linear(d, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
            self.R = nn.ModuleList(
                [nn.Linear(d + 1, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
        if N_terms == 4:
            self.f1 = nn.ModuleList(
                [nn.Linear(d, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
            self.f2 = nn.ModuleList(
                [nn.Linear(d, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
            self.f3 = nn.ModuleList(
                [nn.Linear(d, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])
            self.R = nn.ModuleList(
                [nn.Linear(d + 1, zeta_bis), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [
                    nn.Linear(zeta_bis, d, bias=False)])

    def forward(self, x, h):
        """x is a tensor (space variables)
        h is a tensor or a float (data h used for training of scalar for numerical simulation)"""
        x = x.T
        h = torch.tensor(h).T
        x = x.float()
        x0 = self.f(0, x.T).T

        if num_meth == "Forward Euler":

            if N_terms == 2:
                x1 = x
                for i, module in enumerate(self.f1):
                    x1 = module(x1)
                x0 = x0 + h * x1

            if N_terms == 3:
                x1 = x
                for i, module in enumerate(self.f1):
                    x1 = module(x1)
                x2 = x
                for i, module in enumerate(self.f2):
                    x2 = module(x2)
                x0 = x0 + h * x1 + h ** 2 * x2

            if N_terms == 4:
                x1 = x
                for i, module in enumerate(self.f1):
                    x1 = module(x1)
                x2 = x
                for i, module in enumerate(self.f2):
                    x2 = module(x2)
                x3 = x
                for i, module in enumerate(self.f3):
                    x3 = module(x3)
                x0 = x0 + h * x1 + h ** 2 * x2 + h ** 3 * x3

            xR = torch.cat((x, h), dim=1)
            for i, module in enumerate(self.R):
                xR = module(xR)

            return (x0 + h ** N_terms * xR).T


        else:

            if N_terms == 2:
                x1 = x
                for i, module in enumerate(self.f1):
                    x1 = module(x1)
                x0 = x0 + h ** 2 * x1

            if N_terms == 3:
                x1 = x
                for i, module in enumerate(self.f1):
                    x1 = module(x1)
                x2 = x
                for i, module in enumerate(self.f2):
                    x2 = module(x2)
                x0 = x0 + h ** 2 * x1 + h ** 3 * x2

            if N_terms == 4:
                x1 = x
                for i, module in enumerate(self.f1):
                    x1 = module(x1)
                x2 = x
                for i, module in enumerate(self.f2):
                    x2 = module(x2)
                x3 = x
                for i, module in enumerate(self.f3):
                    x3 = module(x3)
                x0 = x0 + h ** 2 * x1 + h ** 3 * x2 + h ** 4 * x3

            xR = torch.cat((x, h), dim=1)
            for i, module in enumerate(self.R):
                xR = module(xR)

            return (x0 + h ** (N_terms + 1) * xR).T


class Train(NN, DynamicSystEDO):
    """Training of the neural network, depends on the numerical method chosen
    Choice of the numerical method:
        - Forward Euler
        - MidPoint
        - RK2"""

    def Loss(self, Y0, Y1, h, model, meth=num_meth):
        """Computes the Loss function between two series of data Y0 and Y1 according to the numerical method
        Inputs:
        - Y0: Tensor of shape (d,n)
        - Y1: Tensor of shape (d,n)
        - h: Tensor of shape (1,n)
        - model: Neural network which will be optimized
        - meth: Character string - Numerical method used in order to compute predicted values
        Computes a predicted value Y1hat which is a tensor of shape (d,n) and returns the mean squared error between Y1hat and Y1
        => Returns a tensor of shape (1,1)"""
        Y0 = torch.tensor(Y0, dtype=torch.float32)
        Y0.requires_grad = True
        Ymeth = torch.zeros_like(Y0)
        Ymeth.requires_grad = True
        h = torch.tensor(h, dtype=torch.float32)
        h.requires_grad = True
        if meth == "Forward Euler":
            Ymeth = Y0
            Y1hat = Y0 + h * model(Ymeth, h)
            loss = ((1 / h ** 4) * (Y1hat - Y1) ** 2).mean()
        if meth == "MidPoint":
            Ymeth = (Y0 + Y1) / 2
            Y1hat = Y0 + h * model(Ymeth, h)
            loss = ((1 / h ** 6) * (Y1hat - Y1) ** 2).mean()
        if meth == "RK2":
            Y1hat = Y0 + h * model(Y0 + (h / 2) * model(Y0, h), h)
            loss = ((1 / h ** 6) * (Y1hat - Y1) ** 2).mean()

        return loss

    def train(self, model, Data, K=K_data, p=p_train, Nb_epochs=N_epochs, Nb_epochs_print=N_epochs_print, BSZ=BS):
        """Makes the training on the data
        Inputs:
        - model: Neural network which will be optimized
        - Data: Tuple of tensors - Set of data created
        - K: Integer - Number of data
        - p: Float - Proportion of data used for training (default: p_train)
        - Nb_epochs: Integer - Number of epochs for training (dafault: N_epochs)
        - Nb_epochs_print: Integer - Number of epochs between two prints of the value of the Loss (default: N_epochs_print)
        - BSZ: Integer - size of the batches for mini-batching
        => Returns the lists Loss_train and Loss_test of the values of the Loss respectively for training and test,
        and best_model, which is the best apporoximation of the modified field computed"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        Y0_train = Data[0]
        Y0_test = Data[1]
        Y1_train = Data[2]
        Y1_test = Data[3]
        h_train = Data[4]
        h_test = Data[5]
        optimizer = optim.AdamW(model.parameters(), lr=alpha, betas=(0.9, 0.999), eps=1e-8, weight_decay=Lambda,
                                amsgrad=True)  # Algorithme AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Retient le meilleur modèle (celui qui minimise le mieux la loss)
        Loss_train = []  # List for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(Nb_epochs + 1):
            for ixs in torch.split(torch.arange(Y0_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                Y0_batch = Y0_train[:, ixs]
                Y1_batch = Y1_train[:, ixs]
                h_batch = h_train[:, ixs]
                loss_train = self.Loss(Y0_batch, Y1_batch, h_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(Y0_test, Y1_test, h_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % Nb_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                print('    Step', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =',
                      format(loss_test, '.4E'))
        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):",
              str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        return (Loss_train, Loss_test, best_model)


class Integrate(Train, EDONum):

    def integrate(self, Ltr, Lte, model, name, save_fig, save_list, Nb_epochs=N_epochs):
        """Prints the values of the Loss along the epochs, trajectories and errors.
        Inputs:
        - Ltr: List containing the values of Loss_train along the epochs
        - Lte: List containing the values of Loss_test along the epochs
        - model: Best model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure
        - save_list: Boolean - Saves the list of times and corresponding local errors or not
        - Nb_epochs: Number of epochs for training"""

        def write_size():
            """Changes the size of writings on all windows"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(fontsize=7)
            pass

        def write_size3D():
            """Changes the size of writings on all windows - 3d variant"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            axes.zaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            axes.zaxis.set_tick_params(labelsize=7)
            plt.legend(fontsize=7)
            pass

        start_time_integrate = time.time()

        print(" ")
        print(150 * "-")
        print("Integration...")
        print(150 * "-")

        fig = plt.figure()

        ax = fig.add_subplot(2, 1, 2)
        plt.plot(range(Nb_epochs + 1), Ltr, color='green', label='$Loss_{train}$')
        plt.plot(range(Nb_epochs + 1), Lte, color='red', label='$Loss_{test}$')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.title('Evolution of the Loss function (MLP)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        write_size()

        def Fhat(t, y):
            """Vector fiekd learned with the neural network
            Inputs:
            - t: Float - Time
            - y: Array of shape (d,) - Space variable"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            h_tensor = torch.tensor([[h_simul]])
            z = model(y, h_tensor)
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        TT = np.arange(0, T_simul, h_simul)

        # Integration with RK45 (good approximation of exact flow)
        start_time_RK45 = time.time()
        Y_exact_RK45 = self.solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=h_simul, rel_tol=1e-15, abs_tol=1e-15)
        print("Integration time of ODE with RK45 (one trajectory - h:min:s):",
              datetime.timedelta(seconds=time.time() - start_time_RK45))

        # Integration with numerical method

        start_time_exact = time.time()
        Y_exact_meth = EDONum().solveEDO_num(DynamicSystEDO().fEDO, y0=y0_start(dyn_syst), T=T_simul, h=h_simul,
                                             meth=num_meth)
        print("Integration time of ODE with exact field - " + num_meth + " (one trajectory - h:min:s):",
              str(datetime.timedelta(seconds=time.time() - start_time_exact)))

        start_time_app = time.time()
        Y_app_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=h_simul, meth=num_meth)
        print("Integration time of ODE with learned field - " + num_meth + " (one trajectory - h:min:s):",
              str(datetime.timedelta(seconds=time.time() - start_time_app)))

        print("   ")
        # Error computation between trajectory ploted with f for RK45 and f for numerical method
        err_f = np.array([np.linalg.norm((Y_exact_RK45 - Y_exact_meth)[:, i]) for i in
                          range((Y_exact_RK45 - Y_exact_meth).shape[1])])
        Err_f = np.linalg.norm(err_f, np.infty)
        print("Error between trajectories ploted with f for RK45 with f for", num_meth, ":", format(Err_f, '.4E'))

        # Error computation between trajectory ploted with f for RK45 and f_app for numerical method
        err_meth = np.array([np.linalg.norm((Y_exact_RK45 - Y_app_meth)[:, i]) for i in range((Y_exact_RK45 - Y_app_meth).shape[1])])
        Err_meth = np.linalg.norm(err_meth, np.infty)
        print("Error between trajectories ploted with f for RK45 with f_app for", num_meth, ":",
              format(Err_meth, '.4E'))

        if d == 2:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(Y_exact_RK45[0, :], Y_exact_RK45[1, :], color='black', linestyle='dashed', label="$\phi_{nh}^f(y_0)$")
            plt.plot(Y_app_meth[0, :], Y_app_meth[1, :], color='green', label="$(\Phi_{h}^{f_{app}})^n(y_0)$")
            plt.plot(Y_exact_meth[0, :], Y_exact_meth[1, :], color='red', label="$(\Phi_{h}^{f})^n(y_0)$")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.yscale('log')
            plt.plot(TT, err_f, color="blue", label="$| (\Phi_{h}^{f})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.plot(TT, err_meth, color="orange", label="$| (\Phi_{h}^{f_{app}})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            if save_list == True:
                torch.save(TT,"Integrate_Time_RKmodif_Comparison_K_"+str(K))
                torch.save(err_f,"Integrate_Error_f_RKmodif_Comparison_K_"+str(K_data))
                torch.save(err_meth,"Integrate_Error_f_app_RKmodif_Comparison_K_"+str(K))

        if d == 3:
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            ax.plot(Y_exact_RK45[0, :], Y_exact_RK45[1, :], Y_exact_RK45[2, :], color='black', linestyle="dashed",
                    label="$\phi_{nh}^f(y_0)$")
            ax.plot(Y_app_meth[0, :], Y_app_meth[1, :], Y_app_meth[2, :], color='green', linewidth=1,
                    label="$(\Phi_{h}^{f_{app}})^n(y_0)$")
            ax.plot(Y_exact_meth[0, :], Y_exact_meth[1, :], Y_exact_meth[2, :], color='red', linewidth=1,
                    label="$(\Phi_{h}^{f})^n(y_0)$")
            ax.legend()
            ax.set_xlabel('$y_1$')
            ax.set_ylabel('$y_2$')
            ax.set_zlabel('$y_3$')
            plt.title("Trajectories")
            write_size3D()

            ax = fig.add_subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.plot(TT, err_f, color="blue", label="$| (\Phi_{h}^{f})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.plot(TT, err_meth, color="orange", label="$| (\Phi_{h}^{f_{app}})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.ylabel("local error")
            plt.yscale('log')
            ax.legend()
            plt.grid()
            write_size()

            if save_list == True:
                torch.save(TT,"Integrate_Time_RKmodif_Comparison_K_"+str(K))
                torch.save(err_f,"Integrate_Error_f_RKmodif_Comparison_K_"+str(K_data))
                torch.save(err_meth,"Integrate_Error_f_app_RKmodif_Comparison_K_"+str(K))


        f = plt.gcf()
        dpi = f.get_dpi()
        h, w = f.get_size_inches()
        f.set_size_inches(h * 1.7, w * 1.7)
        plt.show()

        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))

        print("Computation time for integration (h:min:s):",
              str(datetime.timedelta(seconds=int(time.time() - start_time_integrate))))

        pass


class Trajectories(Integrate):
    def traj(self, model, name, save_fig, save_list):
        """Prints the global errors according to the step of the numerical method
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure
        - save_list: Boolean -Saves the list of time steps and corresponding global errors"""

        HH = np.exp(np.linspace(np.log(step_h[0]), np.log(step_h[1]), 15))
        ERR_f = []  # Global errors between exact flow and numerical flow with f
        ERR_meth = []  # Global errors between exact flow and numerical flow with f_app
        HH_f = []  # Time steps for errors between exact flow and numerical flow with f
        HH_meth = []  # Time steps for errors between exact flow and numerical flow with f_app

        for hh in HH:
            print(" h= {} \r".format(format(hh, '.4E')), end="")

            def Fhat(t, y):
                """Vector field learned with the neural network
                Inpiuts:
                - t: Float - Time
                - y: Array of shape (d,) - Space variable"""

                y = torch.tensor(y).reshape(d, 1)
                y.requires_grad = True
                h_tensor = torch.tensor([[hh]]).float()
                z = model(y, h_tensor)
                z = z.detach().numpy()
                z = np.array(z, dtype=np.float64)
                return z.reshape(d, )

            # Integration with RK45 (approximation of the exact flow)
            Y_exact_RK45 = self.solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=hh, rel_tol=1e-15, abs_tol=1e-15)
            norm_sol = np.linalg.norm(
                np.array([np.linalg.norm((Y_exact_RK45)[:, i]) for i in range((Y_exact_RK45).shape[1])]),
                np.infty)  # Norm of the exact solution

            # Integration with the numerical method used
            Y_exact_meth = EDONum().solveEDO_num(DynamicSystEDO().fEDO, y0=y0_start(dyn_syst), T=T_simul, h=hh,
                                                 meth=num_meth)
            Y_app_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth)

            # Computation of the error between the trajectory ploted with f and RK45 and the trajectory ploted with f
            # and the numerical method chosen
            err_f = np.array([np.linalg.norm((Y_exact_RK45 - Y_exact_meth)[:, i]) for i in
                              range((Y_exact_RK45 - Y_exact_meth).shape[1])])
            Err_f = np.linalg.norm(err_f, np.infty) / norm_sol
            if Err_f < 1:
                ERR_f = ERR_f + [Err_f]
                HH_f = HH_f + [hh]

            # Computation of the error between the trajectory ploted with f and RK45 and the trajectory ploted with f_app
            # and the numerical method chosen
            err_meth = np.array([np.linalg.norm((Y_exact_RK45 - Y_app_meth)[:, i]) for i in
                                 range((Y_exact_RK45 - Y_app_meth).shape[1])])
            Err_meth = np.linalg.norm(err_meth, np.infty) / norm_sol
            if Err_meth < 1:
                ERR_meth = ERR_meth + [Err_meth]
                HH_meth = HH_meth + [hh]

            # if num_meth == "Forward Euler":
            #     Y_app_star_meth = EDONum().solveEDO_num(Fhat_star, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth)
            #
            #     # Calcul de l'erreur entre la trajectoire tracée avec f et RK45 et celle tracée avec f_app* et la méthode numérique
            #     err_star_meth = np.array([np.linalg.norm((Y_exact_RK45 - Y_app_star_meth)[:, i]) for i in range((Y_exact_RK45 - Y_app_star_meth).shape[1])])
            #     Err_star_meth = np.linalg.norm(err_star_meth, np.infty)
            #     ERR_star_meth = ERR_star_meth + [Err_star_meth]

        if num_meth == "Forward Euler":
            plt.figure()
            plt.title("Error between trajectories with " + num_meth)
            if len(ERR_f) > 0:
                plt.scatter(HH_f, ERR_f, label=num_meth + " - $f$", marker="s", color="red")
                torch.save(HH_f,"Trajectories_Time_steps_f_RKmodif_Comparison_K_50000_Nh_5")
                torch.save(ERR_f, "Trajectories_Errors_f_RKmodif_Comparison_K_50000_Nh_5")
            if len(ERR_meth) > 0:
                plt.scatter(HH_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
                if save_list == True:
                    torch.save(HH_meth, "Trajectories_Time_steps_f_app_RKmodif_Comparison_K_50000_Nh_5")
                    torch.save(ERR_meth, "Trajectories_Errors_f_app_RKmodif_Comparison_K_50000_Nh_5")
            # plt.scatter(HH, ERR_star_meth, label="$|\phi^{f,RK45}_{nh}(y_0) - (\Phi^{f_{app}^*}_{h})^n(y_0)|$", marker="s",color="orange")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time step")
            plt.ylabel("Global error")
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        else:
            plt.figure()
            plt.title("Error between trajectories " + num_meth)
            if len(ERR_f) > 0:
                plt.scatter(HH_f, ERR_f, label=num_meth + " - $f$", marker="s", color="red")
            if len(ERR_meth) > 0:
                plt.scatter(HH_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
                if save_list == True:
                    torch.save(HH_meth, "Trajectories_Time_steps_f_app_RKmodif_Comparison_K_50000_Nh_5")
                    torch.save(ERR_meth, "Trajectories_Errors_f_app_RKmodif_Comparison_K_50000_Nh_5")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time step")
            plt.ylabel("Global error")
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        pass


class Invariant(Integrate):
    def inv_evolution(self, model, name, save_fig):
        """Gives the evolution of the error over the invariants over the numerical flows with these numerical methods:
        - If num_meth is "Forward Euler":
            -> Forward Euler with f
            -> Forward Euler with f_app
            -> DOPRI5 with f
        - If num_meth is "RK2":
            -> RK2 with f
            -> RK2 with f_app
            -> DOPRI5 with f
        - If num_meth is "MidPoint":
            -> MidPoint with f
            -> MidPoint with f_app
            -> RK2 with f
            -> DOPRI5 with f
        Only available for Pendulum and Rigid Body system.
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential namle of the graph
        - save_fig: Boolean - Saves the figure or not"""

        def f_app(x, h):
            """Learned vector field
            Inputs:
            - x: Array or tensor of shape (d,1)
            - h: Float - Step time of the numerical method"""
            x = torch.tensor(x).reshape(d, 1)
            h = torch.tensor([[h]]).float()
            y = model(x, h)
            y = y.detach().numpy()
            return y

        def Fhat(t, y):
            """Vector field learned with the neural network
            Inputs:
            - t: Float - Time
            - y: Array of shape (d,) - Space variable"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            h_tensor = torch.tensor([[h_simul]]).float()
            z = model(y, h_tensor)
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        if dyn_syst == "Pendulum" or dyn_syst == "Rigid Body":

            TT = np.arange(0, T_simul, h_simul)

            def Inv(x):
                """Invariant
                Input:
                - x: Array of shape (d,n) - Space Variable
                returns an array of shape (n,) containing the evaluations of x with the invariant"""
                if dyn_syst == "Pendulum":
                    return 1 - np.cos(x[1, :]) + 0.5 * x[0, :] ** 2
                if dyn_syst == "Rigid Body":
                    return 0.5 * x[0, :] ** 2 + 0.5 * x[1, :] ** 2 + 0.5 * x[2, :] ** 2

            # Integration with RK45 (approximation of the exact flow)
            Y_exact = DynamicSystEDO().solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=h_simul, rel_tol=1e-15,
                                                 abs_tol=1e-15)
            Inv_exact = Inv(Y_exact)

            # Integration with DOPRI5 and computation of error between exact flow and solution with DOPRI5
            Y_DOPRI5 = EDONum().solveEDO_DOPRI5(DynamicSystEDO().fEDO, y0=y0_start(dyn_syst), T=T_simul, h=h_simul)
            Inv_DOPRI5 = Inv(Y_DOPRI5)
            Err_Inv_DOPRI5 = np.abs(Inv_DOPRI5 - Inv_exact)

            # Integration with RK2 (only if the numerical method selected is not RK2)
            if num_meth != "RK2":
                Y_RK2 = EDONum().solveEDO_num(DynamicSystEDO().fEDO, y0_start(dyn_syst), T=T_simul, h=h_simul,
                                              meth="RK2")
                Inv_RK2 = Inv(Y_RK2)
                Err_Inv_RK2 = np.abs(Inv_RK2 - Inv_exact)

            # Integration with the numerical method which is selected
            Y_exact_meth = EDONum().solveEDO_num(DynamicSystEDO().fEDO, y0=y0_start(dyn_syst), T=T_simul, h=h_simul,
                                                 meth=num_meth)
            Inv_exact_meth = Inv(Y_exact_meth)
            Err_Inv_exact_meth = np.abs(Inv_exact_meth - Inv_exact)

            # Integration with the numerical method with f_app
            Y_app_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=h_simul, meth=num_meth)
            Inv_app_meth = Inv(Y_app_meth)
            Err_Inv_app_meth = np.abs(Inv_app_meth - Inv_exact)

            plt.figure()
            if dyn_syst == "Pendulum":
                plt.title("Error for Hamiltonian")
            if dyn_syst == "Rigid Body":
                plt.title("Error for Casimir invariant")
            plt.plot(TT, Err_Inv_app_meth, label=num_meth + " - $f_{app}$", color="green")
            plt.plot(TT, Err_Inv_exact_meth, label=num_meth + " - $f$", color="red")
            plt.plot(TT, Err_Inv_DOPRI5, label="DOPRI5 - $f$", color="orange")
            # if num_meth != "RK2":
            # plt.plot(TT, Err_Inv_RK2, label="Composition - $f_{app}$", color="magenta")
            plt.xlabel("Time")
            plt.ylabel("Error for invariants")
            plt.legend()
            plt.grid()
            plt.yscale('log')
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

            pass

        else:
            print("Only available for Pendulum and Rigid Body system.")

            pass


class TimeCompute(Integrate):
    def timecompute(self, model, name, save_fig, save_list):
        """Prints the points (Time computation, Accuracy) with these numerical methods:
        - If num_meth is "Forward Euler":
            -> Forward Euler with f
            -> Forward Euler with f_app
            -> DOPRI5 with f
        - If num_meth is "RK2":
            -> RK2 with f
            -> RK2 with f_app
            -> DOPRI5 with f
        - If num_meth is "MidPoint":
            -> MidPoint with f
            -> MidPoint with f_app
            -> RK2 with f
            -> DOPRI5 with f
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential namle of the graph
        - save_fig: Boolean - Saves the figure or not
        - save_list: Boolean - Saves the lists of computational times and corresponding global errors"""

        def f_app(x, h):
            """Learned vector field
            Inputs:
            - x: Array or tensor of shape (d,1)
            - h: Float - Step time of the numerical method"""
            x = torch.tensor(x).reshape(d, 1)
            h = torch.tensor([[h]]).float()
            y = model(x, h)
            y = y.detach().numpy()
            return y

        HH = np.exp(np.linspace(np.log(step_h[0]), np.log(step_h[1]), 10))
        HH_bis = np.exp(np.linspace(np.log(step_h[1]), np.log(4 * step_h[1]), 5))[1:]

        ERR_f = []  # Global errors between exact flow and numerical flow computed with f via num_meth
        ERR_meth = []  # Global errors between exact flow and numerical flow computed with f_app via num_meth
        ERR_comp = []  # Global errors between exact flow and numerical flow computed with f_app via composition for Euler
        ERR_RK2 = []  # Global errors between exact flow and numerical flow computed with f via RK2 (Only if num_meth is not RK2)
        ERR_DOPRI5 = []  # Global errors between exact flow and numerical flow computed with f via DOPRI5 (Runge-Kutta of order 5)
        ERR_DOPRI5_bis = []  # Global errors between exact flow and numerical flow computed with f via DOPRI5 (Runge-Kutta of order 5) for larger steps of time

        Time_f = []  # Time of computation of each trajectory via num_meth for f
        Time_meth = []  # Time of computation of each trajectory via num_meth for f_app
        Time_comp = []  # Time of computation of each trajectory via composition method for f_app
        Time_RK2 = []  # Time of computation of each trajectory via RK2 for f (only if num_meth is not RK2)
        Time_DOPRI5 = []  # Time of computation of each trajectory via DOPRI5 for f
        Time_DOPRI5_bis = []  # Time of computation of each trajectory via DOPRI5 for f (larger steps of time)

        if num_meth == "RK2":
            print("Integration with " + num_meth + " and DOPRI5...")
        else:
            print("Integration with " + num_meth + " and DOPRI5...")
        for hh in HH:
            print(" h= {} \r".format(format(hh, '.4E')), end="")

            def Fhat(t, y):
                """Vector field learned with the neural network
                Inputs:
                - t: Float - Time
                - y: Array of shape (d,) - Space variable"""
                y = torch.tensor(y).reshape(d, 1)
                y.requires_grad = True
                h_tensor = torch.tensor([[hh]]).float()
                z = model(y, h_tensor)
                z = z.detach().numpy()
                z = np.array(z, dtype=np.float64)
                return z.reshape(d, )

            alpha_comp = 1 / (2 - 2 ** (1 / 3))
            beta_comp = 1 - 2 * alpha_comp

            def Fhat_2(t, y):
                """Vector field learned with the neural network
                Inputs:
                - t: Float - Time
                - y: Array of shape (d,) - Space variable"""
                y = torch.tensor(y).reshape(d, 1)
                y.requires_grad = True
                h_tensor = torch.tensor([[hh]]).float()
                z = alpha_comp * model(y, alpha_comp * h_tensor)
                z = z.detach().numpy()
                z = np.array(z, dtype=np.float64)
                return z.reshape(d, )

            def Fhat_3(t, y):
                """Vector field learned with the neural network
                Inputs:
                - t: Float - Time
                - y: Array of shape (d,) - Space variable"""
                y = torch.tensor(y).reshape(d, 1)
                y.requires_grad = True
                h_tensor = torch.tensor([[hh]]).float()
                z = beta_comp * model(y, beta_comp * h_tensor)
                z = z.detach().numpy()
                z = np.array(z, dtype=np.float64)
                return z.reshape(d, )

            # Integration with RK45 (approximation of the exact flow)
            Y_exact = DynamicSystEDO().solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=hh, rel_tol=1e-15, abs_tol=1e-15)
            norm_sol = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]),
                                      np.infty)  # Norm of the exact solution

            # Integration with DOPRI5 and computation of error between exact flow and solution with DOPRI5
            time_int_DOPRI5 = time.time()
            Y_DOPRI5 = EDONum().solveEDO_DOPRI5(DynamicSystEDO().fEDO, y0=y0_start(dyn_syst), T=T_simul, h=hh)
            err_DOPRI5 = np.array(
                [np.linalg.norm((Y_exact - Y_DOPRI5)[:, i]) for i in range((Y_exact - Y_DOPRI5).shape[1])])
            Err_DOPRI5 = np.linalg.norm(err_DOPRI5, np.infty) / norm_sol
            if Err_DOPRI5 < 1:
                ERR_DOPRI5 = ERR_DOPRI5 + [Err_DOPRI5]
                Time_DOPRI5 = Time_DOPRI5 + [time.time() - time_int_DOPRI5]

            # Integration with RK2 (only if the numerical method selected is not RK2)
            if num_meth != "RK2":
                time_int_RK2 = time.time()
                Y_RK2 = EDONum().solveEDO_num(DynamicSystEDO().fEDO, y0_start(dyn_syst), T=T_simul, h=hh, meth="RK2")
                Time_RK2 = Time_RK2 + [time.time() - time_int_RK2]

            # Integration with the numerical method which is selected
            time_int_f = time.time()
            Y_exact_meth = EDONum().solveEDO_num(DynamicSystEDO().fEDO, y0=y0_start(dyn_syst), T=T_simul, h=hh,
                                                 meth=num_meth)
            err_f = np.array(
                [np.linalg.norm((Y_exact - Y_exact_meth)[:, i]) for i in range((Y_exact - Y_exact_meth).shape[1])])
            Err_f = np.linalg.norm(err_f, np.infty) / norm_sol
            if Err_f < 1:
                ERR_f = ERR_f + [Err_f]
                Time_f = Time_f + [time.time() - time_int_f]

            # Integration with the composition method (only if num_meth = "MidPoint")
            if num_meth == "MidPoint":
                time_int_comp = time.time()
                Y_exact_comp = EDONum().solveEDO_num(f=Fhat_2, y0=y0_start(dyn_syst), f_bis=Fhat_3, T=T_simul, h=hh,
                                                     meth="composition_MP")
                err_comp = np.array(
                    [np.linalg.norm((Y_exact - Y_exact_comp)[:, i]) for i in range((Y_exact - Y_exact_comp).shape[1])])
                Err_comp = np.linalg.norm(err_comp, np.infty) / norm_sol
                # if Err_comp < 1:
                ERR_comp = ERR_comp + [Err_comp]
                Time_comp = Time_comp + [time.time() - time_int_comp]

            # Integration with the numerical method with f_app
            time_int_meth = time.time()
            Y_app_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth)
            err_meth = np.array(
                [np.linalg.norm((Y_exact - Y_app_meth)[:, i]) for i in range((Y_exact - Y_app_meth).shape[1])])
            Err_meth = np.linalg.norm(err_meth, np.infty) / norm_sol
            if Err_meth < 1:
                ERR_meth = ERR_meth + [Err_meth]
                Time_meth = Time_meth + [time.time() - time_int_meth]

            # Computation of error between trajectory ploted with f via RK45 and f via RK2 (only if num_meth is not RK2)
            if num_meth != "RK2":
                err_RK2 = np.array([np.linalg.norm((Y_exact - Y_RK2)[:, i]) for i in range((Y_exact - Y_RK2).shape[1])])
                Err_RK2 = np.linalg.norm(err_RK2, np.infty) / norm_sol
                ERR_RK2 = ERR_RK2 + [Err_RK2]

        print("Integration with DOPRI5 (larger time steps)...")
        for hh in HH_bis:
            print(" h= {} \r".format(format(hh, '.4E')), end="")

            # Integration with RK45 (approximation of the exact flow)
            Y_exact = DynamicSystEDO().solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=hh, rel_tol=1e-15, abs_tol=1e-15)
            norm_sol = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]),
                                      np.infty)  # Norm of the exact solution

            # Integration with DOPRI5 and computation of error between exact flow and solution with DOPRI5
            time_int_DOPRI5 = time.time()
            Y_DOPRI5 = EDONum().solveEDO_DOPRI5(DynamicSystEDO().fEDO, y0=y0_start(dyn_syst), T=T_simul, h=hh)
            err_DOPRI5 = np.array(
                [np.linalg.norm((Y_exact - Y_DOPRI5)[:, i]) for i in range((Y_exact - Y_DOPRI5).shape[1])])
            Err_DOPRI5 = np.linalg.norm(err_DOPRI5, np.infty) / norm_sol
            if Err_DOPRI5 < 1:
                ERR_DOPRI5 = ERR_DOPRI5 + [Err_DOPRI5]
                Time_DOPRI5 = Time_DOPRI5 + [time.time() - time_int_DOPRI5]

        plt.figure()
        plt.title("Computation time vs Error between trajectories")
        if len(Time_f) > 0:
            plt.scatter(Time_f, ERR_f, label=num_meth + " - $f$", marker="s", color="red")
            if save_list == True:
                torch.save(Time_f, "Time_Time_f_RKmodif_Comparison")
                torch.save(ERR_f, "Time_Error_f_RKmodif_Comparison")
        plt.scatter(Time_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
        if save_list == True:
            torch.save(Time_meth, "Time_Time_f_app_RKmodif_Comparison")
            torch.save(ERR_meth, "Time_Error_f_app_RKmodif_Comparison")
        if len(Time_comp) > 0 and num_meth == "MidPoint":
            plt.scatter(Time_comp, ERR_comp, label="Composition - $f_{app}$", marker="s", color="black")
        plt.scatter(Time_DOPRI5, ERR_DOPRI5, label="DOPRI5 - $f$", marker="s", color="orange")
        plt.scatter(Time_DOPRI5_bis, ERR_DOPRI5_bis, marker="s", color="orange")
        # if num_meth != "RK2":
        # plt.scatter(Time_RK2, ERR_RK2, label="RK2 - $f$", marker="s", color="magenta")
        plt.xlabel("Computation time (s)")
        plt.ylabel("Global error")
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        plt.show()

        pass


class ModelEval:
    def model_eval(self , model):
        """Compuation of the averaged time for evaluation of the function f_app
        - model: torch.nn - Model which has to be evaluated"""

        def f_app(x, h):
            """Learned vector field
            Inputs:
            - x: Array or tensor of shape (d,1)
            - h: Float - Step time of the numerical method"""
            x = torch.tensor(x).reshape(d, 1)
            h = torch.tensor([[h]]).float()
            y = model(x, h)
            y = y.detach().numpy()
            return y

        Y_eval = np.random.uniform(low = -R , high = R , size = (d,1000))
        h_eval = np.exp(np.random.uniform(low = np.log(step_h[0]) , high = np.log(step_h[1]) , size = (1,1000)))
        Time_Eval , Time_Solve_ODE = [] , []

        for k in range(Y_eval.shape[1]):
            time_start = time.time()
            u = f_app(Y_eval[:,k] , h_eval[0,k])
            delta_time = time.time() - time_start
            Time_Eval = Time_Eval + [delta_time]

        for k in range(int(Y_eval.shape[1] / 50)):
            hh = h_simul

            def Fhat(t, y):
                """Vector field learned with the neural network
                Inputs:
                - t: Float - Time
                - y: Array of shape (d,) - Space variable"""
                y = torch.tensor(y).reshape(d, 1)
                y.requires_grad = True
                h_tensor = torch.tensor([[hh]]).float()
                z = model(y, h_tensor)
                z = z.detach().numpy()
                z = np.array(z, dtype=np.float64)
                return z.reshape(d, )

            time_start = time.time()
            Y_app_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth)
            delta_time = time.time() - time_start
            Time_Solve_ODE = Time_Solve_ODE + [delta_time]



        print("Averaged time for evaluation (h:min:s): "+ str(datetime.timedelta(seconds=(statistics.mean(Time_Eval)))) + " +/- " + str(datetime.timedelta(seconds=(statistics.stdev(Time_Eval)))/np.sqrt(Y_eval.shape[1])))
        print("Averaged time for ODE solving (h:min:s): " + str(datetime.timedelta(seconds=(statistics.mean(Time_Solve_ODE)))) + " +/- " + str(datetime.timedelta(seconds=(statistics.stdev(Time_Solve_ODE))) / np.sqrt(Y_eval.shape[1])))
        pass


class Convergence(Integrate):
    def curves(self, model, name, save_fig):
        """Prints the curves of convergence
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        def f_app(x, h):
            """Learned vector field
            Inputs:
            - x: Array or tensor of shape (d,1)
            - h: Float - Step of time of the numerical method"""
            x = torch.tensor(x).reshape(d, 1)
            h = torch.tensor([[h]]).float()
            y = model(x, h)
            y = y.detach().numpy()
            return y

        def f_modif(x, h, ord):
            """Modified vector field (theory), depending on x, h, and for an order ord
            Inputs:
            - x: Array orn tensor of shape (d,1) - Space variable
            - h: Float - Step time for the numerical method
            - ord - Integer - Order chosen to develop the modifdied field"""
            x = np.array(x).reshape(d, 1)

            def ff(x):
                """First term of the theoretical modified vector field
                Input:
                x - Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = self.f_array(0, x)
                return y.reshape(d, 1)

            def dff(x):
                """Gradient of ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = jacobian(ff)(x).reshape(d, d)
                return y

            def dfff(x):
                """Gradient of ff(x) x f(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = dff(x) @ ff(x)
                return y

            def dfff2(x):
                """Gradient of dfff(x) x ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = jacobian(dfff)(x).reshape(d, d)
                y = y @ ff(x)
                return y

            def dfff3(x):
                """Gradient of dfff2(x) x ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = jacobian(dfff2)(x).reshape(d, d)
                y = y @ ff(x)
                return y

            def d2ff(x):
                """Hessian matrix of ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = jacobian(dff)(x).reshape(d, d, d)
                return y

            if num_meth == "Forward Euler":
                def f2(x):
                    """Second term of the theoretical modified vector field for the Forward Euler method
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    y = (1 / 2) * dff(x) @ ff(x)
                    return y

                def df2(x):
                    """Jacobian matrix of the function f2
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    x = x.reshape(d, 1)
                    y = jacobian(f2)(x).reshape(d, d)
                    return y

                def f3(x):
                    """Third term of the theoretical modified vector field for the Forward Euler method
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    x = x.reshape(d, 1)
                    y = (1 / 6) * ((ff(x).T @ d2ff(x) @ ff(x)).reshape(d, 1) + dff(x) @ dff(x) @ ff(x))
                    return y

                def df3(x):
                    """Jacobian matrix of the function f2
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    x = x.reshape(d, 1)
                    y = jacobian(f3)(x).reshape(d, d)
                    return y

                def f4(x):
                    """Fourth term of the theoretical modified vector field for the Forward Euler method
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    x = x.reshape(d, 1)
                    y = (1 / 4) * (df3(x) @ ff(x))
                    return y

                if ord == 1:
                    yy = ff(x)
                if ord == 2:
                    yy = ff(x) + h * f2(x)
                if ord == 3:
                    yy = ff(x) + h * f2(x) + h ** 2 * f3(x)
                if ord == 4:
                    yy = ff(x) + h * f2(x) + h ** 2 * f3(x) + h ** 3 * f4(x)

            if num_meth == "MidPoint":
                def f3(x):
                    """Third term of the theoretical modified vector field for the MidPoint method
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    x = x.reshape(d, 1)
                    y = (1 / 12) * ((1 / 2) * (ff(x).T @ d2ff(x) @ ff(x)).reshape(d, 1) - dff(x) @ dff(x) @ ff(x))
                    return y

                if ord == 2:
                    yy = ff(x)
                if ord == 4:
                    yy = ff(x) + h ** 2 * f3(x)

            if num_meth == "RK2":
                def f2(x):
                    """Second term of the theoretical modified vector field for the Forward Euler method
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    y = (1 / 24) * dfff2(x) + (1 / 8) * dff(x) @ dff(x) @ ff(x)
                    return y

                def df2(x):
                    """Gradient of the Second term of the theoretical modified vector field for the Forward Euler method
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    x = x.reshape(d, 1)
                    y = jacobian(f2)(x).reshape(d, d)
                    return y

                def f3(x):
                    """Third term of the theoretical modified vector field for the Forward Euler method
                    Input:
                    x: Array of shape (d,1) or (d,)"""
                    y = (1 / 24) * dfff3(x) - (1 / 2) * dff(x) @ f2(x) - (1 / 2) * df2(x) @ ff(x)
                    return y

                if ord == 2:
                    yy = ff(x)
                if ord == 3:
                    yy = ff(x) + h ** 2 * f2(x)
                if ord == 4:
                    yy = ff(x) + h ** 2 * f2(x) + h ** 3 * f3(x)

            return yy

        HH = np.exp(np.linspace(np.log(step_h[0]), np.log(step_h[1]), 10))
        HHsq = HH ** 2
        HHcu = HH ** 3
        HHqu = HH ** 4

        XX = list(product(np.linspace(-R, R, 31), repeat=d))

        if num_meth == "Forward Euler":
            err1 = []
            err2 = []
            err3 = []
            err4 = []
            for hh in HH:
                print("  h= {} \r".format(format(hh, '.4E')), end="")
                ListDiff1 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=1).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                ListDiff2 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=2).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                ListDiff3 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=3).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                ListDiff4 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=4).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                err1 = err1 + [ListDiff1]
                err2 = err2 + [ListDiff2]
                err3 = err3 + [ListDiff3]
                err4 = err4 + [ListDiff4]

            plt.figure()
            plt.title("Error between learned field and modified field with " + num_meth)
            plt.scatter(HH, err1, label="Order 1", marker="s")
            plt.scatter(HH, err2, label="Order 2", marker="s")
            plt.scatter(HH, err3, label="Order 3", marker="s")
            plt.scatter(HH, err4, label="Order 4", marker="s")
            plt.plot(HH, HH, label="$h \mapsto h$", linestyle='dashed')
            plt.plot(HH, HHsq, label="$h \mapsto h^2$", linestyle='dashed')
            plt.plot(HH, HHcu, label="$h \mapsto h^3$", linestyle='dashed')
            plt.plot(HH, HHqu, label="$h \mapsto h^4$", linestyle='dashed')
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')

        if num_meth == "MidPoint":
            err2 = []
            err4 = []
            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")
                ListDiff2 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=2).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                ListDiff4 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=4).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                err2 = err2 + [ListDiff2]
                err4 = err4 + [ListDiff4]

            plt.figure()
            plt.title("Error between learned field and modified field for " + num_meth)
            plt.scatter(HH, err2, label="Order 2", marker="s")
            plt.scatter(HH, err4, label="Order 4", marker="s")
            plt.plot(HH, 0.1 * HHsq, label="$h \mapsto Ch^2$", linestyle='dashed')
            plt.plot(HH, 0.1 * HHqu, label="$h \mapsto Ch^4$", linestyle='dashed')
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')

        if num_meth == "RK2":
            # print("Convergence curves can be ploted only for Forward Euler and MidPoint.")
            err2 = []
            err3 = []
            err4 = []
            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")
                ListDiff2 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=2).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                ListDiff3 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=3).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                ListDiff4 = statistics.mean(
                    [np.linalg.norm(f_modif(x, hh, ord=4).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in XX])
                err2 = err2 + [ListDiff2]
                err3 = err3 + [ListDiff3]
                err4 = err4 + [ListDiff4]

            plt.figure()
            plt.title("Error between learned field and modified field with " + num_meth)
            plt.scatter(HH, err2, label="Order 2", marker="s")
            plt.scatter(HH, err3, label="Order 3", marker="s")
            plt.scatter(HH, err4, label="Order 4", marker="s")
            plt.plot(HH, HHsq, label="$h \mapsto h^2$", linestyle='dashed')
            plt.plot(HH, HHcu, label="$h \mapsto h^3$", linestyle='dashed')
            plt.plot(HH, HHqu, label="$h \mapsto h^4$", linestyle='dashed')
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')

        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))

        pass


class Difference(Convergence):
    def f_modif(self, x, h, ord):
        """Modified vector field (theory), depending on x, h, and for an order ord
        Inputs:
        - x: Array orn tensor of shape (d,1) - Space variable
        - h: Float - Step time for the numerical method
        - ord - Integer - Order chosen to develop the modifdied field"""
        x = np.array(x).reshape(d, 1)

        def ff(x):
            """First term of the theoretical modified vector field
            Input:
            x - Array of shape (d,1) or (d,)"""
            x = np.array(x).reshape(d, 1)
            y = self.f_array(0, x)
            return y.reshape(d, 1)

        def dff(x):
            """Gradient of ff(x)
            Input:
            x: Array of shape (d,1) or (d,)"""
            x = np.array(x).reshape(d, 1)
            y = jacobian(ff)(x).reshape(d, d)
            return y

        def d2ff(x):
            """Hessian matrix of ff(x)
            Input:
            x: Array of shape (d,1) or (d,)"""
            x = x.reshape(d, 1)
            y = jacobian(dff)(x).reshape(d, d, d)
            return y

        if num_meth == "Forward Euler":
            def f2(x):
                """Second term of the theoretical modified vector field for the Forward Euler method
                Input:
                x: Array of shape (d,1) or (d,)"""
                y = (1 / 2) * dff(x) @ ff(x)
                return y

            def df2(x):
                """Jacobian matrix of the function f2
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = jacobian(f2)(x).reshape(d, d)
                return y

            def f3(x):
                """Third term of the theoretical modified vector field for the Forward Euler method
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = (1 / 6) * ((ff(x).T @ d2ff(x) @ ff(x)).reshape(d, 1) + dff(x) @ dff(x) @ ff(x))
                return y

            def df3(x):
                """Jacobian matrix of the function f2
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = jacobian(f3)(x).reshape(d, d)
                return y

            def f4(x):
                """Fourth term of the theoretical modified vector field for the Forward Euler method
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = (1 / 4) * (df3(x) @ ff(x))
                return y

            if ord == 1:
                yy = ff(x)
            if ord == 2:
                yy = ff(x) + h * f2(x)
            if ord == 3:
                yy = ff(x) + h * f2(x) + h ** 2 * f3(x)
            if ord == 4:
                yy = ff(x) + h * f2(x) + h ** 2 * f3(x) + h ** 3 * f4(x)

        if num_meth == "MidPoint":
            def f3(x):
                """Third term of the theoretical modified vector field for the MidPoint method
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = (1 / 12) * ((1 / 2) * (ff(x).T @ d2ff(x) @ ff(x)).reshape(d, 1) - dff(x) @ dff(x) @ ff(x))
                return y

            if ord == 2:
                yy = ff(x)
            if ord == 4:
                yy = ff(x) + h ** 2 * f3(x)

        if num_meth == "RK2":
            def ff(x):
                """First term of the theoretical modified vector field
                Input:
                x - Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = self.f_array(0, x)
                return y.reshape(d, 1)

            def dff(x):
                """Gradient of ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = jacobian(ff)(x).reshape(d, d)
                return y

            def dfff(x):
                """Gradient of ff(x) x f(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = dff(x) @ ff(x)
                return y

            def dfff2(x):
                """Gradient of dfff(x) x ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = jacobian(dfff)(x).reshape(d, d)
                y = y @ ff(x)
                return y

            def dfff3(x):
                """Gradient of dfff2(x) x ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = np.array(x).reshape(d, 1)
                y = jacobian(dfff2)(x).reshape(d, d)
                y = y @ ff(x)
                return y

            def d2ff(x):
                """Hessian matrix of ff(x)
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = jacobian(dff)(x).reshape(d, d, d)
                return y

            def f2(x):
                """Second term of the theoretical modified vector field for the Forward Euler method
                Input:
                x: Array of shape (d,1) or (d,)"""
                y = (1 / 24) * dfff2(x) + (1 / 8) * dff(x) @ dff(x) @ ff(x)
                return y

            def df2(x):
                """Gradient of the Second term of the theoretical modified vector field for the Forward Euler method
                Input:
                x: Array of shape (d,1) or (d,)"""
                x = x.reshape(d, 1)
                y = jacobian(f2)(x).reshape(d, d)
                return y

            def f3(x):
                """Third term of the theoretical modified vector field for the Forward Euler method
                Input:
                x: Array of shape (d,1) or (d,)"""
                y = (1 / 24) * dfff3(x) - (1 / 2) * dff(x) @ f2(x) - (1 / 2) * df2(x) @ ff(x)
                return y

            if ord == 2:
                yy = ff(x)
            if ord == 3:
                yy = ff(x) + h ** 2 * f2(x)
            if ord == 4:
                yy = ff(x) + h ** 2 * f2(x) + h ** 3 * f3(x)

        return yy

    def LearnError(self, model):
        """Computes the learning error, i.e. the maximum of the L-infty norm on [-R,R]^d of the difference
        R_app(.,h)-R(.,h) where f_modif(y,h) = f(y) + h*R(y,h) or f(y) + h**2*R(y,h), and
        f_app(y,h) = f(y) + h*R_app(y,h) or f_app(y,h) = f(y) + h**2*R_app(y,h), over step_h
        Inputs:
        - model: Best model learned during training"""

        def f_app(x, h):
            """Learned vector field
            Inputs:
            - x: Array or tensor of shape (d,1)
            - h: Float - Step of time of the numerical method"""
            x = torch.tensor(x).reshape(d, 1)
            h = torch.tensor([[h]]).float()
            y = model(x, h)
            y = y.detach().numpy()
            return y

        delta = 0
        HH = np.exp(np.linspace(np.log(step_h[0]), np.log(step_h[1]), 15))
        XX = list(product(np.linspace(-R, R, 41), repeat=d))

        if num_meth == "Forward Euler":
            err_learn = []
            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")
                err = max(
                    [np.linalg.norm(self.f_modif(x, hh, ord=4).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in
                     XX]) / hh
                err_learn = err_learn + [err]
        if num_meth == "MidPoint":
            err_learn = []
            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")
                err = max(
                    [np.linalg.norm(self.f_modif(x, hh, ord=4).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in
                     XX]) / hh ** 2
                err_learn = err_learn + [err]
        if num_meth == "RK2":
            err_learn = []
            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")
                err = max(
                    [np.linalg.norm(self.f_modif(x, hh, ord=4).reshape(d, 1) - f_app(x, hh).reshape(d, 1)) for x in
                     XX]) / hh ** 2
                err_learn = err_learn + [err]

        delta = max(err_learn)
        torch.save([format(delta, ".4E")], "zeta=" + str(zeta) + ",HL=" + str(HL) + ",K_data=" + str(K_data))
        return delta

    def PlotDiff(self, model, name, save_fig):
        """Prints the contours of the norm of the difference between theoretical modified field at
        the highest order and learned field with neural network, only available in the case d=2
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""
        if d == 3:
            print("Only possible to plot the contours of difference in the case d=2")
        if d == 2:
            def f_app(x, h):
                """Learned vector field
                Inputs:
                - x: Array or tensor of shape (d,1)
                - h: Float - Step of time of the numerical method"""
                x = torch.tensor(x).reshape(d, 1)
                h = torch.tensor([[h]]).float()
                y = model(x, h)
                y = y.detach().numpy()
                return y

            def diff_fn(x):
                """Norm of the difference of f_app and f_modif at h=h_simul"""
                if num_meth == "Forward Euler":
                    dd = np.linalg.norm(
                        f_app(x, h_simul).reshape(d, ) - self.f_modif(x, h_simul, 4).reshape(d, )) / h_simul
                if num_meth == "MidPoint":
                    dd = np.linalg.norm(
                        f_app(x, h_simul).reshape(d, ) - self.f_modif(x, h_simul, 4).reshape(d, )) / h_simul ** 2
                if num_meth == "RK2":
                    dd = np.linalg.norm(
                        f_app(x, h_simul).reshape(d, ) - self.f_modif(x, h_simul, 4).reshape(d, )) / h_simul ** 2
                return dd

            print("Delta =", format(self.LearnError(model), '.4E'))

            nb_disc = 31  # Number of discretizations of the interval [-R,R]
            nb_contour = 20  # Number of contour lines on the plot
            xx = np.linspace(-R, R, nb_disc)
            yy = np.linspace(-R, R, nb_disc)
            XX, YY = np.meshgrid(xx, yy)
            ZZ = np.zeros((nb_disc, nb_disc))  # Matrix of the evaluation of diff_fn at (XX[i,j],YY[i,j])
            for i in range(nb_disc):
                for j in range(nb_disc):
                    ZZ[i, j] = diff_fn(np.array([XX[i, j], YY[i, j]]))
            fig, ax = plt.subplots()

            CS = ax.contour(XX, YY, ZZ, nb_contour)
            CS.levels = [format(val, ".2e") for val in CS.levels]
            ax.clabel(CS, inline=1, fontsize=10)
            ax.set_title("Error between perturbations - theoretical and learned - " + num_meth)
            ax.set_aspect("equal")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            f = plt.gcf()
            dpi = f.get_dpi()
            h, w = f.get_size_inches()
            f.set_size_inches(h * 1.5, w * 1.5)
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))

        pass


class TimeComputeModif(Difference):
    def timecompute_modif(self, model, name, save_fig):
        """Prints the points (Time computation, Accuracy) with these numerical methods:
        - If num_meth is "Forward Euler":
            -> Forward Euler with f_app
            -> Forward Euler with f_modif for orders 1 to 4
        - If num_meth is "MidPoint":
            -> MidPoint with f_app
            -> MidPoint with f_modif for orders 2 and 4
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential namle of the graph
        - save_fig: Boolean - Saves the figure or not"""

        def f_app(x, h):
            """Learned vector field
            Inputs:
            - x: Array or tensor of shape (d,1)
            - h: Float - Step time of the numerical method"""
            x = torch.tensor(x).reshape(d, 1)
            h = torch.tensor([[h]]).float()
            y = model(x, h)
            y = y.detach().numpy()
            return y

        HH = np.exp(np.linspace(np.log(step_h[0]), np.log(step_h[1]), 10))
        HH_bis = np.exp(np.linspace(np.log(step_h[1]), np.log(4 * step_h[1]), 6))[1:]

        ERR_meth = []  # Global errors between exact flow and numerical flow computed with f_app via num_meth
        ERR_f1 = []  # Global errors between exact flow and numerical flow computed with f_modif at order 1 via Forward Euler
        ERR_f2 = []  # Global errors between exact flow and numerical flow computed with f_modif at order 2 via Forward Euler
        ERR_f3 = []  # Global errors between exact flow and numerical flow computed with f_modif at order 3 via Forward Euler
        ERR_f4 = []  # Global errors between exact flow and numerical flow computed with f_modif at order 4 via Forward Euler

        Time_meth = []  # Time of computation of each trajectory via num_meth for f_app
        Time_f1 = []  # Time of computation of each trajectory via num_meth for f_modif at order 1
        Time_f2 = []  # Time of computation of each trajectory via num_meth for f_modif at order 2
        Time_f3 = []  # Time of computation of each trajectory via num_meth for f_modif at order 3
        Time_f4 = []  # Time of computation of each trajectory via num_meth for f_modif at order 4

        if num_meth == "Forward Euler":
            print("Integration with " + num_meth)

            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")

                def Fhat(t, y):
                    """Vector field learned with the neural network
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    h_tensor = torch.tensor([[hh]]).float()
                    z = model(y, h_tensor)
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                # Integration with RK45 (approximation of the exact flow)
                Y_exact = DynamicSystEDO().solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=hh, rel_tol=1e-15,
                                                     abs_tol=1e-15)
                norm_sol = np.linalg.norm(
                    np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]),
                    np.infty)  # Norm of the exact solution

                # Integration with Forward Euler and f_app
                time_meth = time.time()
                Y_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth)
                Time_meth = Time_meth + [time.time() - time_meth]

                # Integration with Forward Euler and modified field
                Y_f = []
                deltaTimef = []
                for order in range(1, 5):
                    time_int_f = time.time()

                    def F_modif(t, y):
                        """Modified vector field at a fixed order
                        Inputs:
                        - t: Float - Time
                        - y: Array of shape (d,) - Space variable"""
                        y = y.reshape(d, 1)
                        z = self.f_modif(y, hh, ord=order)
                        return z.reshape(d, )

                    Y_f.append(EDONum().solveEDO_num(F_modif, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth))
                    deltaTimef.append(time.time() - time_int_f)

                # Time_f1 = Time_f1 + [deltaTimef[0]]
                # Time_f2 = Time_f2 + [deltaTimef[1]]
                # Time_f3 = Time_f3 + [deltaTimef[2]]
                # Time_f4 = Time_f4 + [deltaTimef[3]]

                # Computation of error between trajectory ploted with f via RK45 and f_modif via Forward Euler
                err_f1 = np.array(
                    [np.linalg.norm((Y_exact - Y_f[0])[:, i]) for i in range((Y_exact - Y_f[0]).shape[1])])
                Err_f1 = np.linalg.norm(err_f1, np.infty) / norm_sol
                if Err_f1 < 1:
                    ERR_f1 = ERR_f1 + [Err_f1]
                    Time_f1 = Time_f1 + [deltaTimef[0]]
                err_f2 = np.array(
                    [np.linalg.norm((Y_exact - Y_f[1])[:, i]) for i in range((Y_exact - Y_f[1]).shape[1])])
                Err_f2 = np.linalg.norm(err_f2, np.infty) / norm_sol
                if Err_f2 < 1:
                    ERR_f2 = ERR_f2 + [Err_f2]
                    Time_f2 = Time_f2 + [deltaTimef[1]]
                err_f3 = np.array(
                    [np.linalg.norm((Y_exact - Y_f[2])[:, i]) for i in range((Y_exact - Y_f[2]).shape[1])])
                Err_f3 = np.linalg.norm(err_f3, np.infty) / norm_sol
                if Err_f3 < 1:
                    ERR_f3 = ERR_f3 + [Err_f3]
                    Time_f3 = Time_f3 + [deltaTimef[2]]
                err_f4 = np.array(
                    [np.linalg.norm((Y_exact - Y_f[3])[:, i]) for i in range((Y_exact - Y_f[3]).shape[1])])
                Err_f4 = np.linalg.norm(err_f4, np.infty) / norm_sol
                if Err_f4 < 1:
                    ERR_f4 = ERR_f4 + [Err_f4]
                    Time_f4 = Time_f4 + [deltaTimef[3]]

                # Computation of error between trajectory ploted with f via RK45 and f via Forward Euler
                err_meth = np.array(
                    [np.linalg.norm((Y_exact - Y_meth)[:, i]) for i in range((Y_exact - Y_meth).shape[1])])
                Err_meth = np.linalg.norm(err_meth, np.infty) / norm_sol
                ERR_meth = ERR_meth + [Err_meth]

            plt.figure()
            plt.title("Computation time vs Global error - AI method vs modified field")
            plt.scatter(Time_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
            if len(Time_f1) > 0:
                plt.scatter(Time_f1, ERR_f1, label=num_meth + " 0rder 1", marker="s", color="red")
            if len(Time_f2) > 0:
                plt.scatter(Time_f2, ERR_f2, label=num_meth + " 0rder 2", marker="s", color="orange")
            if len(Time_f3) > 0:
                plt.scatter(Time_f3, ERR_f3, label=num_meth + " 0rder 3", marker="s", color="magenta")
            if len(Time_f4) > 0:
                plt.scatter(Time_f4, ERR_f4, label=num_meth + " 0rder 4", marker="s", color="cyan")
            plt.xlabel("Computation time (s)")
            plt.ylabel("Global error")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        if num_meth == "RK2":
            print("Integration with " + num_meth)

            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")

                def Fhat(t, y):
                    """Vector field learned with the neural network
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    h_tensor = torch.tensor([[hh]]).float()
                    z = model(y, h_tensor)
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                # Integration with RK45 (approximation of the exact flow)
                Y_exact = DynamicSystEDO().solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=hh, rel_tol=1e-15, abs_tol=1e-15)
                norm_sol = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]),np.infty)  # Norm of the exact solution

                # Integration with RK2 and f_app
                time_meth = time.time()
                Y_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth)
                Time_meth = Time_meth + [time.time() - time_meth]

                # Integration with RK2 and modified field
                Y_f = []
                deltaTimef = []
                for order in range(2, 5):
                    time_int_f = time.time()

                    def F_modif(t, y):
                        """Modified vector field at a fixed order
                        Inputs:
                        - t: Float - Time
                        - y: Array of shape (d,) - Space variable"""
                        y = y.reshape(d, 1)
                        z = self.f_modif(y, hh, ord=order)
                        return z.reshape(d, )

                    Y_f.append(EDONum().solveEDO_num(F_modif, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth))
                    deltaTimef.append(time.time() - time_int_f)

                # Time_f1 = Time_f1 + [deltaTimef[0]]
                # Time_f2 = Time_f2 + [deltaTimef[1]]
                # Time_f3 = Time_f3 + [deltaTimef[2]]
                # Time_f4 = Time_f4 + [deltaTimef[3]]

                # Computation of error between trajectory ploted with f via RK45 and f_modif via Forward Euler
                # err_f1 = np.array([np.linalg.norm((Y_exact - Y_f[0])[:, i]) for i in range((Y_exact - Y_f[0]).shape[1])])
                # Err_f1 = np.linalg.norm(err_f1, np.infty)/norm_sol
                # if Err_f1 < 1:
                #    ERR_f1 = ERR_f1 + [Err_f1]
                #    Time_f1 = Time_f1 + [deltaTimef[0]]
                err_f2 = np.array(
                    [np.linalg.norm((Y_exact - Y_f[0])[:, i]) for i in range((Y_exact - Y_f[0]).shape[1])])
                Err_f2 = np.linalg.norm(err_f2, np.infty) / norm_sol
                if Err_f2 < 1:
                    ERR_f2 = ERR_f2 + [Err_f2]
                    Time_f2 = Time_f2 + [deltaTimef[0]]
                err_f3 = np.array(
                    [np.linalg.norm((Y_exact - Y_f[1])[:, i]) for i in range((Y_exact - Y_f[1]).shape[1])])
                Err_f3 = np.linalg.norm(err_f3, np.infty) / norm_sol
                if Err_f3 < 1:
                    ERR_f3 = ERR_f3 + [Err_f3]
                    Time_f3 = Time_f3 + [deltaTimef[1]]
                err_f4 = np.array(
                    [np.linalg.norm((Y_exact - Y_f[2])[:, i]) for i in range((Y_exact - Y_f[2]).shape[1])])
                Err_f4 = np.linalg.norm(err_f4, np.infty) / norm_sol
                if Err_f4 < 1:
                    ERR_f4 = ERR_f4 + [Err_f4]
                    Time_f4 = Time_f4 + [deltaTimef[2]]

                # Computation of error between trajectory ploted with f via RK45 and f via RK2
                err_meth = np.array([np.linalg.norm((Y_exact - Y_meth)[:, i]) for i in range((Y_exact - Y_meth).shape[1])])
                Err_meth = np.linalg.norm(err_meth, np.infty) / norm_sol
                ERR_meth = ERR_meth + [Err_meth]

            plt.figure()
            plt.title("Computation time vs Global error - AI method vs modified field")
            plt.scatter(Time_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
            # if len(Time_f1) > 0:
            #    plt.scatter(Time_f1, ERR_f1, label=num_meth + " 0rder 1", marker="s",color="red")
            if len(Time_f2) > 0:
                plt.scatter(Time_f2, ERR_f2, label=num_meth + " 0rder 2", marker="s", color="red")
            if len(Time_f3) > 0:
                plt.scatter(Time_f3, ERR_f3, label=num_meth + " 0rder 3", marker="s", color="orange")
            if len(Time_f4) > 0:
                plt.scatter(Time_f4, ERR_f4, label=num_meth + " 0rder 4", marker="s", color="magenta")
            plt.xlabel("Computation time (s)")
            plt.ylabel("Global error")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        if num_meth == "MidPoint":
            print("Integration with " + num_meth)

            for hh in HH:
                print(" h= {} \r".format(format(hh, '.4E')), end="")

                def Fhat(t, y):
                    """Vector field learned with the neural network
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    h_tensor = torch.tensor([[hh]]).float()
                    z = model(y, h_tensor)
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                # Integration with RK45 (approximation of the exact flow)
                Y_exact = DynamicSystEDO().solvefEDO(y0=y0_start(dyn_syst), T=T_simul, h=hh, rel_tol=1e-15, abs_tol=1e-15)
                norm_sol = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]), np.infty)  # Norm of the exact solution

                # Integration with MidPoint and f_app
                time_meth = time.time()
                Y_meth = EDONum().solveEDO_num(Fhat, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth)
                Time_meth = Time_meth + [time.time() - time_meth]

                # Integration with MidPoint and modified field
                Y_f = []
                deltaTimef = []
                for order in range(2, 5, 2):
                    time_int_f = time.time()

                    def F_modif(t, y):
                        """Modified vector field at a fixed order
                        Inputs:
                        - t: Float - Time
                        - y: Array of shape (d,) - Space variable"""
                        y = y.reshape(d, 1)
                        z = self.f_modif(y, hh, ord=order)
                        return z.reshape(d, )

                        Y_f.append(EDONum().solveEDO_num(F_modif, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth))
                        deltaTimef.append(time.time() - time_int_f)

                    Y_f.append(EDONum().solveEDO_num(F_modif, y0=y0_start(dyn_syst), T=T_simul, h=hh, meth=num_meth))
                    deltaTimef.append(time.time() - time_int_f)


                err_f2 = np.array([np.linalg.norm((Y_exact - Y_f[0])[:, i]) for i in range((Y_exact - Y_f[0]).shape[1])])
                Err_f2 = np.linalg.norm(err_f2, np.infty) / norm_sol
                if Err_f2 < 1:
                    ERR_f2 = ERR_f2 + [Err_f2]
                    Time_f2 = Time_f2 + [deltaTimef[0]]

                err_f4 = np.array([np.linalg.norm((Y_exact - Y_f[1])[:, i]) for i in range((Y_exact - Y_f[1]).shape[1])])
                Err_f4 = np.linalg.norm(err_f4, np.infty) / norm_sol
                if Err_f4 < 1:
                    ERR_f4 = ERR_f4 + [Err_f4]
                    Time_f4 = Time_f4 + [deltaTimef[1]]

                # Computation of error between trajectory ploted with f via RK45 and f via RK2
                err_meth = np.array([np.linalg.norm((Y_exact - Y_meth)[:, i]) for i in range((Y_exact - Y_meth).shape[1])])
                Err_meth = np.linalg.norm(err_meth, np.infty) / norm_sol
                ERR_meth = ERR_meth + [Err_meth]

            plt.figure()
            plt.title("Computation time vs Global error - AI method vs modified field")
            plt.scatter(Time_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
            # if len(Time_f1) > 0:
            #    plt.scatter(Time_f1, ERR_f1, label=num_meth + " 0rder 1", marker="s",color="red")
            if len(Time_f2) > 0:
                plt.scatter(Time_f2, ERR_f2, label=num_meth + " 0rder 2", marker="s", color="red")
            if len(Time_f4) > 0:
                plt.scatter(Time_f4, ERR_f4, label=num_meth + " 0rder 4", marker="s", color="orange")
            plt.xlabel("Computation time (s)")
            plt.ylabel("Global error")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()



        else:
            print("Only available for Forward Euler, RK2 and MidPoint")
        pass


class Parameters:
    def PrintParameters(self, name_print, save_fig):
        """
        Creates a print of the parameters in order to save them with PDF
        Inputs:
        - name_print: Character string - Potential name of the print
        - save_fig: Boolean - Saves the figure or not
        returns a plot of all parameters selected at the beginning of the simulation
        """
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(webcolors.rgb_to_hex((60, 60, 60)))
        plt.text(0.5, 0.97, "Parameters", horizontalalignment="center", verticalalignment="center",
                 transform=ax.transAxes, fontsize=11, color=webcolors.rgb_to_hex((255, 50, 50)))
        plt.text(0.02, 0.9, "# Math parameters:", transform=ax.transAxes, fontsize=8,
                 color=webcolors.rgb_to_hex((0, 255, 0)))
        plt.text(0.05, 0.86, "- Dynamical system: " + str(dyn_syst), transform=ax.transAxes, fontsize=8, color="white")
        plt.text(0.05, 0.82, "- Numerical method: " + str(num_meth), transform=ax.transAxes, fontsize=8, color="white")
        plt.text(0.05, 0.78, "- Interval where time steps are selected for training: " + str(step_h),
                 transform=ax.transAxes, fontsize=8, color="white")
        plt.text(0.05, 0.74, "- Time for ODE's simulation: " + str(T_simul), transform=ax.transAxes, fontsize=8,
                 color="white")
        plt.text(0.05, 0.70, "- Time step for ODE's simulation: " + str(h_simul), transform=ax.transAxes, fontsize=8,
                 color="white")
        plt.text(0.05, 0.66, "- Amplitude of noise for data's perturbation: " + str(sigma), transform=ax.transAxes,
                 fontsize=8, color="white")
        plt.text(0.02, 0.6, "# AI parameters:", transform=ax.transAxes, fontsize=8,
                 color=webcolors.rgb_to_hex((0, 255, 0)))
        if dyn_syst == "SIR":
            plt.text(0.05, 0.56,
                     "- Domain where data are selected: " + "$\{x + \eta : x\in[0,1]^3, x_1+x_2+x_3=1, \eta \in 0.02\cdot[-1,1]^3\}$",
                     transform=ax.transAxes, fontsize=8, color="white")
        elif dyn_syst == "Rigid Body":
            plt.text(0.05, 0.56,
                     "- Domain where data are selected: " + "$\{x \in [-R,R]^d : |x|\in[0.98,1.02]\}$, R=" + str(R),
                     transform=ax.transAxes, fontsize=8, color="white")
        else:
            plt.text(0.05, 0.56,
                     "- Domain where data are selected: " + "$[-R,R]^d$, R= " + str(R) + " and d= " + str(d),
                     transform=ax.transAxes, fontsize=8, color="white")
        K_str = str(K_data)
        K_data_str = ' '.join([K_str[::-1][i:i + 3] for i in range(0, len(K_str), 3)])[::-1]
        plt.text(0.05, 0.52, "- Number of Data: " + K_data_str, transform=ax.transAxes, fontsize=8, color="white")
        plt.text(0.05, 0.48, "- Number of time steps selected for an initial data: " + str(N_h), transform=ax.transAxes,
                 fontsize=8, color="white")
        plt.text(0.05, 0.44, "- Proportion of data for training: " + str(p_train), transform=ax.transAxes, fontsize=8,
                 color="white")
        plt.text(0.05, 0.40, "- Number of terms in the perturbation (MLP's): " + str(N_terms), transform=ax.transAxes,
                 fontsize=8, color="white")
        plt.text(0.05, 0.36, "- Hidden layers per MLP: " + str(HL), transform=ax.transAxes, fontsize=8, color="white")
        plt.text(0.05, 0.32, "- Neurons on each hidden layer: " + str(zeta), transform=ax.transAxes, fontsize=8,
                 color="white")
        plt.text(0.05, 0.28, "- Learning rate: " + str(format(alpha, '.2E')), transform=ax.transAxes, fontsize=8,
                 color="white")
        plt.text(0.05, 0.24, "- Weight decay: " + str(format(Lambda, '.2E')), transform=ax.transAxes, fontsize=8,
                 color="white")
        plt.text(0.05, 0.20, "- Batch size (mini-batching for training): " + str(BS), transform=ax.transAxes,
                 fontsize=8, color="white")
        plt.text(0.05, 0.16, "- Epochs: " + str(N_epochs), transform=ax.transAxes, fontsize=8, color="white")
        plt.text(0.05, 0.12, "- Epochs between two prints of the loss value: " + str(N_epochs_print),
                 transform=ax.transAxes, fontsize=8, color="white")
        f = plt.gcf()
        dpi = f.get_dpi()
        h, w = f.get_size_inches()
        f.set_size_inches(h * 1.4, w * 1.4)
        plt.show()
        if save_fig == True:
            plt.savefig(name_print + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        pass



# Exacution of the code

start_time = time.time()


def ExData(name_data="DataEDO"):
    """Creates data y0, y1 with the function solvefEDOData
    with the chosen vector field at the beginning of the program
    Input:
    - name_data: Character string - Name of the registered tuple containing the data (default: "DataEDO")"""
    DataEDO = DynamicSystEDO().solvefEDOData(K=K_data, p=p_train, h_data=step_h)
    torch.save(DataEDO, name_data)
    pass


def ExTrain(name_loss_train='Loss_train', name_loss_test='Loss_test', name_model='best_model', name_data='DataEDO'):
    """Launches training and computes Loss_train, loss_test and best_model with the function Train().train
    Saves the files Loss_train, Loss_test and best_model with a given name
    Inputs (character strings):
    - name_loss_train: Name of the file saved for Loss_train (default: "Loss_train")
    - name_loss_test: Name of the file saved for Loss_test (default: "Loss_test")
    - name_model: Name of the file saved for best_model (default: "best_model")
    - name_data: Name of the file containing the created data (default: "DataEDO") used for training"""
    DataEDO = torch.load(name_data)
    Loss_train, Loss_test, best_model = Train().train(model=NN(), Data=DataEDO, K=K_data, p=p_train, Nb_epochs=N_epochs,
                                                      Nb_epochs_print=N_epochs_print)
    torch.save(Loss_train, name_loss_train)
    torch.save(Loss_test, name_loss_test)
    torch.save(best_model, name_model)
    pass


def ExIntegrate(name_loss_train="Loss_train", name_loss_test="Loss_test", name_model="best_model", name_graph="Simulation", save=False, SL = False):
    """Launches integration of the main equation and modified equation with the chosen model
    Inputs:
    - name_loss_train: Character string - Name of the file containing loss_train that we want to print
    - name_loss_test: Character string - Name of the file containing loss_test that we want to print
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)
    - SL: Boolean - Saves the lists of times and corresponding local errors (default: False)"""
    Loss_train = torch.load(name_loss_train)
    Loss_test = torch.load(name_loss_test)
    best_model = torch.load(name_model)
    Integrate().integrate(Nb_epochs=N_epochs, Ltr=Loss_train, Lte=Loss_test, model=best_model, name=name_graph, save_fig=save, save_list=SL)
    pass


def ExTraj(name_model="best_model", name_graph="Simulation_Convergence_Trajectories", save=False, SL=False):
    """plots the curves of convergence between the trajectories integrated with f and f_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)
    - SL: Boolean - Saves the lists of time steps and corresponding global errors (default: False)"""
    best_model = torch.load(name_model)
    Trajectories().traj(model=best_model, name=name_graph, save_fig=save, save_list=SL)
    pass


def ExTime(name_model="best_model", name_graph="Simulation_Time", save=False , SL=False):
    """Plots points (Computation time,Accuracy) depending of the choice
    of numerical method (f, f_app, RK2 with f and DOPRI5 with f)
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)
    - SL: Boolean - Saves the lists of compuational times and corresponfing global errors (default: False)"""
    best_model = torch.load(name_model)
    TimeCompute().timecompute(model=best_model, name=name_graph, save_fig=save, save_list=SL)


def ExModelEval(name_model = "best_model"):
    """Computes the averaged time for evaluation for the selected model
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app"""
    best_model = torch.load(name_model)
    ModelEval().model_eval(best_model)


def ExConv(name_model="best_model", name_graph="Simulation_Convergence", save=False):
    """Plots the curves of convergence between the learned field and the modified field
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    best_model = torch.load(name_model)
    Convergence().curves(model=best_model, name=name_graph, save_fig=save)
    pass


def ExDiff(name_model="best_model", name_graph="Difference", save=False):
    """Plots the contours of the norm of the difference between learned modified field and theoretical modified field at the
    highest order
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    best_model = torch.load(name_model)
    Difference().PlotDiff(model=best_model, name=name_graph, save_fig=save)
    pass


def ExModif(name_model="best_model", name_graph="Simulation_Time_Modif", save=False):
    """Plots points (Computation time,Accuracy) for Forward Euler with f_app and f_modif at orders 1 to 4
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    best_model = torch.load(name_model)
    TimeComputeModif().timecompute_modif(model=best_model, name=name_graph, save_fig=save)


def ExInv(name_model="best_model", name_graph="Simulation_Invariants", save=False):
    """Plots curves (Time,Errors) for Invariants over numerical flows
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    best_model = torch.load(name_model)
    Invariant().inv_evolution(model=best_model, name=name_graph, save_fig=save)


def ExParameters(name_param="parameters", save=False):
    """
    Creates a print of the parameters in order to save them with PDF
    Inputs:
    - name_param: Character string - Name of the print which will be registered
    - save: Boolean - Saves the figure or not (default: True)
    """
    Parameters().PrintParameters(name_print=name_param, save_fig=save)





# ExData()
# ExTrain()

# ExData(name_data="DataEDO,K_data="+str(K_data))

# Zeta = [2000,3000,5000,8000,10000]
# #HHL = [1,2,3,4,5,6,7,8,9,10]
# delta_zeta = []
# for ii in range(len(Zeta)):
#     #for i in range(len(HHL)):
#     zeta = Zeta[ii]
#     #HL = HHL[i]
#     print("zeta="+str(zeta)+", HL="+str(HL))
#     ExTrain(name_data="DataEDO,K_data="+str(K_data))
#     ExDiff()
#
# for i in range(len(Zeta)):
#     delta_zeta = delta_zeta+torch.load("zeta="+str(Zeta[i])+",HL="+str(HL)+",K_data="+str(K_data))
#     # deltaaHHL = []
#     # for i in range(len(HHL)):
#     #     deltaaHHL = deltaaHHL + torch.load("zeta="+str(zeta)+",HL="+str(HHL[i])+",K_data="+str(K_data))
#
#     #torch.save(deltaaHHL,"deltaaHHL, zeta="+str(zeta)+", K_data="+str(K_data))
# torch.save(delta_zeta,"delta_zeta,K_data="+str(K_data))

# DDD = torch.load("DataEDO_RKmodif_2_Comparison_K_50000_Nh_5")
# YY0_train = np.array(DDD[-4])
# YY0_test = np.array(DDD[-3])
# h_train = np.array(DDD[-2])
# h_test = np.array(DDD[-1])
#
# YY1_train , YY1_test = np.ones_like(YY0_train) , np.ones_like(YY0_test)
#
# K = YY0_train.shape[1]
# print("Train data...")
# for k in range(K):
#     print("  {} % \r".format(str(int(1000 * (k + 1) / K) / 10).rjust(3)), end="")
#     YY1_train[:, k] = DynamicSystEDO().solvefEDO(y0=YY0_train[:, k], T=2 * h_train[0, k], h=h_train[0, k])[:, 1]
#
# K = YY0_test.shape[1]
# print("Test data...")
# for k in range(K):
#     print("  {} % \r".format(str(int(1000 * (k + 1) / K) / 10).rjust(3)), end="")
#     YY1_test[:, k] = DynamicSystEDO().solvefEDO(y0=YY0_test[:, k], T=2 * h_test[0, k], h=h_test[0, k])[:, 1]

print("Time for code execution (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time))))