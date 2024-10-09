# noinspection PyInterpreter
import sys
from time import sleep
import warnings

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
import matplotlib as mpl
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
import tikzplotlib

def path_save(simulation):
    if simulation == "Data_Number_Fixed":
        return "/Users/maximebouchereau/Python/Comparison_RK_modif/RKmodif_K_50000_Nh_5_vs_RKmodif_2_K_50000_Nh_5/"
    if simulation == "Training_Time_Fixed":
        return "/Users/maximebouchereau/Python/Comparison_RK_modif/RKmodif_K_50000_Nh_5_vs_RKmodif_2_K_76129_Nh_5_vs_Parallel_Training/"

def parameters_simul(simulation):
    if simulation == "Data_Number_Fixed":
        return "_K_50000_Nh_5"
    if simulation == "Training_Time_Fixed":
        return "_K_76129_Nh_5"
    if simulation == "Training_Time_Fixed_PT":
        return "_K_105735_Nh_5"

def time_step_range(simulation):
    if simulation == "0.001_1.0":
        return "_h_0.001_1.0"
    if simulation == "0.0001_1.0":
        return "_h_0.0001_1.0"

path = path_save("Data_Number_Fixed")
path_2 = path_save("Training_Time_Fixed")
parameters = parameters_simul("Training_Time_Fixed")
parameters_2 = parameters_simul("Training_Time_Fixed_PT")
TS = time_step_range("0.0001_1.0")


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

def ExError(save=False):
    plt.figure()
    plt.title("Comparison of local errors")
    plt.yscale('log')
    TT_1 = torch.load(path+"Integrate_Time_RKmodif_Comparison_K_50000_Nh_5")
    TT_2 = torch.load(path_2+"Integrate_Time_RKmodif_2_Comparison"+parameters)
    TT_3 = torch.load(path_2+"Integrate_Time_RKmodif_2_Comparison"+parameters_2)

    err_f = torch.load(path+"Integrate_Error_f_RKmodif_Comparison_K_50000_Nh_5")
    err_f_app_1 = torch.load(path+"Integrate_Error_f_app_RKmodif_Comparison_K_50000_Nh_5")
    err_f_app_2 = torch.load(path_2+"Integrate_Error_f_app_RKmodif_2_Comparison"+parameters)
    err_f_app_3 = torch.load(path_2 + "Integrate_Error_f_app_RKmodif_2_Comparison" + parameters_2)
    plt.plot(TT_1, err_f, color="red", label="Forward Euler")
    plt.plot(TT_1, err_f_app_1, color="blue", label="Unic learning")
    plt.plot(TT_2, err_f_app_2, color="orange", label="Separate learning")
    plt.plot(TT_3, err_f_app_3, color="green", label="Separate learning - PT")
    plt.xlabel("t")
    plt.ylabel("Local error")
    plt.legend()
    plt.grid()
    write_size()
    if save == True:
        plt.savefig(path_2+"Comparison_RKmodif_1_2_Local_Error"+parameters+".pdf")
    plt.show()
    pass

def ExTrajectories(save=False):
    plt.figure()
    plt.title("Comparison of global errors")
    plt.yscale('log')
    plt.xscale('log')
    HH_f = torch.load(path+"Trajectories_Time_steps_f_RKmodif_Comparison_K_50000_Nh_5"+TS)
    ERR_f =  torch.load(path+"Trajectories_Errors_f_RKmodif_Comparison_K_50000_Nh_5"+TS)
    HH_meth_1 = torch.load(path+"Trajectories_Time_steps_f_app_RKmodif_Comparison_K_50000_Nh_5"+TS)
    ERR_meth_1 = torch.load(path+"Trajectories_Errors_f_app_RKmodif_Comparison_K_50000_Nh_5"+TS)
    HH_meth_2 = torch.load(path_2+"Trajectories_Time_steps_f_app_RKmodif_2_Comparison"+parameters+TS)
    ERR_meth_2 = torch.load(path_2+"Trajectories_Errors_f_app_RKmodif_2_Comparison"+parameters+TS)
    HH_meth_3 = torch.load(path_2 + "Trajectories_Time_steps_f_app_RKmodif_2_Comparison" + parameters_2 + TS)
    ERR_meth_3 = torch.load(path_2 + "Trajectories_Errors_f_app_RKmodif_2_Comparison" + parameters_2 + TS)
    plt.scatter(HH_f, ERR_f, color="red", label="Forward Euler" , marker = "s")
    plt.scatter(HH_meth_1, ERR_meth_1, color="blue", label="Unic learning" , marker = "s")
    plt.scatter(HH_meth_2, ERR_meth_2, color="orange", label="Separate learning" , marker = "s")
    plt.scatter(HH_meth_3, ERR_meth_3, color="green", label="Separate learning - PT", marker="s")
    plt.xlabel("h")
    plt.ylabel("Global error")
    plt.legend()
    plt.grid()
    write_size()
    if save == True:
        plt.savefig(path_2+"Comparison_RKmodif_1_2_Trajectories"+parameters+".pdf")
    plt.show()
    pass

def ExTime(save=False):
    plt.figure()
    plt.title("Comparison of computational times")
    plt.yscale('log')
    plt.xscale('log')
    Time_f = torch.load(path+"Time_Time_f_RKmodif_Comparison_K_50000_Nh_5")
    ERR_f = torch.load(path+"Time_Error_f_RKmodif_Comparison_K_50000_Nh_5")
    Time_f_app_1 = torch.load(path+"Time_Time_f_app_RKmodif_Comparison_K_50000_Nh_5")
    ERR_f_app_1 = torch.load(path+"Time_Error_f_app_RKmodif_Comparison_K_50000_Nh_5")
    Time_f_app_2 = torch.load(path_2+"Time_Time_f_app_RKmodif_2_Comparison"+parameters)
    ERR_f_app_2 = torch.load(path_2+"Time_Error_f_app_RKmodif_2_Comparison"+parameters)
    Time_f_app_3 = torch.load(path_2 + "Time_Time_f_app_RKmodif_2_Comparison" + parameters_2)
    ERR_f_app_3 = torch.load(path_2 + "Time_Error_f_app_RKmodif_2_Comparison" + parameters_2)
    plt.scatter(Time_f, ERR_f, color="red", label="Forward Euler" , marker = "s")
    plt.scatter(Time_f_app_1, ERR_f_app_1, color="blue", label="Unic learning" , marker = "s")
    plt.scatter(Time_f_app_2, ERR_f_app_2, color="orange", label="Separate learning" , marker = "s")
    plt.scatter(Time_f_app_3, ERR_f_app_3, color="green", label="Separate learning - PT", marker="s")
    plt.xlabel("Time (s)")
    plt.ylabel("Global error")
    plt.legend()
    plt.grid()
    write_size()
    if save == True:
        plt.savefig(path_2+"Comparison_RKmodif_1_2_Time"+parameters+".pdf")
    plt.show()
    pass

def ExNhError(save=False):
    plt.figure()
    plt.title("Comparison of Integration errors")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("error")

    path = "/Users/maximebouchereau/Python/Comparison_RK_modif/RKmodif_2_Comparison_Variable_Nh_K_10000/"
    name_time = "Integrate_Time_RKmodif2_Comparison_K_10000_Nh_"
    name_err = "Integrate_Error_f_app_RKmodif_2_Comparison_K_10000_Nh_"

    colors = ["darkgreen","limegreen","chartreuse","greenyellow","yellow"]

    i = 0
    for NNh in [1,3,5,7,9]:
        Time_f_app = torch.load(path+name_time+str(NNh))
        Err_f_app = torch.load(path+name_err+str(NNh))
        plt.plot(Time_f_app,Err_f_app,color = colors[i] , label = "$N_h = $"+str(NNh))
        i += 1

    plt.legend()
    plt.grid()
    if save == True:
        plt.savefig(path + "Comparison_RKmodif_2_Integration_Variable_Nh_k_10000.pdf")
    plt.show()
    pass

def ExNhTraj(save=False):
    plt.figure()
    plt.title("Comparison of Global errors")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel("error")

    path = "/Users/maximebouchereau/Python/Comparison_RK_modif/RKmodif_2_Comparison_Variable_Nh_K_10000/"
    name_time = "Trajectories_Time_steps_f_app_RKmodif_2_Comparison_K_10000_Nh_"
    name_err = "Trajectories_Errors_f_app_RKmodif_2_Comparison_K_10000_Nh_"

    colors = ["darkgreen","limegreen","chartreuse","greenyellow","yellow"]

    i = 0
    for NNh in [1,3,5,7,9]:
        Time_f_app = torch.load(path+name_time+str(NNh))
        Err_f_app = torch.load(path+name_err+str(NNh))
        plt.scatter(Time_f_app,Err_f_app,color = colors[i] , label = "$N_h = $"+str(NNh) , marker = "s")
        i += 1

    plt.legend()
    plt.grid()
    if save == True:
        plt.savefig(path + "Comparison_RKmodif_2_Trajectories_Variable_Nh_k_10000.pdf")
    plt.show()
    pass