import numpy as np
import scipy.io
import numpy.matlib
import os
import h5py
import math
import csv
from copy import copy

from scipy import linalg
import scipy.special
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.special
from scipy import stats
from scipy.linalg import eigh

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression,  RANSACRegressor
from sklearn import manifold

import scipy.special
from scipy import stats
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from scipy.stats import zscore

import tensorflow as tf

from keras.layers import Dense, Input, Concatenate, Reshape, Lambda, Add, Multiply
from keras.layers import Conv1D, Flatten, MaxPooling1D, Activation, Dropout, GaussianNoise, BatchNormalization, LayerNormalization, MaxPooling2D
from keras.layers import Conv2D, Conv2DTranspose, ThresholdedReLU, UpSampling2D, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling2D, PReLU
from keras.constraints import max_norm, UnitNorm, Constraint, MinMaxNorm, NonNeg
from keras.initializers import Constant
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, kullback_leibler_divergence
from keras.regularizers import l1, l2, l1_l2
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K

import pathlib

import seaborn as sns
from .mpl_functions import adjust_spines
from .lollipop_figure import Lollipop

class LatentAnalysis():

    def __init__(self):
        self.dt = 1.0/15000.0
        self.N_frames = 16375
        self.N_pol_theta = 20
        self.N_pol_eta = 24
        self.N_pol_phi = 16
        self.N_pol_xi = 20
        self.N_const = 3
        self.muscle_names = ['b1','b2','b3','i1','i2','iii1','iii24','iii3','hg1','hg2','hg3','hg4','freq']
        self.c_muscle = ['darkred','red','orangered','dodgerblue','blue','darkgreen','lime','springgreen','indigo','fuchsia','mediumorchid','deeppink','k']
        self.N_window = 9
        self.working_dir = pathlib.Path.cwd()
        # mass fly
        self.m_fly = 1.06e-6
        # gravitational acceration
        self.g = 9800.0

    def load_dataset(self,dataset_file):
        self.data_file = h5py.File(str(dataset_file),'r')
        self.data_keys = list(self.data_file.keys())
        self.N_files = len(self.data_keys)
        print('Number of files: '+str(self.N_files))
        # data lists:
        self.a_theta_L_mov = []
        self.a_eta_L_mov = []
        self.a_phi_L_mov = []
        self.a_xi_L_mov = []
        self.a_x_L_mov = []
        self.a_y_L_mov = []
        self.a_z_L_mov = []
        self.a_theta_R_mov = []
        self.a_eta_R_mov = []
        self.a_phi_R_mov = []
        self.a_xi_R_mov = []
        self.a_x_R_mov = []
        self.a_y_R_mov = []
        self.a_z_R_mov = []
        self.s1_s2_mov = []
        self.T_mov = []
        self.time_wb_mov = []
        self.freq_mov = []
        self.t_wbs_mov = []
        self.b1_wbs = []
        self.b2_wbs = []
        self.b3_wbs = []
        self.i1_wbs = []
        self.i2_wbs = []
        self.iii1_wbs = []
        self.iii24_wbs = []
        self.iii3_wbs = []
        self.hg1_wbs = []
        self.hg2_wbs = []
        self.hg3_wbs = []
        self.hg4_wbs = []
        for i in range(self.N_files):
            print('file nr: '+str(i+1))
            mov_group = self.data_file[self.data_keys[i]]
            n_triggers = np.squeeze(np.copy(mov_group['N_triggers']))
            m_keys = list(mov_group.keys())
            mov_keys = [k for k in m_keys if 'mov_' in k]
            n_mov = len(mov_keys)
            for j in range(n_mov):
                a_theta_L_i = np.copy(mov_group[mov_keys[j]]['a_theta_L'])
                a_eta_L_i = np.copy(mov_group[mov_keys[j]]['a_eta_L'])
                a_phi_L_i = np.copy(mov_group[mov_keys[j]]['a_phi_L'])
                a_xi_L_i = np.copy(mov_group[mov_keys[j]]['a_xi_L'])
                a_x_L_i = np.copy(mov_group[mov_keys[j]]['a_x_L'])
                a_y_L_i = np.copy(mov_group[mov_keys[j]]['a_y_L'])
                a_z_L_i = np.copy(mov_group[mov_keys[j]]['a_z_L'])
                a_theta_R_i = np.copy(mov_group[mov_keys[j]]['a_theta_R'])
                a_eta_R_i = np.copy(mov_group[mov_keys[j]]['a_eta_R'])
                a_phi_R_i = np.copy(mov_group[mov_keys[j]]['a_phi_R'])
                a_xi_R_i = np.copy(mov_group[mov_keys[j]]['a_xi_R'])
                a_x_R_i = np.copy(mov_group[mov_keys[j]]['a_x_R'])
                a_y_R_i = np.copy(mov_group[mov_keys[j]]['a_y_R'])
                a_z_R_i = np.copy(mov_group[mov_keys[j]]['a_z_R'])
                s1_s2_i = np.copy(mov_group[mov_keys[j]]['s1_s2'])
                T_i = np.copy(mov_group[mov_keys[j]]['T'])
                time_wb_i = np.copy(mov_group[mov_keys[j]]['time_wb'])
                freq_i = np.copy(mov_group[mov_keys[j]]['freq'])
                t_wbs_i = np.copy(mov_group[mov_keys[j]]['t_wbs'])
                if a_theta_L_i.size==0:
                    print('mov nr: '+str(j+1))
                    print('error empty array')
                else:
                    self.a_theta_L_mov.append(a_theta_L_i)
                    self.a_eta_L_mov.append(a_eta_L_i)
                    self.a_phi_L_mov.append(a_phi_L_i)
                    self.a_xi_L_mov.append(a_xi_L_i)
                    self.a_x_L_mov.append(a_x_L_i)
                    self.a_y_L_mov.append(a_y_L_i)
                    self.a_z_L_mov.append(a_z_L_i)
                    self.a_theta_R_mov.append(a_theta_R_i)
                    self.a_eta_R_mov.append(a_eta_R_i)
                    self.a_phi_R_mov.append(a_phi_R_i)
                    self.a_xi_R_mov.append(a_xi_R_i)
                    self.a_x_R_mov.append(a_x_R_i)
                    self.a_y_R_mov.append(a_y_R_i)
                    self.a_z_R_mov.append(a_z_R_i)
                    self.s1_s2_mov.append(s1_s2_i)
                    self.T_mov.append(T_i)
                    self.time_wb_mov.append(time_wb_i)
                    self.freq_mov.append(freq_i)
                    self.t_wbs_mov.append(t_wbs_i)
                    b1_i = np.copy(mov_group[mov_keys[j]]['b1_wbs'])
                    b2_i = np.copy(mov_group[mov_keys[j]]['b2_wbs'])
                    b3_i = np.copy(mov_group[mov_keys[j]]['b3_wbs'])
                    i1_i = np.copy(mov_group[mov_keys[j]]['i1_wbs'])
                    i2_i = np.copy(mov_group[mov_keys[j]]['i2_wbs'])
                    iii1_i = np.copy(mov_group[mov_keys[j]]['iii1_wbs'])
                    iii24_i = np.copy(mov_group[mov_keys[j]]['iii24_wbs'])
                    iii3_i = np.copy(mov_group[mov_keys[j]]['iii3_wbs'])
                    hg1_i = np.copy(mov_group[mov_keys[j]]['hg1_wbs'])
                    hg2_i = np.copy(mov_group[mov_keys[j]]['hg2_wbs'])
                    hg3_i = np.copy(mov_group[mov_keys[j]]['hg3_wbs'])
                    hg4_i = np.copy(mov_group[mov_keys[j]]['hg4_wbs'])
                    self.b1_wbs.append(b1_i)
                    self.b2_wbs.append(b2_i)
                    self.b3_wbs.append(b3_i)
                    self.i1_wbs.append(i1_i)
                    self.i2_wbs.append(i2_i)
                    self.iii1_wbs.append(iii1_i)
                    self.iii24_wbs.append(iii24_i)
                    self.iii3_wbs.append(iii3_i)
                    self.hg1_wbs.append(hg1_i)
                    self.hg2_wbs.append(hg2_i)
                    self.hg3_wbs.append(hg3_i)
                    self.hg4_wbs.append(hg4_i)
        self.data_file.close()

    def create_dataset(self,outliers):
        self.a_mod_theta_L = []
        self.a_mod_eta_L = []
        self.a_mod_phi_L = []
        self.a_mod_xi_L = []
        self.a_mod_x_L = []
        self.a_mod_y_L = []
        self.a_mod_z_L = []
        self.a_mod_theta_R = []
        self.a_mod_eta_R = []
        self.a_mod_phi_R = []
        self.a_mod_xi_R = []
        self.a_mod_x_R = []
        self.a_mod_y_R = []
        self.a_mod_z_R = []
        self.T_vec = []
        self.freq_vec = []
        self.ca_traces = []
        self.ca_mod_traces = []
        self.ca_spline = []
        self.N_movs_total = len(self.a_theta_L_mov)
        print('total number of movies: '+str(self.N_movs_total))
        for i in range(self.N_movs_total):
            self.a_mod_theta_L.append(np.gradient(self.a_theta_L_mov[i],axis=1))
            self.a_mod_eta_L.append(np.gradient(self.a_eta_L_mov[i],axis=1))
            self.a_mod_phi_L.append(np.gradient(self.a_phi_L_mov[i],axis=1))
            self.a_mod_xi_L.append(np.gradient(self.a_xi_L_mov[i],axis=1))
            self.a_mod_x_L.append(np.gradient(self.a_x_L_mov[i],axis=1))
            self.a_mod_y_L.append(np.gradient(self.a_y_L_mov[i],axis=1))
            self.a_mod_z_L.append(np.gradient(self.a_z_L_mov[i],axis=1))
            self.a_mod_theta_R.append(np.gradient(self.a_theta_R_mov[i],axis=1))
            self.a_mod_eta_R.append(np.gradient(self.a_eta_R_mov[i],axis=1))
            self.a_mod_phi_R.append(np.gradient(self.a_phi_R_mov[i],axis=1))
            self.a_mod_xi_R.append(np.gradient(self.a_xi_R_mov[i],axis=1))
            self.a_mod_x_R.append(np.gradient(self.a_x_R_mov[i],axis=1))
            self.a_mod_y_R.append(np.gradient(self.a_y_R_mov[i],axis=1))
            self.a_mod_z_R.append(np.gradient(self.a_z_R_mov[i],axis=1))
            n_wbs_i = self.a_theta_L_mov[i].shape[1]
            #ca_traces_i = np.zeros((12,n_wbs_i))
            ca_traces_i = np.zeros((13,n_wbs_i))
            ca_traces_i[0,:] = self.b1_wbs[i]
            ca_traces_i[1,:] = self.b2_wbs[i]
            ca_traces_i[2,:] = self.b3_wbs[i]
            ca_traces_i[3,:] = self.i1_wbs[i]
            ca_traces_i[4,:] = self.i2_wbs[i]
            ca_traces_i[5,:] = self.iii1_wbs[i]
            ca_traces_i[6,:] = self.iii24_wbs[i]
            ca_traces_i[7,:] = self.iii3_wbs[i]
            ca_traces_i[8,:] = self.hg1_wbs[i]
            ca_traces_i[9,:] = self.hg2_wbs[i]
            ca_traces_i[10,:] = self.hg3_wbs[i]
            ca_traces_i[11,:] = self.hg4_wbs[i]
            ca_traces_i[12,:] = self.freq_mov[i]
            self.ca_traces.append(ca_traces_i)
            ca_mod_traces_i = np.diff(ca_traces_i,axis=1)
            self.ca_mod_traces.append(ca_mod_traces_i)
            # spline smoothing
            ca_spline_smooth = self.spline_fit(ca_traces_i,0.2)
            self.ca_spline.append(ca_spline_smooth)
        # create dataset
        self.X_data_list = []
        self.Y_data_list = []
        self.X_mean_list = []
        self.X_gradient_list = []
        train_inds_list = []
        test_inds_list  = []
        wb_cntr = 0
        for i in range(self.N_movs_total):
            n_wbs_i = self.a_theta_L_mov[i].shape[1]
            n_half = int(np.floor(self.N_window))
            N_i = n_wbs_i-2*n_half
            if N_i>2:
                if np.sum(outliers[:,0]==i)<1:
                    X_i = np.zeros((N_i,self.N_window,13))
                    Y_i = np.zeros((N_i,80))
                    x_i = np.transpose(self.ca_traces[i])
                    x_i = self.Muscle_scale(x_i)
                    y_i = np.transpose(np.concatenate((self.a_theta_L_mov[i],self.a_eta_L_mov[i],self.a_phi_L_mov[i],self.a_xi_L_mov[i]),axis=0))
                    y_i = self.Wingkin_scale(y_i)
                    for j in range(N_i):
                        X_i[j,:,:] = x_i[j:(j+self.N_window),:]
                        Y_i[j,:] = y_i[j,:]
                        # check if frequency is within range
                        if self.freq_mov[i][j]>150 and self.freq_mov[i][j]<250:
                            # Check if wing kinematics are within range
                            wingkin_max = np.zeros(4)
                            wingkin_max[0] = (np.amax(np.abs(self.a_theta_L_mov[i][:,j]))>((60/180)*np.pi))
                            wingkin_max[1] = (np.amax(np.abs(self.a_eta_L_mov[i][:,j]))>((150/180)*np.pi))
                            wingkin_max[2] = (np.amax(np.abs(self.a_phi_L_mov[i][:,j]))>((120/180)*np.pi))
                            wingkin_max[3] = (np.amax(np.abs(self.a_xi_L_mov[i][:,j]))>((90/180)*np.pi))
                            if np.sum(wingkin_max)<1:
                                if j<30:
                                    # first 30 wingbeats are validation
                                    test_inds_list.append(wb_cntr)
                                else:
                                    # remaining wingbeats are training
                                    train_inds_list.append(wb_cntr)
                        wb_cntr = wb_cntr+1
                else:
                    outlier_ind = int(np.squeeze(np.argwhere(outliers[:,0]==i)))
                    X_i = np.zeros((N_i,self.N_window,13))
                    Y_i = np.zeros((N_i,80))
                    x_i = np.transpose(self.ca_traces[i])
                    x_i = self.Muscle_scale(x_i)
                    y_i = np.transpose(np.concatenate((self.a_theta_L_mov[i],self.a_eta_L_mov[i],self.a_phi_L_mov[i],self.a_xi_L_mov[i]),axis=0))
                    y_i = self.Wingkin_scale(y_i)
                    for j in range(N_i):
                        X_i[j,:,:] = x_i[j:(j+self.N_window),:]
                        Y_i[j,:] = y_i[j,:]
                        # check if frequency is within range
                        if j<outliers[outlier_ind,1] or j>outliers[outlier_ind,2]:
                            if self.freq_mov[i][j]>150 and self.freq_mov[i][j]<250:
                                # Check if wing kinematics are within range
                                wingkin_max = np.zeros(4)
                                wingkin_max[0] = (np.amax(np.abs(self.a_theta_L_mov[i][:,j]))>((60/180)*np.pi))
                                wingkin_max[1] = (np.amax(np.abs(self.a_eta_L_mov[i][:,j]))>((150/180)*np.pi))
                                wingkin_max[2] = (np.amax(np.abs(self.a_phi_L_mov[i][:,j]))>((120/180)*np.pi))
                                wingkin_max[3] = (np.amax(np.abs(self.a_xi_L_mov[i][:,j]))>((90/180)*np.pi))
                                if np.sum(wingkin_max)<1:
                                    if j<30:
                                        # first 30 wingbeats are validation
                                        test_inds_list.append(wb_cntr)
                                    else:
                                        # remaining wingbeats are training
                                        train_inds_list.append(wb_cntr)
                        wb_cntr = wb_cntr+1
                self.X_data_list.append(X_i)
                self.Y_data_list.append(Y_i)
                self.X_mean_list.append(np.mean(X_i,axis=1))
                self.X_gradient_list.append(np.mean(np.diff(X_i,axis=1),axis=1))
        self.X_data = np.concatenate(self.X_data_list,axis=0)
        self.Y_data = np.concatenate(self.Y_data_list,axis=0)
        self.X_mean = np.concatenate(self.X_mean_list,axis=0)
        self.X_gradient = np.concatenate(self.X_gradient_list,axis=0)
        print(self.X_data.shape)
        print(self.Y_data.shape)
        # Create training and testing set:
        self.N_wbs = self.Y_data.shape[0]
        self.unshuffled_inds = np.copy(np.array(train_inds_list))
        wb_ids_train = np.array(train_inds_list)
        np.random.shuffle(wb_ids_train)
        wb_ids_test = np.array(test_inds_list)
        np.random.shuffle(wb_ids_test)
        self.train_inds = wb_ids_train
        self.N_train     = self.train_inds.shape[0]
        print('N train: '+str(self.N_train))
        self.test_inds  = wb_ids_test
        self.N_test     = self.test_inds.shape[0]
        print('N test: '+str(self.N_test))
        self.N_test     = self.test_inds.shape[0]
        self.X_train     = self.X_data[self.train_inds,:,:]
        self.X_mean_train = self.X_mean[self.train_inds,:]
        self.Y_train     = self.Y_data[self.train_inds,:]
        self.X_test     = self.X_data[self.test_inds,:,:]
        self.X_mean_test = self.X_mean[self.test_inds,:]
        self.Y_test     = self.Y_data[self.test_inds,:]
        self.Y_wm_train = np.hstack((self.Y_train,self.X_data[self.train_inds,2,:]))
        self.Y_wm_test  = np.hstack((self.Y_test,self.X_data[self.test_inds,2,:]))

    def Muscle_scale(self,X_in):
        X_out = X_in
        X_out[:,:12] = np.clip(X_in[:,:12],-0.5,1.5)
        X_out[:,12] = (np.clip(X_in[:,12],150.0,250.0)-150.0)/100.0
        return X_out

    def Muscle_scale_inverse(self,X_in):
        X_out = X_in
        X_out[:,:12] = X_in[:,:12]
        X_out[:,12] = X_in[:,12]*100.0+150.0
        return X_out

    def Wingkin_scale(self,X_in):
        X_out = (1.0/np.pi)*np.clip(X_in,-np.pi,np.pi)
        #X_out = np.clip(X_in,-np.pi,np.pi)
        return X_out

    def Wingkin_scale_inverse(self,X_in):
        X_out = X_in*np.pi
        return X_out

    def LegendreFit(self,trace_in,b1_in,b2_in,N_pol,N_const):
        N_pts = trace_in.shape[0]
        X_Legendre = self.LegendrePolynomials(N_pts,N_pol,N_const)
        A = X_Legendre[:,:,0]
        B = np.zeros((2*N_const,N_pol))
        # data points:
        b = np.transpose(trace_in)
        # restriction vector (add zeros to smooth the connection!!!!!)
        d = np.zeros(2*N_const)
        d_gradient_1 = b1_in
        d_gradient_2 = b2_in
        for j in range(N_const):
            d[j] = d_gradient_1[4-j]*np.power(N_pts/2.0,j)
            d[N_const+j] = d_gradient_2[4-j]*np.power(N_pts/2.0,j)
            d_gradient_1 = np.diff(d_gradient_1)
            d_gradient_2 = np.diff(d_gradient_2)
            B[j,:]             = np.transpose(X_Legendre[0,:,j])
            B[N_const+j,:]     = np.transpose(X_Legendre[-1,:,j])
        # Restricted least-squares fit:
        ATA = np.dot(np.transpose(A),A)
        ATA_inv = np.linalg.inv(ATA)
        AT = np.transpose(A)
        BT = np.transpose(B)
        BATABT     = np.dot(B,np.dot(ATA_inv,BT))
        c_ls     = np.linalg.solve(ATA,np.dot(AT,b))
        c_rls     = c_ls-np.dot(ATA_inv,np.dot(BT,np.linalg.solve(BATABT,np.dot(B,c_ls)-d)))
        return c_rls

    def LegendrePolynomials(self,N_pts,N_pol,n_deriv):
        L_basis = np.zeros((N_pts,N_pol,n_deriv))
        x_basis = np.linspace(-1.0,1.0,N_pts,endpoint=True)
        for i in range(n_deriv):
            if i==0:
                # Legendre basis:
                for n in range(N_pol):
                    if n==0:
                        L_basis[:,n,i] = 1.0
                    elif n==1:
                        L_basis[:,n,i] = x_basis
                    else:
                        for k in range(n+1):
                            L_basis[:,n,i] += (1.0/np.power(2.0,n))*np.power(scipy.special.binom(n,k),2)*np.multiply(np.power(x_basis-1.0,n-k),np.power(x_basis+1.0,k))
            else:
                # Derivatives:
                for n in range(N_pol):
                    if n>=i:
                        L_basis[:,n,i] = n*L_basis[:,n-1,i-1]+np.multiply(x_basis,L_basis[:,n-1,i])
        return L_basis

    def TemporalBC(self,a_c,N_pol,N_const):
        X_Legendre = self.LegendrePolynomials(30,N_pol,N_const)
        trace = np.dot(X_Legendre[:,:,0],a_c)
        b_L = np.zeros(9)
        b_L[0:4] = trace[-5:-1]
        b_L[4] = 0.5*(trace[0]+trace[-1])
        b_L[5:9] = trace[1:5]
        b_R = np.zeros(9)
        b_R[0:4] = trace[-5:-1]
        b_R[4] = 0.5*(trace[0]+trace[-1])
        b_R[5:9] = trace[1:5]
        c_per = self.LegendreFit(trace,b_L,b_R,N_pol,N_const)
        return c_per

    def spline_fit(self,data_in,smoothing):
        n = data_in.shape[1]
        n_range = np.arange(n)
        m = data_in.shape[0]
        fit_out = np.zeros(data_in.shape)
        for j in range(m):
            tck = interpolate.splrep(n_range,data_in[j,:],s=smoothing)
            ynew = interpolate.splev(n_range, tck, der=0)
            fit_out[j,:] = ynew
        return fit_out

    def build_network(self,N_filters,N_latent):

        l2_norm = 1e-5

        input_enc = Input(shape=(self.N_window,13,1))

        # Split per sclerite

        b_enc   = Lambda(lambda x: tf.slice(x,(0,0,0,0),(-1,-1,3,1)))(input_enc)
        i_enc   = Lambda(lambda x: tf.slice(x,(0,0,3,0),(-1,-1,2,1)))(input_enc)
        iii_enc = Lambda(lambda x: tf.slice(x,(0,0,5,0),(-1,-1,3,1)))(input_enc)
        hg_enc  = Lambda(lambda x: tf.slice(x,(0,0,8,0),(-1,-1,4,1)))(input_enc)
        f_enc   = Lambda(lambda x: tf.slice(x,(0,0,12,0),(-1,-1,1,1)))(input_enc)

        # b-encoder:
        b_enc = Conv2D(filters=N_filters,kernel_size=(self.N_window,1),strides=(self.N_window,1),activation='selu')(b_enc)
        b_enc = Conv2D(filters=4*N_filters,kernel_size=(1,3),strides=(1,3),activation='selu')(b_enc)
        b_enc = Flatten()(b_enc)
        b_enc = Dense(N_filters*8,activation='selu')(b_enc)
        b_enc = Dense(1,activation='linear',use_bias=False)(b_enc)

        # i-encoder:
        i_enc = Conv2D(filters=N_filters,kernel_size=(self.N_window,1),strides=(self.N_window,1),activation='selu')(i_enc)
        i_enc = Conv2D(filters=4*N_filters,kernel_size=(1,2),strides=(1,2),activation='selu')(i_enc)
        i_enc = Flatten()(i_enc)
        i_enc = Dense(N_filters*8,activation='selu')(i_enc)
        i_enc = Dense(1,activation='linear',use_bias=False)(i_enc)

        # iii-encoder:
        iii_enc = Conv2D(filters=N_filters,kernel_size=(self.N_window,1),strides=(self.N_window,1),activation='selu')(iii_enc)
        iii_enc = Conv2D(filters=4*N_filters,kernel_size=(1,3),strides=(1,3),activation='selu')(iii_enc)
        iii_enc = Flatten()(iii_enc)
        iii_enc = Dense(N_filters*8,activation='selu')(iii_enc)
        iii_enc = Dense(1,activation='linear',use_bias=False)(iii_enc)

        # hg-encoder:
        hg_enc = Conv2D(filters=N_filters,kernel_size=(self.N_window,1),strides=(self.N_window,1),activation='selu')(hg_enc)
        hg_enc = Conv2D(filters=4*N_filters,kernel_size=(1,4),strides=(1,4),activation='selu')(hg_enc)
        hg_enc = Flatten()(hg_enc)
        hg_enc = Dense(N_filters*8,activation='selu')(hg_enc)
        hg_enc = Dense(1,activation='linear',use_bias=False)(hg_enc)

        # f-encoder
        f_enc = Conv2D(filters=N_filters,kernel_size=(self.N_window,1),strides=(self.N_window,1),activation='selu')(f_enc)
        f_enc = Conv2D(filters=4*N_filters,kernel_size=(1,1),strides=(1,1),activation='selu')(f_enc)
        f_enc = Flatten()(f_enc)
        f_enc = Dense(N_filters*8,activation='selu')(f_enc)
        f_enc = Dense(1,activation='linear',use_bias=False)(f_enc)

        # Concatenate:
        encoder_output = Concatenate()([b_enc,i_enc,iii_enc,hg_enc,f_enc])

        input_dec = Input(shape=(5,))

        dec1 = Dropout(0.5)(input_dec)
        dec1 = Dense(1024,activation='selu')(dec1)
        dec1 = Dropout(0.5)(dec1)
        decoded1 = Dense(80,activation='linear')(dec1)

        N_neurons = 2

        b_dec   = Lambda(lambda x: tf.slice(x,(0,0),(-1,1)))(input_dec)
        b_dec     = Lambda(lambda x: K.stop_gradient(x))(b_dec)
        b_dec     = Dense(3*N_neurons,activation='tanh')(b_dec)
        b_dec     = Reshape((1,3,N_neurons))(b_dec)

        i_dec   = Lambda(lambda x: tf.slice(x,(0,1),(-1,1)))(input_dec)
        i_dec     = Lambda(lambda x: K.stop_gradient(x))(i_dec)
        i_dec     = Dense(2*N_neurons,activation='tanh')(i_dec)
        i_dec     = Reshape((1,2,N_neurons))(i_dec)

        iii_dec = Lambda(lambda x: tf.slice(x,(0,2),(-1,1)))(input_dec)
        iii_dec = Lambda(lambda x: K.stop_gradient(x))(iii_dec)
        iii_dec = Dense(3*N_neurons,activation='tanh')(iii_dec)
        iii_dec = Reshape((1,3,N_neurons))(iii_dec)

        hg_dec  = Lambda(lambda x: tf.slice(x,(0,3),(-1,1)))(input_dec)
        hg_dec  = Lambda(lambda x: K.stop_gradient(x))(hg_dec)
        hg_dec     = Dense(4*N_neurons,activation='tanh')(hg_dec)
        hg_dec     = Reshape((1,4,N_neurons))(hg_dec)

        f_dec   = Lambda(lambda x: tf.slice(x,(0,4),(-1,1)))(input_dec)
        f_dec   = Lambda(lambda x: K.stop_gradient(x))(f_dec)
        f_dec     = Dense(1*N_neurons,activation='tanh')(f_dec)
        f_dec     = Reshape((1,1,N_neurons))(f_dec)

        dec2 = Concatenate(axis=2)([b_dec,i_dec,iii_dec,hg_dec,f_dec])
        dec2 = Dropout(0.5)(dec2)
        dec2 = Conv2DTranspose(N_filters,(self.N_window,1),strides=(1,1),input_shape=(1,13,N_neurons),activation='selu')(dec2)
        min_max = MinMaxNorm(min_value=0.0, max_value=1.0)
        decoded2 = Conv2DTranspose(1,(1,1),strides=(1,1),input_shape=(self.N_window,13,N_filters),kernel_constraint=min_max,bias_constraint=min_max,activation='linear',name='muscle_trace')(dec2)

        encoder_model             = Model(inputs=input_enc,outputs=encoder_output)
        wingkin_decoder_model     = Model(inputs=input_dec,outputs=decoded1,name='wingkin')
        muscle_decoder_model     = Model(inputs=input_dec,outputs=decoded2,name='muscle')

        encoder_decoder1_mdl = muscle_decoder_model(encoder_model(input_enc))
        encoder_decoder2_mdl = wingkin_decoder_model(encoder_model(input_enc))

        model = Model(inputs=input_enc,outputs=[encoder_decoder1_mdl,encoder_decoder2_mdl])

        model_list = [model,encoder_model,wingkin_decoder_model,muscle_decoder_model]
        
        return model_list

    def load_network(self,N_filters,N_latent,weights_fldr_in):
        self.train_m = True
        self.train_w = True
        self.weights_fldr = weights_fldr_in

        os.chdir(self.weights_fldr)
        weight_file   = 'network_latent_'+str(N_latent)+'.h5'
        weight_file_1 = 'encoder_weights_latent_'+str(N_latent)+'.h5'
        weight_file_2 = 'wingkin_decoder_weights_latent_'+str(N_latent)+'.h5'
        weight_file_3 = 'muscle_decoder_weights_latent_'+str(N_latent)+'.h5'

        model_list = self.build_network(N_filters,N_latent)

        self.network                     = model_list[0]
        self.encoder_network             = model_list[1]
        self.wingkin_decoder_network     = model_list[2]
        self.muscle_decoder_network     = model_list[3]

        self.network.load_weights(weight_file)
        self.network.summary()

        self.encoder_network.load_weights(weight_file_1)
        self.encoder_network.summary()

        self.wingkin_decoder_network.load_weights(weight_file_2)
        self.wingkin_decoder_network.summary()

        self.muscle_decoder_network.load_weights(weight_file_3)
        self.muscle_decoder_network.summary()

    def train_network(self,N_filters,N_latent,weights_fldr_in):

        self.weights_fldr = weights_fldr_in

        self.train_m = True
        self.train_w = True

        model_list = self.build_network(N_filters,N_latent)

        self.network = model_list[0]
        self.encoder_network = model_list[1]
        self.wingkin_decoder_network = model_list[2]
        self.muscle_decoder_network = model_list[3]

        self.network.summary()
        self.encoder_network.summary()
        self.wingkin_decoder_network.summary()
        self.muscle_decoder_network.summary()

        N_epochs = 200
        lr = 1.0e-4
        decay = lr/N_epochs
        batch_size = 100
        muscle_mse_weight = 50
        muscle_loss_scaling  = 1.0
        wingkin_loss_scaling = 1.0


        self.network.compile(
                loss={'wingkin':'mse','muscle':'mse'},
                optimizer=tf.optimizers.Adam(lr=lr,decay=decay),
                metrics={'wingkin':'mse','muscle':'mse'}
                )

        history = self.network.fit(
                x=self.X_train,
                y={'muscle':self.X_train,'wingkin':self.Y_train},
                epochs=N_epochs,
                shuffle=True,
                batch_size=batch_size,
                validation_data=(self.X_test,{'muscle':self.X_test,'wingkin':self.Y_test}),
                verbose=1
                )

        os.chdir(self.weights_fldr)
        weight_file   = 'network_latent_'+str(N_latent)+'.h5'
        weight_file_1 = 'encoder_weights_latent_'+str(N_latent)+'.h5'
        weight_file_2 = 'wingkin_decoder_weights_latent_'+str(N_latent)+'.h5'
        weight_file_3 = 'muscle_decoder_weights_latent_'+str(N_latent)+'.h5'
        weight_file_4 = 'm_trace_weights_latent_'+str(N_latent)+'.h5'

        self.network.save_weights(weight_file)
        self.encoder_network.save_weights(weight_file_1)
        self.wingkin_decoder_network.save_weights(weight_file_2)
        self.muscle_decoder_network.save_weights(weight_file_3)

    def predict(self,X_in):
        prediction = self.network.predict(X_in)
        return prediction

    def encode(self,X_in):
        encoded = self.encoder_network.predict(X_in)
        return encoded 

    def decode_wingkin(self,X_in):
        decoded = self.wingkin_decoder_network.predict(X_in)
        return decoded

    def decode_muscle(self,X_in):
        decoded = self.muscle_decoder_network.predict(X_in)
        return decoded

    def q_mult(self,qA,qB):

        try:
            nA = qA.shape[1]
        except:
            nA = 1

        try:
            nB = qB.shape[1]
        except:
            nB = 1

        if nA==1 and nB==1:
            QA = np.squeeze(np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
                [qA[1],qA[0],-qA[3],qA[2]],
                [qA[2],qA[3],qA[0],-qA[1]],
                [qA[3],-qA[2],qA[1],qA[0]]]))
            qC = np.dot(QA,qB)
            qC_norm = math.sqrt(pow(qC[0],2)+pow(qC[1],2)+pow(qC[2],2)+pow(qC[3],2))
            if qC_norm>0.01:
                qC /= qC_norm
            else:
                qC = np.array([1.0,0.0,0.0,0.0])
        elif nA==1 and nB>1:
            QA = np.squeeze(np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
                [qA[1],qA[0],-qA[3],qA[2]],
                [qA[2],qA[3],qA[0],-qA[1]],
                [qA[3],-qA[2],qA[1],qA[0]]]))
            qC = np.zeros((4,nB))
            for i in range(nB):
                qCi = np.dot(QA,qB[:,i])
                qC_norm = math.sqrt(pow(qCi[0],2)+pow(qCi[1],2)+pow(qCi[2],2)+pow(qCi[3],2))
                if qC_norm>0.01:
                    qCi /= qC_norm
                else:
                    qCi = np.array([1.0,0.0,0.0,0.0])
                qC[:,i] = qCi        
        else:
            qC = np.zeros((4,nA))
            for i in range(nA):
                QAi = np.array([[qA[0,i],-qA[1,i],-qA[2,i],-qA[3,i]],
                    [qA[1,i],qA[0,i],-qA[3,i],qA[2,i]],
                    [qA[2,i],qA[3,i],qA[0,i],-qA[1,i]],
                    [qA[3,i],-qA[2,i],qA[1,i],qA[0,i]]])
                qCi = np.dot(QAi,qB[:,i])
                qC_norm = math.sqrt(pow(qCi[0],2)+pow(qCi[1],2)+pow(qCi[2],2)+pow(qCi[3],2))
                if qC_norm>0.01:
                    qCi /= qC_norm
                else:
                    qCi = np.array([1.0,0.0,0.0,0.0])
                qC[:,i] = qCi
        return qC

    def comp_R(self,q):
        R = np.squeeze(np.array([[2*pow(q[0],2)-1+2*pow(q[1],2), 2*q[1]*q[2]+2*q[0]*q[3], 2*q[1]*q[3]-2*q[0]*q[2]],
            [2*q[1]*q[2]-2*q[0]*q[3], 2*pow(q[0],2)-1+2*pow(q[2],2), 2*q[2]*q[3]+2*q[0]*q[1]],
            [2*q[1]*q[3]+2*q[0]*q[2], 2*q[2]*q[3]-2*q[0]*q[1], 2*pow(q[0],2)-1+2*pow(q[3],2)]]))
        return R

    def quat_mat(self,s_in):
        q0 = np.squeeze(s_in[0])
        q1 = np.squeeze(s_in[1])
        q2 = np.squeeze(s_in[2])
        q3 = np.squeeze(s_in[3])
        tx = np.squeeze(s_in[4])
        ty = np.squeeze(s_in[5])
        tz = np.squeeze(s_in[6])
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        return M

    def plot_latent_modes(self,N_latent,plot_loc):

        N_steps = 9

        X_latent  = self.encode(self.X_data)

        Z_latent = zscore(X_latent,axis=0)

        X_muscle = self.decode_muscle(X_latent)

        muscle_mean = np.mean(X_muscle[:,2,:],axis=0)
        muscle_std  = np.std(X_muscle[:,2,:],axis=0)
        muscle_min  = np.amin(X_muscle[:,2,:],axis=0)
        muscle_max  = np.amax(X_muscle[:,2,:],axis=0)

        br_palette = sns.color_palette("coolwarm", N_steps)

        os.chdir(plot_loc)
        sns.palplot(br_palette)
        plt.savefig('color_bar.svg',dpi=300)
        #fig2 = color_bar_plot.get_figure()
        #fig2.savefig('color_bar.svg', dpi=300)

        mode_names = ['b_','i_','iii_','hg_','f_']

        line_width = 1.0

        sigma_range = np.linspace(-3,3,num=N_steps)

        t_window = np.linspace(0,0.005*self.N_window,num=self.N_window)

        t = np.linspace(0,1,num=100)
        X_theta = self.LegendrePolynomials(100,self.N_pol_theta,3)
        X_eta = self.LegendrePolynomials(100,self.N_pol_eta,3)
        X_phi = self.LegendrePolynomials(100,self.N_pol_phi,3)
        X_xi = self.LegendrePolynomials(100,self.N_pol_xi,3)

        X_theta2 = self.LegendrePolynomials(1000,self.N_pol_theta,3)
        X_eta2 = self.LegendrePolynomials(1000,self.N_pol_eta,3)
        X_phi2 = self.LegendrePolynomials(1000,self.N_pol_phi,3)
        X_xi2 = self.LegendrePolynomials(1000,self.N_pol_xi,3)

        fig = plt.figure()
        fig.set_size_inches(N_latent*4,18)
        gs = GridSpec(5,N_latent)

        m_trend_i = np.zeros((13,N_steps))

        beta = (-60.0/180.0)*np.pi

        for i in range(N_latent):

            # create axes
            ax_m       = fig.add_subplot(gs[0,i])
            ax_phi   = fig.add_subplot(gs[1,i])
            ax_theta = fig.add_subplot(gs[2,i])
            ax_eta   = fig.add_subplot(gs[3,i])
            ax_xi    = fig.add_subplot(gs[4,i])

            mean_i = np.mean(X_latent[:,i])
            std_i = np.std(X_latent[:,i])
            min_i = np.amin(X_latent[:,i])
            max_i = np.amax(X_latent[:,i])

            z_range = np.linspace(min_i,max_i,num=N_steps)

            Z_i = np.zeros((N_steps,N_latent))
            Z_i[:,i] = z_range

            M_i = np.squeeze(self.decode_muscle(Z_i))

            Y_i = self.decode_wingkin(Z_i)

            phi_i   = self.Wingkin_scale_inverse(np.dot(X_phi[:,:,0],np.transpose(Y_i[:,44:60])))*(180.0/np.pi)
            theta_i = self.Wingkin_scale_inverse(np.dot(X_theta[:,:,0],np.transpose(Y_i[:,0:20])))*(180.0/np.pi)
            eta_i   = self.Wingkin_scale_inverse(np.dot(X_eta[:,:,0],np.transpose(Y_i[:,20:44])))*(180.0/np.pi)
            xi_i    = self.Wingkin_scale_inverse(np.dot(X_xi[:,:,0],np.transpose(Y_i[:,60:80])))*(180.0/np.pi)

            bins_i = np.linspace(min_i,max_i,num=N_steps)
            inds_i = np.digitize(X_latent[:,i],bins_i,right=True)

            # Sort on minimum phi:
            min_phi_ind = np.argmin(np.amin(phi_i,axis=0))

            z_order = [1,-1,1,-1,1]

            wingkin = np.zeros((8,1000))

            if z_order[i] == 1:
                for j in range(N_steps):

                    ax_phi.plot(t,phi_i[:,j],color=br_palette[j],linewidth=line_width)
                    ax_theta.plot(t,theta_i[:,j],color=br_palette[j],linewidth=line_width)
                    ax_eta.plot(t,eta_i[:,j],color=br_palette[j],linewidth=line_width)
                    ax_xi.plot(t,xi_i[:,j],color=br_palette[j],linewidth=line_width)

                    m_trend_i[0,j]     = M_i[j,2,0]
                    m_trend_i[1,j]     = M_i[j,2,1]
                    m_trend_i[2,j]     = M_i[j,2,2]
                    m_trend_i[3,j]     = M_i[j,2,3]
                    m_trend_i[4,j]     = M_i[j,2,4]
                    m_trend_i[5,j]     = M_i[j,2,5]
                    m_trend_i[6,j]     = M_i[j,2,6]
                    m_trend_i[7,j]     = M_i[j,2,7]
                    m_trend_i[8,j]     = M_i[j,2,8]
                    m_trend_i[9,j]     = M_i[j,2,9]
                    m_trend_i[10,j] = M_i[j,2,10]
                    m_trend_i[11,j] = M_i[j,2,11]
                    m_trend_i[12,j] = M_i[j,2,12]

                    wingkin[0,:] = self.Wingkin_scale_inverse(np.dot(X_theta2[:,:,0],np.transpose(Y_i[j,0:20])))
                    wingkin[1,:] = self.Wingkin_scale_inverse(np.dot(X_eta2[:,:,0],np.transpose(Y_i[j,20:44])))
                    wingkin[2,:] = self.Wingkin_scale_inverse(np.dot(X_phi2[:,:,0],np.transpose(Y_i[j,44:60])))
                    wingkin[3,:] = self.Wingkin_scale_inverse(np.dot(X_xi2[:,:,0],np.transpose(Y_i[j,60:80])))
                    wingkin[4,:] = self.Wingkin_scale_inverse(np.dot(X_theta2[:,:,0],np.transpose(Y_i[j,0:20])))
                    wingkin[5,:] = self.Wingkin_scale_inverse(np.dot(X_eta2[:,:,0],np.transpose(Y_i[j,20:44])))
                    wingkin[6,:] = self.Wingkin_scale_inverse(np.dot(X_phi2[:,:,0],np.transpose(Y_i[j,44:60])))
                    wingkin[7,:] = self.Wingkin_scale_inverse(np.dot(X_xi2[:,:,0],np.transpose(Y_i[j,60:80])))

                    plot_name = mode_names[i]+str(j+1)
                    save_loc = plot_loc /'lollipop'

                    self.make_lollipop_figure(wingkin,plot_name,save_loc,beta,br_palette[j])

                if i==0:
                    ax_m.plot(sigma_range,m_trend_i[0,:],color=self.c_muscle[0],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[1,:],color=self.c_muscle[1],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[2,:],color=self.c_muscle[2],linewidth=line_width)
                elif i==1:
                    ax_m.plot(sigma_range,m_trend_i[3,:],color=self.c_muscle[3],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[4,:],color=self.c_muscle[4],linewidth=line_width)
                elif i==2:
                    ax_m.plot(sigma_range,m_trend_i[5,:],color=self.c_muscle[5],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[6,:],color=self.c_muscle[6],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[7,:],color=self.c_muscle[7],linewidth=line_width)
                elif i==3:
                    ax_m.plot(sigma_range,m_trend_i[8,:],color=self.c_muscle[8],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[9,:],color=self.c_muscle[9],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[10,:],color=self.c_muscle[10],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[11,:],color=self.c_muscle[11],linewidth=line_width)
                elif i==4:
                    ax_m.plot(sigma_range,m_trend_i[12,:],color=self.c_muscle[12],linewidth=line_width)

            else:
                for j in range(N_steps-1,-1,-1):

                    ax_phi.plot(t,phi_i[:,j],color=br_palette[N_steps-1-j],linewidth=line_width)
                    ax_theta.plot(t,theta_i[:,j],color=br_palette[N_steps-1-j],linewidth=line_width)
                    ax_eta.plot(t,eta_i[:,j],color=br_palette[N_steps-1-j],linewidth=line_width)
                    ax_xi.plot(t,xi_i[:,j],color=br_palette[N_steps-1-j],linewidth=line_width)

                    m_trend_i[0,N_steps-1-j]     = M_i[j,2,0]
                    m_trend_i[1,N_steps-1-j]     = M_i[j,2,1]
                    m_trend_i[2,N_steps-1-j]     = M_i[j,2,2]
                    m_trend_i[3,N_steps-1-j]     = M_i[j,2,3]
                    m_trend_i[4,N_steps-1-j]     = M_i[j,2,4]
                    m_trend_i[5,N_steps-1-j]     = M_i[j,2,5]
                    m_trend_i[6,N_steps-1-j]     = M_i[j,2,6]
                    m_trend_i[7,N_steps-1-j]     = M_i[j,2,7]
                    m_trend_i[8,N_steps-1-j]     = M_i[j,2,8]
                    m_trend_i[9,N_steps-1-j]     = M_i[j,2,9]
                    m_trend_i[10,N_steps-1-j] = M_i[j,2,10]
                    m_trend_i[11,N_steps-1-j] = M_i[j,2,11]
                    m_trend_i[12,N_steps-1-j] = M_i[j,2,12]

                    wingkin[0,:] = self.Wingkin_scale_inverse(np.dot(X_theta2[:,:,0],np.transpose(Y_i[j,0:20])))
                    wingkin[1,:] = self.Wingkin_scale_inverse(np.dot(X_eta2[:,:,0],np.transpose(Y_i[j,20:44])))
                    wingkin[2,:] = self.Wingkin_scale_inverse(np.dot(X_phi2[:,:,0],np.transpose(Y_i[j,44:60])))
                    wingkin[3,:] = self.Wingkin_scale_inverse(np.dot(X_xi2[:,:,0],np.transpose(Y_i[j,60:80])))
                    wingkin[4,:] = self.Wingkin_scale_inverse(np.dot(X_theta2[:,:,0],np.transpose(Y_i[j,0:20])))
                    wingkin[5,:] = self.Wingkin_scale_inverse(np.dot(X_eta2[:,:,0],np.transpose(Y_i[j,20:44])))
                    wingkin[6,:] = self.Wingkin_scale_inverse(np.dot(X_phi2[:,:,0],np.transpose(Y_i[j,44:60])))
                    wingkin[7,:] = self.Wingkin_scale_inverse(np.dot(X_xi2[:,:,0],np.transpose(Y_i[j,60:80])))

                    plot_name = mode_names[i]+str(N_steps-j)
                    save_loc = plot_loc / 'lollipop'

                    self.make_lollipop_figure(wingkin,plot_name,save_loc,beta,br_palette[N_steps-1-j])

                if i==0:
                    ax_m.plot(sigma_range,m_trend_i[0,:],color=self.c_muscle[0],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[1,:],color=self.c_muscle[1],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[2,:],color=self.c_muscle[2],linewidth=line_width)
                elif i==1:
                    ax_m.plot(sigma_range,m_trend_i[3,:],color=self.c_muscle[3],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[4,:],color=self.c_muscle[4],linewidth=line_width)
                elif i==2:
                    ax_m.plot(sigma_range,m_trend_i[5,:],color=self.c_muscle[5],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[6,:],color=self.c_muscle[6],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[7,:],color=self.c_muscle[7],linewidth=line_width)
                elif i==3:
                    ax_m.plot(sigma_range,m_trend_i[8,:],color=self.c_muscle[8],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[9,:],color=self.c_muscle[9],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[10,:],color=self.c_muscle[10],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[11,:],color=self.c_muscle[11],linewidth=line_width)
                elif i==4:
                    ax_m.plot(sigma_range,m_trend_i[12,:],color=self.c_muscle[12],linewidth=line_width)

            ax_m.axvline(x=0,color=(0.5,0.5,0.5),linewidth=0.5)

            ax_phi.set_xlim([0,1])
            ax_theta.set_xlim([0,1])
            ax_eta.set_xlim([0,1])
            ax_xi.set_xlim([0,1])

            ax_m.set_xlim([-3,3])

            ax_phi.set_ylim([-90,120])
            ax_theta.set_ylim([-45,45])
            ax_eta.set_ylim([-150,90])
            ax_xi.set_ylim([-60,60])

            ax_m.set_ylim([-0.1,1.1])

            if i==0:
                adjust_spines(ax_phi,['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
                adjust_spines(ax_theta,['left'],yticks=[-30,0,30],linewidth=0.8,spineColor='k')
                adjust_spines(ax_eta,['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
                adjust_spines(ax_xi,['left','bottom'],xticks=[0,1],yticks=[-45,0,45],linewidth=0.8,spineColor='k')
                adjust_spines(ax_m,['left','bottom'],xticks=[-3,0,3],yticks=[0,1],linewidth=0.8,spineColor='k')
            else:
                adjust_spines(ax_phi,[],linewidth=0.8,spineColor='k')
                adjust_spines(ax_theta,[],linewidth=0.8,spineColor='k')
                adjust_spines(ax_eta,[],linewidth=0.8,spineColor='k')
                adjust_spines(ax_xi,['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(ax_m,['bottom'],xticks=[-3,0,3],linewidth=0.8,spineColor='k')

        os.chdir(plot_loc)
        file_name = 'latent_modes.svg'
        fig.savefig(file_name, dpi=300)

    def compute_FT(self,a_phi_in,a_theta_in,a_eta_in,a_xi_in,f_in):

        X_theta = self.LegendrePolynomials(100,self.N_pol_theta,3)
        X_eta = self.LegendrePolynomials(100,self.N_pol_eta,3)
        X_phi = self.LegendrePolynomials(100,self.N_pol_phi,3)
        X_xi = self.LegendrePolynomials(100,self.N_pol_xi,3)

        phi_i = np.dot(X_phi[:,:,0],a_phi_in)
        theta_i = -np.dot(X_theta[:,:,0],a_theta_in)
        eta_i = np.dot(X_eta[:,:,0],a_eta_in)
        phi_dot_i = np.dot(X_phi[:,:,1],a_phi_in)*f_in
        theta_dot_i = -np.dot(X_theta[:,:,1],a_theta_in)*f_in
        eta_dot_i = np.dot(X_eta[:,:,1],a_eta_in)*f_in
        phi_ddot_i = np.dot(X_phi[:,:,2],a_phi_in)*f_in*f_in
        theta_ddot_i = -np.dot(X_theta[:,:,2],a_theta_in)*f_in*f_in
        eta_ddot_i = np.dot(X_eta[:,:,2],a_eta_in)*f_in*f_in

        q_beta = np.array([np.cos(-np.pi/4.0),0,np.sin(-np.pi/4.0),0])

        R_beta = self.comp_R(q_beta)

        N_i = phi_i.shape[0]

        alpha = np.zeros(N_i)
        U = np.zeros(N_i)
        L = np.zeros(N_i)
        D = np.zeros(N_i)

        FT_A = np.zeros((6,N_i))
        FT_I = np.zeros((6,N_i))
        FT_I_acc = np.zeros((6,N_i))
        FT_I_vel = np.zeros((6,N_i))

        rho_air     = 1.18e-9
        S_yy         = 5.313
        S_xx         = 0.2932
        S_xx_asym     = -0.2198
        c_mean         = 0.946

        wing_cg_L = np.array([-0.1769,1.4362,0.0])

        MwL = np.array([[1.853e-9,0,0,0,0,-2.565e-9],
            [0,1.853e-9,0,0,0,-3.184e-10],
            [0,0,1.853e-9,2.565e-9,3.184e-10,0],
            [0,0,2.565e-9,7.931e-9,6.940e-10,0],
            [0,0,3.184e-10,6.940e-10,4.491e-10,0],
            [-2.565e-9,-3.184e-10,0,0,0,8.380e-9]])

        omega = np.zeros((3,N_i))
        omega_dot = np.zeros((3,N_i))

        for i in range(N_i):
            q_phi     = np.array([[np.cos(phi_i[i]/2)],[np.sin(phi_i[i]/2)],[0],[0]])
            q_theta = np.array([[np.cos(theta_i[i]/2)],[0],[0],[np.sin(theta_i[i]/2)]])
            q_eta     = np.array([[np.cos(eta_i[i]/2)],[0],[np.sin(eta_i[i]/2)],[0]])
            phi_dot_vec     = np.array([[phi_dot_i[i]],[0],[0]])
            theta_dot_vec     = np.array([[0],[0],[theta_dot_i[i]]])
            eta_dot_vec     = np.array([[0],[eta_dot_i[i]],[0]])
            phi_ddot_vec     = np.array([[phi_ddot_i[i]],[0],[0]])
            theta_ddot_vec     = np.array([[0],[0],[theta_ddot_i[i]]])
            eta_ddot_vec     = np.array([[0],[eta_ddot_i[i]],[0]])

            R_L = self.comp_R(self.q_mult(self.q_mult(q_phi,q_theta),q_eta))
            w_L = np.squeeze(np.dot(self.comp_R(self.q_mult(q_eta,q_theta)),phi_dot_vec)+np.dot(self.comp_R(q_eta),theta_dot_vec)+eta_dot_vec)
            w_dot_L = np.squeeze(np.dot(np.transpose(self.comp_R(self.q_mult(q_eta,q_theta))),phi_ddot_vec)+np.dot(np.transpose(self.comp_R(q_eta)),theta_ddot_vec)+eta_ddot_vec)

            omega[:,i] = w_L
            omega_dot[:,i] = w_dot_L

            # Aerodynamic forces:
            alpha[i] = np.arctan2(-w_L[0],-w_L[2])
            U[i] = 2.7*np.sqrt(w_L[0]**2+w_L[2]**2)
            CL = (0.225+1.58*np.sin(2.13*np.abs(alpha[i])-7.2/180.0*np.pi))
            CD = (1.92-1.55*np.cos(2.04*np.abs(alpha[i])-9.82/180.0*np.pi))
            CR = 2.08
            F_rot = CR*rho_air*(np.sqrt(S_xx*S_yy)*np.sqrt(w_L[0]**2+w_L[2]**2)*w_L[1]+S_xx_asym*np.sign(w_L[1])*w_L[1]**2)
            L_i = 0.5*rho_air*CL*S_yy*(w_L[0]**2+w_L[2]**2)
            D_i = 0.5*rho_air*CD*S_yy*(w_L[0]**2+w_L[2]**2)
            F_i = np.array([[np.sin(alpha[i])*L_i-np.cos(alpha[i])*D_i],[0],[np.cos(alpha[i])*L_i+np.sin(alpha[i])*D_i+F_rot]])
            L[i] = L_i+np.cos(alpha[i])*F_rot
            D[i] = D_i+np.sin(alpha[i])*F_rot

            x_cp_L = (0.82*np.abs(alpha[i])/np.pi-0.2)*c_mean

            wing_cp_L = np.array([x_cp_L,0.7*2.7,0.0])

            cp_cross = np.array([[0, -wing_cp_L[2], wing_cp_L[1]],[wing_cp_L[2],0,-wing_cp_L[0]],[-wing_cp_L[1],wing_cp_L[0],0]])

            FT_A[:3,i] = np.squeeze(F_i)
            FT_A[3:,i] = np.squeeze(np.dot(cp_cross,F_i))

            w_L_cross = np.array([[0.0,-w_L[2],w_L[1]],[w_L[2],0.0,-w_L[0]],[-w_L[1],w_L[0],0.0]])

            FT_I_acc[:3,i] = np.dot(-MwL[:3,3:],w_dot_L)
            FT_I_acc[3:,i] = np.dot(-MwL[3:,3:],w_dot_L)

            FT_I_vel[:3,i] = np.squeeze(-MwL[0,0]*np.dot(w_L_cross,np.dot(w_L_cross,wing_cg_L)))
            FT_I_vel[3:,i] = np.squeeze(-np.dot(w_L_cross,np.dot(MwL[3:,3:],w_L)))

            FT_I[:3,i] = FT_I_acc[:3,i]+FT_I_vel[:3,i]
            FT_I[3:,i] = FT_I_acc[3:,i]+FT_I_vel[3:,i]

        #return FT_A, FT_I, FT_I_acc, FT_I_vel, omega, omega_dot
        return alpha, U, L, D, omega, omega_dot

    def plot_LD_latent_modes(self,N_latent,plot_loc):

        N_steps = 9

        X_latent  = self.encode(self.X_data)

        Z_latent = zscore(X_latent,axis=0)

        X_muscle = self.decode_muscle(X_latent)

        muscle_mean = np.mean(X_muscle[:,2,:],axis=0)
        muscle_std  = np.std(X_muscle[:,2,:],axis=0)
        muscle_min  = np.amin(X_muscle[:,2,:],axis=0)
        muscle_max  = np.amax(X_muscle[:,2,:],axis=0)

        br_palette = sns.color_palette("coolwarm", N_steps)

        mode_names = ['b_','i_','iii_','hg_','f_']

        line_width = 1.0

        sigma_range = np.linspace(-3,3,num=N_steps)

        t_window = np.linspace(0,0.005*self.N_window,num=self.N_window)

        t = np.linspace(0,1,num=100)
        X_theta = self.LegendrePolynomials(100,self.N_pol_theta,3)
        X_eta = self.LegendrePolynomials(100,self.N_pol_eta,3)
        X_phi = self.LegendrePolynomials(100,self.N_pol_phi,3)
        X_xi = self.LegendrePolynomials(100,self.N_pol_xi,3)

        X_theta2 = self.LegendrePolynomials(1000,self.N_pol_theta,3)
        X_eta2 = self.LegendrePolynomials(1000,self.N_pol_eta,3)
        X_phi2 = self.LegendrePolynomials(1000,self.N_pol_phi,3)
        X_xi2 = self.LegendrePolynomials(1000,self.N_pol_xi,3)

        fig = plt.figure()
        fig.set_size_inches(N_latent*4,18)
        gs = GridSpec(5,N_latent)

        m_trend_i = np.zeros((13,N_steps))

        for i in range(N_latent):

            # create axes
            ax_m       = fig.add_subplot(gs[0,i])
            ax_phi   = fig.add_subplot(gs[1,i])
            ax_theta = fig.add_subplot(gs[2,i])
            ax_eta   = fig.add_subplot(gs[3,i])
            ax_xi    = fig.add_subplot(gs[4,i])

            mean_i = np.mean(X_latent[:,i])
            std_i = np.std(X_latent[:,i])
            min_i = np.amin(X_latent[:,i])
            max_i = np.amax(X_latent[:,i])

            z_range = np.linspace(min_i,max_i,num=N_steps)

            Z_i = np.zeros((N_steps,N_latent))
            Z_i[:,i] = z_range

            M_i = np.squeeze(self.decode_muscle(Z_i))

            Y_i = self.decode_wingkin(Z_i)

            Z_0 = np.zeros((N_steps,N_latent))

            Y_0 = self.decode_wingkin(Z_0)

            phi_i   = self.Wingkin_scale_inverse(np.dot(X_phi[:,:,0],np.transpose(Y_i[:,44:60])))*(180.0/np.pi)

            a_phi_i   = np.transpose(self.Wingkin_scale_inverse(Y_i[:,44:60]))
            #a_phi_i   = np.transpose(self.Wingkin_scale_inverse(Y_0[:,44:60]))
            a_theta_i = np.transpose(self.Wingkin_scale_inverse(Y_i[:,0:20]))
            a_eta_i   = np.transpose(self.Wingkin_scale_inverse(Y_i[:,20:44]))
            a_xi_i    = np.transpose(self.Wingkin_scale_inverse(Y_i[:,60:80]))

            bins_i = np.linspace(min_i,max_i,num=N_steps)
            inds_i = np.digitize(X_latent[:,i],bins_i,right=True)

            # Sort on minimum phi:
            min_phi_ind = np.argmin(np.amin(phi_i,axis=0))

            z_order = [1,-1,1,-1,1]

            wingkin = np.zeros((8,1000))
            beta = (10.0/180.0)*np.pi

            if z_order[i] == 1:
                for j in range(N_steps):

                    m_trend_i[0,j]     = M_i[j,2,0]
                    m_trend_i[1,j]     = M_i[j,2,1]
                    m_trend_i[2,j]     = M_i[j,2,2]
                    m_trend_i[3,j]     = M_i[j,2,3]
                    m_trend_i[4,j]     = M_i[j,2,4]
                    m_trend_i[5,j]     = M_i[j,2,5]
                    m_trend_i[6,j]     = M_i[j,2,6]
                    m_trend_i[7,j]     = M_i[j,2,7]
                    m_trend_i[8,j]     = M_i[j,2,8]
                    m_trend_i[9,j]     = M_i[j,2,9]
                    m_trend_i[10,j] = M_i[j,2,10]
                    m_trend_i[11,j] = M_i[j,2,11]
                    m_trend_i[12,j] = M_i[j,2,12]

                    alpha, U, L, D, w, w_dot = self.compute_FT(a_phi_i[:,j],a_theta_i[:,j],a_eta_i[:,j],a_xi_i[:,j],200.0)

                    ax_phi.plot(t,np.abs(alpha)*(180.0/np.pi),color=br_palette[j],linewidth=line_width)
                    ax_theta.plot(t,U,color=br_palette[j],linewidth=line_width)
                    ax_eta.plot(t,L/(self.g*self.m_fly),color=br_palette[j],linewidth=line_width)
                    ax_xi.plot(t,D/(self.g*self.m_fly),color=br_palette[j],linewidth=line_width)

                if i==0:
                    ax_m.plot(sigma_range,m_trend_i[0,:],color=self.c_muscle[0],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[1,:],color=self.c_muscle[1],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[2,:],color=self.c_muscle[2],linewidth=line_width)
                elif i==1:
                    ax_m.plot(sigma_range,m_trend_i[3,:],color=self.c_muscle[3],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[4,:],color=self.c_muscle[4],linewidth=line_width)
                elif i==2:
                    ax_m.plot(sigma_range,m_trend_i[5,:],color=self.c_muscle[5],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[6,:],color=self.c_muscle[6],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[7,:],color=self.c_muscle[7],linewidth=line_width)
                elif i==3:
                    ax_m.plot(sigma_range,m_trend_i[8,:],color=self.c_muscle[8],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[9,:],color=self.c_muscle[9],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[10,:],color=self.c_muscle[10],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[11,:],color=self.c_muscle[11],linewidth=line_width)
                elif i==4:
                    ax_m.plot(sigma_range,m_trend_i[12,:],color=self.c_muscle[12],linewidth=line_width)

            else:
                for j in range(N_steps-1,-1,-1):

                    m_trend_i[0,N_steps-1-j]     = M_i[j,2,0]
                    m_trend_i[1,N_steps-1-j]     = M_i[j,2,1]
                    m_trend_i[2,N_steps-1-j]     = M_i[j,2,2]
                    m_trend_i[3,N_steps-1-j]     = M_i[j,2,3]
                    m_trend_i[4,N_steps-1-j]     = M_i[j,2,4]
                    m_trend_i[5,N_steps-1-j]     = M_i[j,2,5]
                    m_trend_i[6,N_steps-1-j]     = M_i[j,2,6]
                    m_trend_i[7,N_steps-1-j]     = M_i[j,2,7]
                    m_trend_i[8,N_steps-1-j]     = M_i[j,2,8]
                    m_trend_i[9,N_steps-1-j]     = M_i[j,2,9]
                    m_trend_i[10,N_steps-1-j] = M_i[j,2,10]
                    m_trend_i[11,N_steps-1-j] = M_i[j,2,11]
                    m_trend_i[12,N_steps-1-j] = M_i[j,2,12]

                    alpha, U, L, D, w, w_dot = self.compute_FT(a_phi_i[:,j],a_theta_i[:,j],a_eta_i[:,j],a_xi_i[:,j],200.0)

                    ax_phi.plot(t,np.abs(alpha)*(180.0/np.pi),color=br_palette[N_steps-1-j],linewidth=line_width)
                    ax_theta.plot(t,U,color=br_palette[N_steps-1-j],linewidth=line_width)
                    ax_eta.plot(t,L/(self.g*self.m_fly),color=br_palette[N_steps-1-j],linewidth=line_width)
                    ax_xi.plot(t,D/(self.g*self.m_fly),color=br_palette[N_steps-1-j],linewidth=line_width)

                if i==0:
                    ax_m.plot(sigma_range,m_trend_i[0,:],color=self.c_muscle[0],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[1,:],color=self.c_muscle[1],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[2,:],color=self.c_muscle[2],linewidth=line_width)
                elif i==1:
                    ax_m.plot(sigma_range,m_trend_i[3,:],color=self.c_muscle[3],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[4,:],color=self.c_muscle[4],linewidth=line_width)
                elif i==2:
                    ax_m.plot(sigma_range,m_trend_i[5,:],color=self.c_muscle[5],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[6,:],color=self.c_muscle[6],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[7,:],color=self.c_muscle[7],linewidth=line_width)
                elif i==3:
                    ax_m.plot(sigma_range,m_trend_i[8,:],color=self.c_muscle[8],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[9,:],color=self.c_muscle[9],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[10,:],color=self.c_muscle[10],linewidth=line_width)
                    ax_m.plot(sigma_range,m_trend_i[11,:],color=self.c_muscle[11],linewidth=line_width)
                elif i==4:
                    ax_m.plot(sigma_range,m_trend_i[12,:],color=self.c_muscle[12],linewidth=line_width)

            ax_m.axvline(x=0,color=(0.5,0.5,0.5),linewidth=0.5)

            ax_phi.set_xlim([0,1])
            ax_theta.set_xlim([0,1])
            ax_eta.set_xlim([0,1])
            ax_xi.set_xlim([0,1])

            ax_m.set_xlim([-3,3])

            ax_phi.set_ylim([0,100])
            ax_theta.set_ylim([0,3000])
            ax_eta.set_ylim([-0.2,0.8])
            ax_xi.set_ylim([-0.1,0.4])

            ax_m.set_ylim([-0.1,1.1])

            if i==0:
                adjust_spines(ax_phi,['left'],yticks=[0,90],linewidth=0.8,spineColor='k')
                adjust_spines(ax_theta,['left'],yticks=[0,2000],linewidth=0.8,spineColor='k')
                adjust_spines(ax_eta,['left'],yticks=[0,0.5],linewidth=0.8,spineColor='k')
                adjust_spines(ax_xi,['left','bottom'],xticks=[0,1],yticks=[0,0.25],linewidth=0.8,spineColor='k')
                adjust_spines(ax_m,['left','bottom'],xticks=[-3,0,3],yticks=[0,1],linewidth=0.8,spineColor='k')
            else:
                adjust_spines(ax_phi,[],linewidth=0.8,spineColor='k')
                adjust_spines(ax_theta,[],linewidth=0.8,spineColor='k')
                adjust_spines(ax_eta,[],linewidth=0.8,spineColor='k')
                adjust_spines(ax_xi,['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(ax_m,['bottom'],xticks=[-3,0,3],linewidth=0.8,spineColor='k')

        os.chdir(plot_loc)
        file_name = 'latent_modes_LD.svg'
        fig.savefig(file_name, dpi=300)

    def make_lollipop_figure(self,wingkin,plot_name,save_loc,beta,m_clr):
        LP = Lollipop(self.working_dir)
        LP.Renderer()
        LP.ConstructModel(True)
        s_thorax = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0])
        s_head = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.55,0.0,0.42])
        s_abdomen = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1])
        body_scale = [0.80,0.85,0.90]
        body_clr = [(0.7,0.7,0.7)]
        LP.SetBodyColor(body_clr)
        LP.SetBodyScale(body_scale)
        LP.SetBodyState(s_thorax,s_head,s_abdomen)
        wing_length = 2.0
        joint_L = np.array([0.0,0.5,0.0])
        joint_R = np.array([0.0,-0.5,0.0])
        LE_pt = 0.1
        TE_pt = -0.2
        theta_L = wingkin[0,:]
        eta_L = wingkin[1,:]
        phi_L = wingkin[2,:]
        xi_L = wingkin[3,:]
        theta_R = wingkin[4,:]
        eta_R = wingkin[5,:]
        phi_R = wingkin[6,:]
        xi_R = wingkin[7,:]
        n_pts = xi_R.shape[0]
        FX_L = np.zeros(n_pts)
        FY_L = np.zeros(n_pts)
        FZ_L = np.zeros(n_pts)
        FX_R = np.zeros(n_pts)
        FY_R = np.zeros(n_pts)
        FZ_R = np.zeros(n_pts)
        FX_mean = 0.0
        FY_mean = 0.0
        FZ_mean = 0.0
        MX_mean = 0.0
        MY_mean = 0.0
        MZ_mean = 0.0
        FX_0 = 0.0
        FY_0 = 0.0
        FZ_0 = 0.0
        MX_0 = 0.0
        MY_0 = 0.0
        MZ_0 = 0.0

        LP.set_wing_motion_direct(theta_L,eta_L,phi_L,xi_L,theta_R,eta_R,phi_R,xi_R,n_pts)
        LP.set_forces_direct(FX_L,FY_L,FZ_L,FX_R,FY_R,FZ_R)
        LP.set_mean_forces(FX_mean,FY_mean,FZ_mean,MX_mean,MY_mean,MZ_mean)
        LP.set_FT_0(FX_0,FY_0,FZ_0,MX_0,MY_0,MZ_0)
        Fg = np.array([0.0,0.0,0.0])
        LP.set_Fg(Fg)
        FD = np.array([0.0,0.0,0.0])
        LP.set_FD(FD)
        LP.compute_tip_forces(wing_length,joint_L,joint_R,LE_pt,TE_pt,m_clr,m_clr,0,beta)
        img_width = 1000
        img_height = 800
        p_scale = 2.5
        clip_range = [0,16]
        cam_pos = [12,0,0]
        view_up = [0,0,1]
        img_name = plot_name+'_front.jpg'
        LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)
        cam_pos = [0,12,0]
        view_up = [0,0,1]
        img_name = plot_name+'_side.jpg'
        LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)
        cam_pos = [0,0,12]
        view_up = [1,0,0]
        img_name = plot_name+'_top.jpg'
        LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)
        clip_range = [0,16]
        cam_pos = [-12,0,0]
        view_up = [0,0,1]
        img_name = plot_name+'_back.jpg'
        LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)

    def create_plot_locations(self):
        # Create plot and plot/lollipop directorys in current working directory
        lollipop_dir = self.working_dir / 'plots' / 'lollipop'
        lollipop_dir.mkdir(parents=True, exist_ok=True)
        print(f'created: {str(lollipop_dir)}')
