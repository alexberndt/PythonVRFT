# test_reference.py - Unittest for virtual reference algorithm
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 10th January 2021, by alessior@kth.se
#
# Copyright (c) [2017-2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of PythonVRFT.
# PythonVRFT is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with PythonVRFT.
# If not, see <https://opensource.org/licenses/MIT>.
#

from unittest import TestCase
import numpy as np
import scipy.signal as scipysig
from vrft.iddata import *
from vrft.utils import *
from vrft.vrft_algo import *


class TestReference(TestCase):
    def test_virtual_reference_basic(self):
        # wrong system
        with self.assertRaises(ValueError):
            virtual_reference(1, 1, 0)

        # system cannot be a constant 
        with self.assertRaises(ValueError):
            virtual_reference([1],[1], 0)

        # system cannot be a constant of (2/3)
        with self.assertRaises(ValueError):
            virtual_reference(np.array(2), np.array(3), 0)

        # wrong data
        with self.assertRaises(ValueError):
            virtual_reference([1], [1, 1], 0)


    def test_virtual_reference_1_order_system(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.ones(len(t)).tolist()

        num = [0.1]
        den = [1, -0.9]
        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t,y = scipysig.dlsim(sys, u, t)
        y = y[:,0]
        data = iddata(y,u,t_step,[0,0])
        
        # wrong initial conditions
        with self.assertRaises(ValueError):
            r, _ = virtual_reference(data, num, den)

        num = [0.1]
        den = [1, -0.9]
        data = iddata(y,u,t_step,[0])
        r, _ = virtual_reference(data, num, den)
        for i in range(len(r)):
            self.assertTrue(np.isclose(r[i], u[i]))


    def test_virtual_reference_2_order_system_1(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.ones(len(t)).tolist()

        num = [1-1.6+0.63]
        den = [1, -1.6, 0.63]
        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t, y = scipysig.dlsim(sys, u, t)
        y = y[:,0]
        data = iddata(y,u,t_step,[0,0])
        r, _ = virtual_reference(data, num, den)
        for i in range(len(r)):
            self.assertTrue(np.isclose(r[i], u[i]))

     
    def test_virtual_reference_2_order_system_2(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.random.normal(size=len(t))

        omega = 10
        alpha = np.exp(-t_step*omega)
        num = [(1-alpha)**2] 
        den = [1, -2*alpha, alpha**2, 0]

        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t, y = scipysig.dlsim(sys, u, t)
        data = iddata(y,u,t_step,[0,0,0])
        r, _ = virtual_reference(data, num, den)
        for i in range(len(r)):
            self.assertTrue(np.isclose(r[i], u[i]))


    def test_virtual_reference_4_order_system_1(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.random.normal(size=len(t))

        num = [ 0.50666]
        den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
        sys = scipysig.TransferFunction(num, den, dt=t_step)

        t, y = scipysig.dlsim(sys, u, t)
        data = iddata(y,u,t_step,[0,0,0,0])
        r, _ = virtual_reference(data, num, den)
        for i in range(len(r)):
            # print(" r[{}] = {} \t u[{}] = {}".format(i,r[i],i,u[i]))
            self.assertTrue(np.isclose(r[i], u[i]))

    def test_virtual_reference_4_order_system_2(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.random.normal(size=len(t))

        num = [0.28261, 0.50666]
        den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
        sys = scipysig.TransferFunction(num, den, dt=t_step)

        # z,p,k = scipysig.tf2zpk(num, den)
        # print("zeroes: ")
        # for zero in z:
        #     print("|{}| = {}".format(zero, np.abs(zero)))
        # print("poles: ")
        # for pole in p:
        #     print("|{}| = {}".format(pole, np.abs(pole)))

        t, y = scipysig.dlsim(sys, u, t)
        data = iddata(y,u,t_step,[0,0,0])
        r, _ = virtual_reference(data, num, den)
        for i in range(len(r)):
            # print(" r[{}] = {} \t u[{}] = {}".format(i,r[i],i,u[i]))
            self.assertTrue(np.isclose(r[i], u[i]))
        


 