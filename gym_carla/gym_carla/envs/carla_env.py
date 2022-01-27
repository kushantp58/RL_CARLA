#!/usr/bin/env python
from __future__ import division
import copy
from logging import disable
from os import strerror
import numpy as np
import random
import time
import math
from collections import deque
import pygame
import gym
from gym import spaces
from gym.utils import seeding
from pygame import display
import carla
import cv2
import roslibpy
import json
import glob
import sys,os
import csv


from .coordinates import train_coordinates
from .misc import _vec_decompose, delta_angle_between
from .carla_logger import *
from casadi import *



from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

# ==============================================================================
# -- MPC CONTROLLER ------------------------------------------------------------
# ==============================================================================

class mpc_control(object):
    def __init__(self):
        #self._contPub = pub
        self._yawOld = 0
        self._X_State_Limit = np.inf
        self._Y_State_Limit = np.inf
        self._stop = 0
        self._Vmpc = 0
        self._xDis = None
        self._yDis = None
        self.str = 0.0
        self.rvr = 0.0 
        self.thr = 0.0
        self.brk = 0.0
        self.Initialize_MPC()

    def Initialize_MPC(self):
        seconds = time.time()
        while(time.time() - seconds  <1.5):
            print("Waiting")
            time.sleep(0.2)
        self._T = 0.1  # Sampling time [s]
        self._T_interpolation = 0.02 # Interpolation time [s]
        self._N = 25   # Prediction horizon [samples]
        self._L = 2.45  # Wheel Base
        self._ActPathParam = 0 
        #   St : Front wheel steerng angle (rad)
        #   v  : Vehicle speed  
        self._Errold = 0
        self._IntErrVel = 0.0
        
        self._v_max = 12             # Maximum forward speed 16 m/s
        self._v_min = -self._v_max/4        # Maximum backward speed 4 m/s 
        self._St_max = 1.1344               # Maximum steering angle 65 degree
        self._St_min = -self._St_max        # Maximum steering angle 65 degree
        self._a_max = 5                     # Maximum acceleration 4 m/s**2
        self._a_min = self._a_max * -1.5    # Maximum decceleration 6 m/s**2
        self._dSt_max = pi/3                # Maximum change of steering angle 
        self._dSt_min = -self._dSt_max      # Maximum change of steering angle

        # States
        self._x = SX.sym('x')           # Vehicle position in x
        self._y = SX.sym('y')           # Vehicle position in y
        self._theta = SX.sym('theta')   # Vehicle orientation angle 
        self._Vel = SX.sym('v')         # Vehicle speed
        self._Steer = SX.sym('Steer')   # Vehicle steering angle (virtual central wheel)

        self._states = vertcat(self._x, self._y, self._theta, self._Vel, self._Steer)
        self._n_states = SX.size(self._states)
        self._StAct = 0
        self._SpAct = 0
        self._Vact = 0
        self._Vdes = 4

        # Sytem Model
        self._VelCMD = SX.sym('VelCMD') # System Inputs - Car Speed
        self._St = SX.sym('St')         # System Inputs - Steering angle
        self._controls = vertcat(self._VelCMD, self._St)    # System Control Inputs
        self._n_control = SX.size(self._controls)

        beta = atan2(tan(self._St),2)
        
        self._Vx = self._Vel  * cos(beta + self._theta)
        self._Vy = self._Vel  * sin(beta + self._theta)
        self._W = self._Vel  * cos(beta) *  tan(self._St) / self._L

        self._rhs = vertcat( self._Vx,
        self._Vy,
        self._W,
        self._VelCMD,
        self._St)

        self._f = Function('f',[self._states,self._controls],[self._rhs]) # Nonlinear mapping Function f(x,u)
        
        self._U = SX.sym('U', self._n_control[0], self._N)         # Decision variables (controls)
        self._P = SX.sym('P', self._n_states[0] + 4)                # Parameters (initial state of the car) + reference state
        self._X = SX.sym('X', self._n_states[0], (self._N+1))      # A matrix that represents the states over the optimization problem
        
        self._obj = 0 # Objective function
        self._g = SX([]) # Constraints vector

        # Weighing matrices (States)
        self._Q = DM([[15,0,0],[0,5,0],[0,0,10]])

        # Weighing matrices (Controls)
        self._R = DM([[2,0],[0,200]])
        # self._R = [[self._P[-3], self._P[-1]], [self._P[-1],self._P[-2]]]

        self._con_old = DM.zeros(self._n_control[0],1)

        self._st = self._X[:,0]   # initial state
        self._g = vertcat(self._g, self._st[:] - self._P[0:5])     # initial condition constraints

        position = SX.sym('Position',2)
        '''
            P[0]: Current X            P[5]: Ref X                             
            P[1]: Current Y            P[6]: Ref Y                                           
            P[2]: Current Yaw          P[7]: Ref Heading  
            P[3]: Current Velocity     P[8]: Ref Velocity            
            P[4]: Current Steering          
        '''
        # Compute Objective
        for k in range(self._N):
            self._st = self._X[:,k]
            self._con = self._U[:,k]

            yawRef = self._P[7]

            ErrX = self._P[5] - self._st[0]
            ErrY = self._P[6] - self._st[1]
            ErrLateral = -ErrX*sin(yawRef) + ErrY*cos(yawRef)

            objT1 = ErrLateral * ErrLateral * self._Q[0,0]
            objT2 = self._Q[1,1] * (1-sin(self._st[2])*sin(yawRef) - cos(self._st[2])*cos(yawRef))
            objT3 = (self._P[8] - self._st[3])**2 * self._Q[2,2]

            objT4 = mtimes(self._con[0:2].T-self._con_old[0:2].T,mtimes(self._R, self._con[0:2]-self._con_old[0:2])) 

            self._obj = self._obj + objT1 + objT2 + objT3 + objT4
            self._con_old = self._con

            self._st_next = self._X[:,k+1]
        
            k1 = self._f(self._st, self._con)
            k2 = self._f(self._st + k1*self._T/2, self._con)
            k3 = self._f(self._st + k2*self._T/2, self._con)
            k4 = self._f(self._st + k3*self._T, self._con)

            gradientRK4 = (k1 + 2*k2 + 2*k3 + k4) / 6
            st_next_RK4 = self._st[0:3] + self._T * gradientRK4[0:3]
            st_next_RK4 = vertcat(st_next_RK4, k1[3:5])

            self._g = vertcat(self._g, self._st_next - st_next_RK4) # compute constraints

        # Compute Constraints
        for k in range(self._N):
            dV = self._X[3,k+1] - self._X[3,k]
            self._g = vertcat(self._g, dV)
        
        for k in range(self._N):
            dU = self._U[1,k] - self._X[4,k]
            self._g = vertcat(self._g, dU)

        # print(self._g)
        # Make the decision variables on column vector 
        self._OPT_variables = reshape(self._X, self._n_states[0] * (self._N+1), 1)
        self._OPT_variables = vertcat(self._OPT_variables, reshape(self._U, self._n_control[0] * self._N, 1))
        
        self._nlp_prob = {'f':self._obj, 'x':self._OPT_variables, 'g':self._g, 'p':self._P}

        # Pick an NLP solver
        self._MySolver = 'ipopt'

        # Solver options
        self._opts = {'ipopt.max_iter':100, 'ipopt.print_level':0,'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

        self._solver = nlpsol('solver',self._MySolver,self._nlp_prob,self._opts)

        # Constraints initialization:
        self._lbx = DM.zeros(self._n_states[0] * (self._N+1) + self._n_control[0] * self._N,1)
        self._ubx = DM.zeros(self._n_states[0] * (self._N+1) + self._n_control[0] * self._N,1)

        self._lbg = DM.zeros(self._n_states[0] * (self._N+1) + 2 * self._N,1) 
        self._ubg = DM.zeros(self._n_states[0] * (self._N+1) + 2 * self._N,1)

        # inequality constraints (state constraints)

        # constraint on X position (0->a,0)
        a = self._n_states[0]*(self._N+1)
        self._lbx[0:a:self._n_states[0],0] = DM(-self._X_State_Limit)
        self._ubx[0:a:self._n_states[0],0] = DM(self._X_State_Limit)

        # constraint on Y position (0->a,1)
        self._lbx[1:a:self._n_states[0],0] = DM(-self._Y_State_Limit)
        self._ubx[1:a:self._n_states[0],0] = DM(self._Y_State_Limit)
        # constraint on yaw angle (0->a,2)
        self._lbx[2:a:self._n_states[0],0] = DM(-np.inf)
        self._ubx[2:a:self._n_states[0],0] = DM(np.inf)

        # constraint on velocity (state) (0->a,3)
        self._lbx[3:a:self._n_states[0],0] = DM(self._v_min)
        self._ubx[3:a:self._n_states[0],0] = DM(self._v_max)

        # constraint on steering angle (state) (0->a,4)
        self._lbx[4:a:self._n_states[0],0] = DM(self._St_min)
        self._ubx[4:a:self._n_states[0],0] = DM(self._St_max)

        # constraint on velocity input (a->end)
        self._lbx[a:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._v_min)
        self._ubx[a:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._v_max)

        # constraint on steering input (a->end)
        self._lbx[a + 1:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._St_min)
        self._ubx[a + 1:self._n_states[0] * (self._N+1) + self._n_control[0]*self._N:self._n_control[0],0] = DM(self._St_max)

        # equality constraints
        self._lbg[0:self._n_states[0]*(self._N+1),0] = DM(0)
        self._ubg[0:self._n_states[0]*(self._N+1),0] = DM(0)

        # constraint on vehicle acceleration
        self._lbg[self._n_states[0]*(self._N+1): self._n_states[0]*(self._N+1) + self._N,0] = DM(self._a_min * self._T)
        self._ubg[self._n_states[0]*(self._N+1): self._n_states[0]*(self._N+1) + self._N,0] = DM(self._a_max * self._T)

        # constraint on steering rate
        self._lbg[self._n_states[0]*(self._N+1) + self._N: self._n_states[0]*(self._N+1) + 2 * self._N,0] = DM(self._dSt_min * self._T)
        self._ubg[self._n_states[0]*(self._N+1) + self._N: self._n_states[0]*(self._N+1) + 2 * self._N,0] = DM(self._dSt_max * self._T)

        self._args = {}
        self._args['lbg'] = self._lbg       # dU and States constraints
        self._args['ubg'] = self._ubg       # dU and States constraints
        self._args['lbx'] = self._lbx       #  input constraints
        self._args['ubx'] = self._ubx       #  input constraints
        self._u0  = DM.zeros(self._N ,self._n_control[0])    # Control inputs
        self._x0  = DM.ones(self._N+1 ,self._n_states[0])   # Initialization of the states decision variables
        self._T_old = 0
        self._T_old_interpolation = 0
        self._CompTime = 0
        self._yaw = 0
        print("MPC Controller Initialized")

    def feedbackCallback(self,xEgo,yEgo,yawEgo,vEgo,stEgo,xRef,yRef,yawRef,vRef):
        self._yawOld = self._yaw
        self._VactOld = self._Vact 
        self._xDis = xEgo
        self._yDis = yEgo
        self._yaw = yawEgo
        self._Vact = vEgo
        self._StAct = stEgo
        self._Xs = [xRef,yRef,yawRef,vRef]
        self.runMPCstep()
        
    def runMPCstep(self):
        seconds = time.time()
        print("Errtime: ", self._T)
        print("Err: ", seconds - self._T_old)
        if abs(self._yawOld - self._yaw) > 6:
            for n in range(DM.size(self._x0)[0]):
                self._x0[n,2] = -1* self._x0[n,2]
        if((seconds - self._T_old) > self._T):
            self._T_old = seconds
            # self._Xs = [5,5,0]
            self._ActPathParam = self._x0[0,3].__float__()
            
            ErrX = self._Xs[0] - self._xDis
            ErrY = self._Xs[1] - self._yDis
            yawErr = 1 - (sin(self._yaw)*sin(self._Xs[2]) + cos(self._yaw)*cos(self._Xs[2]))

            Err = ErrX**2 + ErrY**2 
            LongErr = ErrX*cos(self._Xs[2]) + ErrY*sin(self._Xs[2])  
            LatErr = -ErrX*sin(self._Xs[2]) + ErrY*cos(self._Xs[2])  

            print("ErrX: " + str(ErrX))
            print("ErrY: " + str(ErrY))
            print("yaw: " + str(self._Xs[2]))
            print("ErrYaw: " + str(yawErr))
            print("LongErr: " + str(LongErr))
            print("LateralErr: " + str(LatErr))
            print("===========================")
            if(Err > 0.15):
                '''
                P[0]: Current X            P[6]: Ref X                             
                P[1]: Current Y            P[7]: Ref Y                                           
                P[2]: Current Yaw          P[8]: Ref Heading  
                P[4]: Current Velocity     P[9]: Ref Velocity             
                P[5]: Current Steering          
                '''
                R = [self._xDis, self._yDis, self._yaw, self._Vact, self._StAct] + self._Xs
                R = np.array(R)
                R = np.reshape(R, (R.size, 1))
                self._args['p'] = R
                self._args['x0'] = vertcat(reshape(self._x0.T, self._n_states[0]*(self._N+1), 1), reshape(self._u0.T, self._n_control[0] * self._N, 1))   # initial condition for optimization variable                   
                self.sol = self._solver(
                x0 = self._args['x0'],
                lbx = self._args['lbx'],
                ubx = self._args['ubx'],
                lbg = self._args['lbg'],
                ubg = self._args['ubg'],
                p = self._args['p'])

                self._u = reshape(self.sol['x'][self._n_states[0]*(self._N+1):].T,self._n_control[0],(self._N)).T
                self._Vmpc = self._u[0,0].__float__()
                self._Stdes = self._u[0,1].__float__()
                self.Vehicle_ContMsg()

                self._xStates = reshape(self.sol['x'][0:self._n_states[0]*(self._N+1)].T,self._n_states[0],(self._N+1)).T  
                self._x0[0:self._N,:] = self._xStates[1:self._N+1,:]
                self._x0[self._N,:] = self._xStates[self._N,:]
                self._u0[0:self._N-1,:] = self._u[1:self._N,:]
                self._u0[self._N-1,:] = self._u[self._N-1,:]

                self._CompTime = time.time() - seconds
                print("Comp Time: " + str(self._CompTime))
            
            else:
                print ('ARRIVED')
                self.Vehicle_ContMsg(True)

    def runInterpolationStep(self):
        seconds = time.time()
        if(self._T_old != 0):
            if((seconds - self._T_old_interpolation) > self._T_interpolation):
                self._T_old_interpolation = seconds
                w1 = (self._T_old_interpolation - self._T_old)/self._T
                w0 = 1 - w1
                # w0 = w0/self._T
                # w1 = w1/self._T
                # self._Vmpc = w0 * self._u[0,0].__float__() + w1 * self._u[1,0].__float__()
                # self._Stdes = w0 * self._u[0,1].__float__() + w1 * self._u[1,1].__float__()
                self.Vehicle_ContMsg()

            
    def Vehicle_ContMsg(self,Arrived = False):       
        if(not Arrived):
            ErrVel = self._Vmpc - self._Vact
            self._IntErrVel = self._IntErrVel + ErrVel * self._T

            # if(self._Errold * (ErrVel+2) < 0):
            #     self._IntErrVel = 0
            if (ErrVel > 0):
                AppliedPedal = 0.6 * ErrVel + 0.1 * self._IntErrVel #+ 0.1 * (ErrVel - self._Errold)/self._T
            else:
                AppliedPedal = 0.8 * ErrVel + 0.1 * self._IntErrVel
            self._Errold = ErrVel

            if(AppliedPedal>1):
                AppliedPedal = 1
            elif(AppliedPedal < 0):
                AppliedPedal = 0.1 * AppliedPedal
                if AppliedPedal < -1:
                    AppliedPedal = -1

            if(self._Vdes < 0.01 and self._Vact < 0.1):
                AppliedPedal = -0.5
            

            if(AppliedPedal > 0):
                self._throttle = AppliedPedal
                self._brake = 0
            else:
                self._throttle = 0
                self._brake = -AppliedPedal

            self._reverse = 0
            self._steer = self._Stdes
            self._hand_brake = 0
            self._reverse = 0
        else:
            self._steer = 0
            self._reverse = 0
            self._throttle = 0
            self._brake = 1
            self._hand_brake = 0
        
        if(self._stop == 1):
            self._throttle = 0
            self._brake = 1
        
        if(self._stop == 5):
            self._throttle = 0
            self._brake = 0.5 * self._brake + 0.5 
        self.str = self._steer
        self.rvr = self._reverse
        self.thr = self._throttle
        self.brk = self._brake
  

# ==============================================================================
# -- Carla Env -------------------------------------------------------------
# ==============================================================================

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        self.logger = setup_carla_logger(
            "output_logger", experiment_name=str(params['port']))
        self.logger.info("Env running in port {}".format(params['port']))
        # parameters
        self.dt = params['dt']
        self.port = params['port']
        self.task_mode = params['task_mode']
        self.code_mode = params['code_mode']
        self.max_time_episode = params['max_time_episode']
        self.obs_size = params['obs_size']
        self.state_size = (self.obs_size[0], self.obs_size[1] - 36)

        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.host = params['host']
        self.mpc_control = mpc_control() ## mpc control class object

        ##### Reference values to be passed to the roslibpy publisher
        self._xRefArr = 0.0
        self._yRefArr = 0.0
        self._yawRefArr = 0.0
        self._vRefArr = 0.0 #self.desired_speed



        # action and observation space
        self.action_space = spaces.Box(
            np.array([-2.0, -2.0]), np.array([2.0, 2.0]), dtype=np.float32)
        self.state_space = spaces.Box(
            low=-50.0, high=50.0, shape=(12, ), dtype=np.float32)
        

        # Connect to carla server and get world object
        # print('connecting to Carla server...')
        self._make_carla_client(self.host, self.port)

        # Load routes
        self.starts, self.dests = train_coordinates(self.task_mode)
        self.route_deterministic_id = 0

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            'sensor.other.collision')

        self.CAM_RES = 1024
        # Add camera sensor
        self.camera_img = np.zeros((self.CAM_RES, self.CAM_RES, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.CAM_RES))
        self.camera_bp.set_attribute('image_size_y', str(self.CAM_RES))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # A dict used for storing state data
        self.state_info = {}

        # a dict used for storing flags
        self.flag_info = {}

        # A list stores the ids for each episode
        self.actors = []

        self.xEgo = 0.0
        self.yEgo = 0.0
        self.yawEgo = 0.0
        self.vEgo = 0.0

        # Future distances to get heading
        self.distances = [1., 5., 10.]

    def reset(self):

        while True:
            try:
                self.collision_sensor = None
                self.lane_sensor = None

                # Delete sensors, vehicles and walkers
                while self.actors:
                    (self.actors.pop()).destroy()

                # Disable sync mode
                self._set_synchronous_mode(False)

                # Spawn the ego vehicle at a random position between start and dest
                # Start and Destination
                if self.task_mode == 'Straight':
                    self.route_id = 0
                elif self.task_mode == 'Curve':
                    self.route_id = 1  #np.random.randint(2, 4)
                elif self.task_mode == 'Long' or self.task_mode == 'Lane' or self.task_mode == 'Lane_test' :
                    if self.code_mode == 'train':
                        self.route_id = np.random.randint(0, 4)
                    elif self.code_mode == 'test':
                        self.route_id = self.route_deterministic_id
                        self.route_deterministic_id = (
                            self.route_deterministic_id + 1) % 4
                elif self.task_mode == 'U_curve':
                    self.route_id = 0
                self.start = self.starts[self.route_id]
                self.dest = self.dests[self.route_id]

                # The tuple (x,y) for the current waypoint
                self.current_wpt = np.array((self.start[0], self.start[1],
                                             self.start[5]))

                # Appending the values of x,y,yaw of starts to the RefArr
                self._xRefArr = self.current_wpt[0] 
                self._yRefArr = self.current_wpt[1]
                self._yawRefArr = self.current_wpt[2]
                print("Current waypoint:",self.current_wpt)
               
                # global xStart
                # global yStart
                # global zStart
                # global yawStart
                # xStart = self._xRefArr
                # yStart = self._yRefArr
                # zStart = 0.2
                # yawStart = self._yawRefArr



                ego_spawn_times = 0
                while True:
                    if ego_spawn_times > self.max_ego_spawn_times:
                        self.reset()
                    transform = self._set_carla_transform(self.start)
                    # Code_mode == train, spwan randomly between start and destination
                    if self.code_mode == 'train':
                        transform = self._get_random_position_between(
                            start=self.start,
                            dest=self.dest,
                            transform=transform)
                    if self._try_spawn_ego_vehicle_at(transform):
                        break
                    else:
                        ego_spawn_times += 1
                        time.sleep(0.1)
                    self.tf = transform

                # Add collision sensor
                self.collision_sensor = self.world.try_spawn_actor(
                    self.collision_bp, carla.Transform(), attach_to=self.ego)
                self.actors.append(self.collision_sensor)
                self.collision_sensor.listen(
                    lambda event: get_collision_hist(event))
                
                def get_camera_img(data):
                    self.og_camera_img = data
                self.collision_hist = []
                self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
                self.actors.append(self.camera_sensor)
                self.camera_sensor.listen(lambda data: get_camera_img(data))


                def get_collision_hist(event):
                    impulse = event.normal_impulse
                    intensity = np.sqrt(impulse.x**2 + impulse.y**2 +
                                        impulse.z**2)
                    self.collision_hist.append(intensity)
                    if len(self.collision_hist) > self.collision_hist_l:
                        self.collision_hist.pop(0)

                self.collision_hist = []

                # Update timesteps
                self.time_step = 1
                self.reset_step += 1

                # Enable sync mode
                self.settings.synchronous_mode = True
                self.world.apply_settings(self.settings)

                # Set the initial speed to desired speed
                yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
                init_speed = carla.Vector3D(
                    x=self.desired_speed * np.cos(yaw),
                    y=self.desired_speed * np.sin(yaw))
                self.ego.set_velocity(init_speed)
                self.world.tick()
                self.world.tick()

                # Get waypoint infomation
                ego_x, ego_y = self._get_ego_pos()
                self.current_wpt = self._get_waypoint_xyz()

                delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
                road_heading = np.array([
                    np.cos(wpt_yaw / 180 * np.pi),
                    np.sin(wpt_yaw / 180 * np.pi)
                ])
                ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
                ego_heading_vec = np.array(
                    [np.cos(ego_heading),
                     np.sin(ego_heading)])

                future_angles = self._get_future_wpt_angle(
                    distances=self.distances)

                # Update State Info (Necessary?)
                velocity = self.ego.get_velocity()
                accel = self.ego.get_acceleration()
                dyaw_dt = self.ego.get_angular_velocity().z
                v_t_absolute = np.array([velocity.x, velocity.y])
                a_t_absolute = np.array([accel.x, accel.y])
                c = self.ego.get_control()
                self.xEgo = ego_x
                self.yEgo = ego_y
                self.yawEgo = ego_yaw
                #self.stEgo = delta_yaw
                self.stEgo = c.steer * -math.pi/3  
                self.vEgo = math.sqrt(velocity.x**2 + velocity.y**2)



                # decompose v and a to tangential and normal in ego coordinates
                v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
                a_t = _vec_decompose(a_t_absolute, ego_heading_vec)

                #self.mpc_control.feedbackCallback(self.xEgo,self.yEgo,self.yawEgo,self.vEgo,self.stEgo,self._xRefArr,self._yRefArr,self._yawRefArr,self._vRefArr)

                # Reset action of last time step
                # TODO:[another kind of action]
                self.last_action = np.array([0.0, 0.0])

                #pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]
                pos_err_vec = np.array((ego_x,ego_y)) - self.current_wpt[0:2]
                print("pos_err_vec:",pos_err_vec)

                self.state_info['velocity_t'] = v_t
                self.state_info['acceleration_t'] = a_t

#                self.state_info['ego_heading'] = ego_heading
                self.state_info['delta_yaw_t'] = delta_yaw
                self.state_info['dyaw_dt_t'] = dyaw_dt


                self.state_info['lateral_dist_t'] =  np.linalg.norm(pos_err_vec) * \
                                                    np.sign(pos_err_vec[0] * road_heading[1] - \
                                                            pos_err_vec[1] * road_heading[0])
                self.state_info['action_t_1'] = self.last_action
                self.state_info['angles_t'] = future_angles

                # End State variable initialized
                self.isCollided = False
                self.isTimeOut = False
                self.isSuccess = False
                self.isOutOfLane = False
                self.isSpecialSpeed = False
                self.flag_info['collision'] = False
                self.flag_info['timeout'] = False
                self.flag_info['success'] = False
                self.flag_info['out_of_lane'] = False
                self.flag_info['special_speed'] = False

                return self._get_obs(), copy.deepcopy(self.state_info)

            except:
                self.logger.error("Env reset() error")
                time.sleep(2)
                self._make_carla_client(self.host, self.port)

    def to_display_surface(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, display, mode='human'):
        camera_surface = self.to_display_surface(self.og_camera_img)
        display.blit(camera_surface, (0, 0))
      

    def step(self, action):

        try:
            # Assign acc/steer/brake to action signal
            # Ver. 1 input is the value of control signal
            # throttle_or_brake, steer = action[0], action[1]
            # if throttle_or_brake >= 0:
            #     throttle = throttle_or_brake
            #     brake = 0
            # else:
            #     throttle = 0
            #     brake = -throttle_or_brake

            # Ver. 2 input is the delta value of control signal
            # TODO:[another kind of action] change the action space to [-2, 2]
            current_action = np.array(action) + self.last_action
            current_action = np.clip(
                current_action, -1.0, 1.0, dtype=np.float32)

            #self._vRefArr,self._yawRefArr = current_action
            #value_ab = ((value_xy - x) / (y - x)) * (b - a) + a
            #self._yawRefArr = ((self._yawRefArr + 1) / 2) * (720) - 360
            #self._vRefArr = ((self._vRefArr + 1) / 2) * (12)
            throttle_or_brake, steer = current_action
            #self.mpc_control.runInterpolationStep()
            #throttle_or_brake = self.mpc_control.thr
            #throttle_or_brake = self.rc._control.brake
            #steer = self.mpc_control.str


            if throttle_or_brake >= 0:
                throttle = throttle_or_brake
                brake = 0
            else:
                throttle = 0
                brake = -throttle_or_brake

            # Apply control
            act = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake))
            self.ego.apply_control(act)


            for _ in range(4):
                self.world.tick()

            # Get waypoint infomation
            ego_x, ego_y = self._get_ego_pos()
            self.current_wpt = self._get_waypoint_xyz()

            delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
            road_heading = np.array(
                [np.cos(wpt_yaw / 180 * np.pi),
                 np.sin(wpt_yaw / 180 * np.pi)])
            ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
            ego_heading_vec = np.array((np.cos(ego_heading),
                                        np.sin(ego_heading)))

            future_angles = self._get_future_wpt_angle(
                distances=self.distances)

            # Get dynamics info
            velocity = self.ego.get_velocity()
            accel = self.ego.get_acceleration()
            dyaw_dt = self.ego.get_angular_velocity().z
            v_t_absolute = np.array([velocity.x, velocity.y])
            a_t_absolute = np.array([accel.x, accel.y])
            c = self.ego.get_control()
            #zEgo = t.location.z
            # self.xEgo = t.location.x
            # self.yEgo = t.location.y
            # self.yawEgo = -t.rotation.yaw * math.pi/180
            #self._yawRefArr = self._yawRefArr * math.pi/180
            self.xEgo = ego_x
            self.yEgo = ego_y
            self.yawEgo = ego_yaw
            self.vEgo = math.sqrt(velocity.x**2 + velocity.y**2)
            self.stEgo = c.steer * -math.pi/3  
        
            print("Message sent")

            # decompose v and a to tangential and normal in ego coordinates
            v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
            a_t = _vec_decompose(a_t_absolute, ego_heading_vec)
            
            #self.mpc_control.feedbackCallback(self.xEgo,self.yEgo,self.yawEgo,self.vEgo,self.stEgo,self._xRefArr,self._yRefArr,self._yawRefArr,self._vRefArr)

            #pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]
            pos_err_vec = np.array((ego_x,ego_y)) - self.current_wpt[0:2]
            print("pos_err_vec:",pos_err_vec)

            self.state_info['velocity_t'] = v_t
            self.state_info['acceleration_t'] = a_t

#            self.state_info['ego_heading'] = ego_heading
            self.state_info['delta_yaw_t'] = delta_yaw
            self.state_info['dyaw_dt_t'] = dyaw_dt
 

            self.state_info['lateral_dist_t'] = np.linalg.norm(pos_err_vec) * \
                                                np.sign(pos_err_vec[0] * road_heading[1] - \
                                                        pos_err_vec[1] * road_heading[0])
            self.state_info['action_t_1'] = self.last_action
            self.state_info['angles_t'] = future_angles

            # Update timesteps
            self.time_step += 1
            self.total_step += 1
            self.last_action = current_action

            # calculate reward
            isDone = self._terminal()
            current_reward = self._get_reward(np.array(current_action))

            


            return (self._get_obs(), current_reward, isDone,
                    copy.deepcopy(self.state_info),copy.deepcopy(self.flag_info))

        except:
            self.logger.error("Env step() error")
            time.sleep(2)
            return (self._get_obs(), 0.0, True, copy.deepcopy(self.state_info),copy(self.flag_info))

    # def render(self, mode='human'):
    #     pass

    def close(self):
        while self.actors:
            (self.actors.pop()).destroy()

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = self._get_ego_pos()

        # If at destination
        dest = self.dest
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 2.0:
            print("Get destination! Episode Done.")
            self.logger.debug('Get destination! Episode cost %d steps in route %d.' % (self.time_step, self.route_id))
            self.isSuccess = True
            self.flag_info['success'] = True
            return True

        # If collides
        if len(self.collision_hist) > 0:
            print("Collision happened! Episode Done.")
            self.logger.debug(
                'Collision happened! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isCollided = True
            self.flag_info['collision'] = True
            return True

        # If reach maximum timestep
        if self.time_step >= self.max_time_episode:
            print("Time out! Episode Done.")
            self.logger.debug('Time out! Episode cost %d steps in route %d.' %
                              (self.time_step, self.route_id))
            self.isTimeOut = True
            self.flag_info['timeout'] = True
            return True

        # If out of lane
        # if len(self.lane_invasion_hist) > 0:
        if abs(self.state_info['lateral_dist_t']) > 1.2:
            print("lane invasion happened! Episode Done.")
            if self.state_info['lateral_dist_t'] > 0:
                self.logger.debug(
                    'Left Lane invasion! Episode cost %d steps in route %d.' %
                    (self.time_step, self.route_id))
            else:
                self.logger.debug(
                    'Right Lane invasion! Episode cost %d steps in route %d.' %
                    (self.time_step, self.route_id))
            self.isOutOfLane = True
            self.flag_info['out_of_lane'] = True
            return True

        # If speed is special
        velocity = self.ego.get_velocity()
        v_norm = np.linalg.norm(np.array((velocity.x, velocity.y)))
        if v_norm < 4:
            self.logger.debug(
                'Speed too slow! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isSpecialSpeed = True
            self.flag_info['special_speed'] = True
            return True
        elif v_norm > (1.5 * self.desired_speed):
            self.logger.debug(
                'Speed too fast! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isSpecialSpeed = True
            self.flag_info['special_speed'] = True
            return True

        print("Flags from environment:",self.isCollided,self.isTimeOut,self.isOutOfLane,self.isSpecialSpeed,self.isSuccess)
        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker' or actor.type_id == 'sensor.camera.rgb' or actor.type_id == 'sensor.other.collision':
                        actor.stop()
                    actor.destroy()

    def _create_vehicle_bluepprint(self,
                                   actor_filter,
                                   color=None,
                                   number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.
        Args:
            actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
        Returns:
            bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints
                if int(x.get_attribute('number_of_wheels')) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_ego_pos(self):
        """Get the ego vehicle pose (x, y)."""
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        return ego_x, ego_y

    def _set_carla_transform(self, pose):
        """Get a carla tranform object given pose.
        Args:
            pose: [x, y, z, pitch, roll, yaw].
        Returns:
            transform: the carla transform object
        """
        transform = carla.Transform()
        transform.location.x = pose[0]
        transform.location.y = pose[1]
        transform.location.z = pose[2]
        transform.rotation.pitch = pose[3]
        transform.rotation.roll = pose[4]
        transform.rotation.yaw = pose[5]
        return transform

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
            transform: the carla transform object.
        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = self.world.spawn_actor(self.ego_bp, transform)
        if vehicle is not None:
            self.actors.append(vehicle)
            self.ego = vehicle
            return True
        return False

    def _get_obs(self):
        # [img version]
        # current_obs = self.camera_img[36:, :, :].copy()
        # return np.float32(current_obs / 255.0)

        # [vec version]
        return np.float32(self._info2normalized_state_vector())

    def _get_reward(self, action):
        """
        calculate the reward of current state
        params:
            action: np.array of shape(2,)
        """
        # end state
        # reward for done: collision/out/SpecislSPeed & Success
        r_step = 10.0
        if self.isCollided or self.isOutOfLane or self.isSpecialSpeed:
            r_done = -500.0
            return r_done
        if self.isSuccess:
            r_done = 300.0
            return r_done

        # reward for speed
        v = self.ego.get_velocity()
        ego_velocity = np.array([v.x, v.y])
        speed_norm = np.linalg.norm(ego_velocity)
        delta_speed = speed_norm - self.desired_speed
        r_speed = -delta_speed**2 / 5.0
        # print("r_speed:", speed_norm)

        # reward for steer
        delta_yaw, _, _ = self._get_delta_yaw()
        r_steer = -100 * (delta_yaw * np.pi / 180)**2
        # print("r_steer:", delta_yaw, '------>', r_steer)

        # reward for action smoothness
        r_action_regularized = -5 * np.linalg.norm(action)**2
        # print("r_action:", action, '------>', r_action_regularized)

        # reward for lateral distance to the center of road
        lateral_dist = self.state_info['lateral_dist_t']
        r_lateral = -10.0 * lateral_dist**2
        # print("r_lateral:", lateral_dist, '-------->', r_lateral)

        return r_speed + r_steer + r_action_regularized + r_lateral + r_step

    def _make_carla_client(self, host, port):
        while True:
            try:
                self.logger.info("connecting to Carla server...")
                self.client = carla.Client(self.host,port)
                self.client.set_timeout(10.0)

                # Set map
                if self.task_mode == 'Straight':
                    self.world = self.client.load_world('Town01')
                elif self.task_mode == 'Curve':
                    # self.world = self.client.load_world('Town01')
                    self.world = self.client.load_world('Town05')
                elif self.task_mode == 'Long':
                    self.world = self.client.load_world('Town01')
                    # self.world = self.client.load_world('Town02')
                elif self.task_mode == 'Lane':
                    # self.world = self.client.load_world('Town01')
                    self.world = self.client.load_world('Town05')
                elif self.task_mode == 'U_curve':
                    self.world = self.client.load_world('Town03')
                elif self.task_mode == 'Lane_test':
                    self.world = self.client.load_world('Town03')
                self.map = self.world.get_map()

                # Set weather
                self.world.set_weather(carla.WeatherParameters.ClearNoon)
                self.logger.info(
                    "Carla server port {} connected!".format(port))
                break
            except Exception:
                self.logger.error(
                    'Fail to connect to carla-server...sleeping for 2')
                time.sleep(2)

    def _get_random_position_between(self, start, dest, transform):
        """
        get a random carla position on the line between start and dest
        """
        if self.task_mode == 'Straight':
            # s_x, s_y, s_z = start[0], start[1], start[2]
            # d_x, d_y, d_z = dest[0], dest[1], dest[2]

            # ratio = np.random.rand()
            # new_x = (d_x - s_x) * ratio + s_x
            # new_y = (d_y - s_y) * ratio + s_y
            # new_z = (d_z - s_z) * ratio + s_z

            # transform.location = carla.Location(x=new_x, y=new_y, z=new_z)
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 30)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        elif self.task_mode == 'Curve':
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 45)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        elif self.task_mode == 'Long' or self.task_mode == 'Lane':
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 60)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        return transform

    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt = self.map.get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            wpt_yaw = self.current_wpt[2] % 360
        else:
            wpt_yaw = current_wpt.transform.rotation.yaw % 360
        ego_yaw = self.ego.get_transform().rotation.yaw % 360

        delta_yaw = ego_yaw - wpt_yaw
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        return delta_yaw, wpt_yaw, ego_yaw

    def _get_waypoint_xyz(self):
        """
        Get the (x,y) waypoint of current ego position
            if t != 0 and None, return the wpt of last moment
            if t == 0 and None wpt: return self.starts
        """
        waypoint = self.map.get_waypoint(location=self.ego.get_location())
        if waypoint:
            return np.array(
                (waypoint.transform.location.x, waypoint.transform.location.y,
                 waypoint.transform.rotation.yaw))
        else:
            return self.current_wpt

    def _get_future_wpt_angle(self, distances):
        """
        Get next wpts in distances
        params:
            distances: list of int/float, the dist of wpt which user wants to get
        return:
            future_angles: np.array, <current_wpt, wpt(dist_i)> correspond to the dist in distances
        """
        angles = []
        current_wpt = self.map.get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            current_road_heading = self.current_wpt[3]
        else:
            current_road_heading = current_wpt.transform.rotation.yaw

        for d in distances:
            wpt_heading = current_wpt.next(d)[0].transform.rotation.yaw
            delta_heading = delta_angle_between(current_road_heading,
                                                wpt_heading)
            angles.append(delta_heading)

        return np.array(angles, dtype=np.float32)

    def _info2normalized_state_vector(self):
        '''
        params: dict of ego state(velocity_t, accelearation_t, dist, command, delta_yaw_t, dyaw_dt_t)
        type: np.array
        return: array of size[9,], torch.Tensor (v_x, v_y, a_x, a_y
                                                 delta_yaw, dyaw, d_lateral, action_last,
                                                 future_angles)
        '''
        velocity_t = self.state_info['velocity_t']
        accel_t = self.state_info['acceleration_t']

        delta_yaw_t = np.array(self.state_info['delta_yaw_t']).reshape(
            (1, )) / 2.0
        dyaw_dt_t = np.array(self.state_info['dyaw_dt_t']).reshape((1, )) / 5.0

        lateral_dist_t = self.state_info['lateral_dist_t'].reshape(
            (1, )) * 10.0
        action_last = self.state_info['action_t_1'] * 10.0

        future_angles = self.state_info['angles_t'] / 2.0


        info_vec = np.concatenate([
            velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
            action_last, future_angles
        ],
                                  axis=0)
        info_vec = info_vec.squeeze()

        return info_vec