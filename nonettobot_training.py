#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import random
import math
from geometry_msgs.msg import Twist, Pose, Accel
from gazebo_msgs.srv import GetModelState, SetModelState, SpawnModel, SpawnModelRequest, DeleteModel, DeleteModelRequest
from sensor_msgs.msg import LaserScan, JointState
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Empty as EmptyMsg

import take_photo
from PIL import Image
import argparse
import imutils
import cv2
import detect_object

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import create_obj
import robot_obj_ops
import predict_motion
import train_robot

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)

get_model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)

obj_counter = 0
obj_names = []
obj_masses = []
obj_shapes = []
obj_volumes = []
obj_init_poses = []
obj_final_poses = []
robot_final_poses = []
robot_init_poses = []
robot_speed = []
duration = []
robots_speed = 0
name = ""
obj_shape = 0
obj_mass = 0
obj_radius = 0
edgeX = 0

# train_causes = np.array([])
# train_result = np.array([])
# test_causes = []
# model = tf.keras.Sequential()
# graph = tf.Graph()

def create_world():
    global obj_names, obj_masses, obj_volumes, obj_shapes, obj_final_poses, robot_final_poses, robot_speed, pub_move, duration, obj_init_poses, robot_init_poses, obj_counter, nns
    global robots_speed, name, obj_shape, obj_mass, obj_radius

    robot_x = random.uniform(-5,0)
    robot_y = random.uniform(-5,0)

    obj_x = random.uniform(0,5)
    obj_y = random.uniform(0,5)

    robot_obj_ops.set_robot_pos(robot_x, robot_y, 0)

    robot_init_poses.append(robot_obj_ops.get_robot_pos())
    robot_init = robot_obj_ops.get_robot_pos()

    obj_mass = random.uniform(0, 10)
    obj_shape = random.randint(0,2)

    obj_masses.append(obj_mass)
    obj_shapes.append(obj_shape)

    name = "obj_" + str(obj_counter)
    obj_names.append(name)

    robots_speed = random.uniform(1.5, 2)

    if obj_shape == 0:
        obj_radius = random.uniform(0.1, 0.7)
        obj_volumes.append(robot_obj_ops.get_volume(0, obj_radius, 0, 0, 0, 0))
        create_obj.create_sdf_file_sphere(obj_mass, obj_radius)
        create_obj.create_obj(obj_x, obj_y, obj_radius, 0, obj_counter)
        obj_init = robot_obj_ops.get_obj_pos(name)
        obj_init_poses.append(obj_init)
        # wanted_x = random.uniform(0, 0.5)
        # wanted_y = random.uniform(0, 0.1)
        # robot_prediction = train_robot.main([0, robot_init.position.x, \
        #                                 robot_init.position.y, \
        #                                 robot_obj_ops.get_volume(0, obj_radius, 0, 0, 0, 0), \
        #                                 obj_mass, obj_init.position.x, \
        #                                 obj_init.position.y, \
        #                                 wanted_x, \
        #                                 wanted_y])
        # predictions = predict_motion.main([0, robot_init.position.x,\
        #                                 robot_init.position.y, \
        #                                 robot_prediction, \
        #                                 robot_obj_ops.get_volume(0, obj_radius, 0, 0, 0, 0), \
        #                                 obj_mass, obj_init.position.x, \
        #                                 obj_init.position.y])

    elif obj_shape == 1:
        global edgeX
        edgeX = random.uniform(0.1, 0.7)
        edgeY = random.uniform(0.1, 0.7)
        edgeZ = random.uniform(0.1, 0.7)
        obj_volumes.append(robot_obj_ops.get_volume(1, 0, 0, edgeX, edgeY, edgeZ))
        create_obj.create_sdf_file_cube(obj_mass, edgeX, edgeY, edgeZ)
        create_obj.create_obj(obj_x, obj_y, edgeZ / 2, 1, obj_counter)
        obj_init = robot_obj_ops.get_obj_pos(name)
        obj_init_poses.append(obj_init)
        # wanted_x = random.uniform(0, 0.5)
        # wanted_y = random.uniform(0, 0.1)
        # robot_prediction = train_robot.main([1, robot_init.position.x, \
        #                     robot_init.position.y, \
        #                     robot_obj_ops.get_volume(1, 0, 0, edgeX, edgeY, edgeZ), \
        #                     obj_mass, \
        #                     obj_init.position.x, \
        #                     obj_init.position.y, \
        #                     wanted_x, \
        #                     wanted_y])
        # predictions = predict_motion.main([1, robot_init.position.x, \
        #                     robot_init.position.y, \
        #                     robot_prediction, \
        #                     robot_obj_ops.get_volume(1, 0, 0, edgeX, edgeY, edgeZ), \
        #                     obj_mass, \
        #                     obj_init.position.x, \
        #                     obj_init.position.y])

    elif obj_shape == 2:
        obj_radius = random.uniform(0.1, 0.7)
        obj_length = random.uniform(0.1, 0.7)
        obj_volumes.append(robot_obj_ops.get_volume(2, obj_radius, obj_length, 0, 0, 0))
        create_obj.create_sdf_file_cylinder(obj_mass, obj_radius, obj_length)
        create_obj.create_obj(obj_x, obj_y, obj_length / 2, 2, obj_counter)
        obj_init = robot_obj_ops.get_obj_pos(name)
        obj_init_poses.append(obj_init)
        # wanted_x = random.uniform(0, 0.5)
        # wanted_y = random.uniform(0, 0.1)
        # robot_prediction = train_robot.main([2, robot_init.position.x, \
        #                     robot_init.position.y, \
        #                     robot_obj_ops.get_volume(2, obj_radius, obj_length, 0, 0, 0), \
        #                     obj_mass, \
        #                     obj_init.position.x, \
        #                     obj_init.position.y, \
        #                     wanted_x, \
        #                     wanted_y])
        # predictions = predict_motion.main([2, robot_init.position.x, \
        #                     robot_init.position.y, \
        #                     robot_prediction, \
        #                     robot_obj_ops.get_volume(2, obj_radius, obj_length, 0, 0, 0), \
        #                     obj_mass, \
        #                     obj_init.position.x, \
        #                     obj_init.position.y])

def get_result():
    global obj_names, obj_masses, obj_volumes, obj_shapes, obj_final_poses, robot_final_poses, robot_speed, pub_move, duration, obj_init_poses, robot_init_poses, obj_counter, nns
    global robots_speed

    if obj_shape == 0 or obj_shape == 2:
        robot_obj_ops.wait_for_robot(obj_radius)
    elif obj_shape == 1:
        robot_obj_ops.wait_for_robot(edgeX / 2)

    initial_time = rospy.Time.now()
    robot_obj_ops.wait_till_obj_stops(name)
    final_time = rospy.Time.now()

    # rospy.sleep(final_time - initial_time + rospy.Duration(1))

    duration.append(final_time - initial_time)
    # print "DURATION: ", (final_time - initial_time)

    robot_fin = robot_obj_ops.get_obj_pos("mobile_base")
    robot_final_poses.append(robot_fin)

    obj_fin = robot_obj_ops.get_obj_pos(name)
    obj_final_poses.append(obj_fin)
    # print ("Wanted location: x=", wanted_x, " y=", wanted_y)
    # print ("Predicted speed: ", robots_speed)
    # print ("Predicted location: x=", predictions[0], " y=", predictions[1])
    # print("Real location: ", obj_fin.position.x, " ", obj_fin.position.y)

    # print "LASER CALLBACK"
    rospy.sleep(0.5)
    obj_counter = obj_counter + 1
    create_obj.delete_obj(name)
    # robot_pos = [robot_init.position.x, robot_init.position.y, robot_fin.position.x, robot_fin.position.y]
    # obj_pos = [obj_init.position.x, obj_init.position.y, obj_fin.position.x, obj_fin.position.y]

    # plt.plot([robot_pos[0], robot_pos[2]], [robot_pos[1], robot_pos[3]], 'g-')
    # plt.plot([obj_pos[0], obj_pos[2]], [obj_pos[1], obj_pos[3]], 'b-')
    # plt.plot([obj_pos[0], predictions[0]], [obj_pos[1], predictions[1]], 'r-')
    # plt.plot([obj_pos[0], wanted_x], [obj_pos[1], wanted_y], 'y-')
    # red_patch = mpatches.Patch(color='red', label='Predicted path')
    # blue_patch = mpatches.Patch(color='blue', label='Actual path')
    # green_patch = mpatches.Patch(color='green', label='Robot\'s path')
    # yellow_patch = mpatches.Patch(color='yellow', label='Desired path')
    # plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch], loc='best')
    # plt.xlabel('x axis of the plane')
    # plt.ylabel('y axis of the plane')
    # plt.show()

# LASER CALLBACK------------------------------------------------------------------------------------------------------------------------------------------------

def laser_callback(data):
    global robots_speed, robot_speed
    push_duration = random.uniform(0, 10)

    start = rospy.Time.now()
    robot_speed.append(robots_speed)

    rate = rospy.Rate(10)

    # print(push_duration)
    # im = Image.open("photo.jpg")
    # colors = im.getcolors()
    # print("colors ", colors)
    # data = im.getdata()
    # print("data ", data)
    # pixels = im.getpixel()
    # print("pixels ", pixels)
    img_title = rospy.get_param('~image_title', 'photo.jpg')
    camera = take_photo.main()

    while detect_object.main() < 100:
        move = Twist()
        move.angular.z = 0.1
        move.linear.x = 0
        pub_move.publish(move)
        img_title = rospy.get_param('~image_title', 'photo.jpg')
        camera = take_photo.main()

    while detect_object.main() < 2000:
        move = Twist()
        move.angular.z = 0
        move.linear.x = 0.5
        pub_move.publish(move)

    while rospy.Time.now() - start < rospy.Duration(push_duration):
        # print ('Number of points in laser scan is: ', len(data.ranges))
        # print ('The distance to the rightmost scanned point is: ', data.ranges[0])
        # print ('The distance to the leftmost scanned point is: ', data.ranges[-1])
        # print ('The distance to the middle scanned point is: ', data.ranges[len(data.ranges)/2])
        # rate.sleep()

        move = Twist()
        move.angular.z = 0
        move.linear.x = robots_speed
        pub_move.publish(move)
        # if not np.isnan(data.ranges[len(data.ranges)/2]):
        #     deli = input()
        #     move.angular.z = 0
        #     move.linear.x = robots_speed
        # elif not np.isnan(data.ranges[-1]):
        #     deli = input()
        #     move.angular.z = 0.1
        #     move.linear.x = robots_speed
        # elif  not np.isnan(data.ranges[0]):
        #     deli = input()
        #     move.angular.z = -0.1
        #     move.linear.x = robots_speed
        # else:
        #     # deli = input()
        #     # move.angular.z = 0.2
        #     move.linear.x = 1
        # pub_move.publish(move)
        # rate.sleep()

        state_robot = get_model_coordinates(model_name="mobile_base")
        state_obj = get_model_coordinates(model_name=name)
        if not (state_robot.twist.linear.x < 0.00001) or not (state_robot.twist.linear.x > -0.00001) or not (state_obj.twist.linear.x < 0.00001) or not (state_obj.twist.linear.x > -0.00001):
            continue
    get_result()


# INITALIZE ROBOT-----------------------------------------------------------------------------------------------------------------------------------------------

def initialize_robot():
    rospy.init_node('move_nonettobot')
    # global model
    # read_training_set()
    # print("read the training set, waiting for the training")
    # gr = train_model()
    # print("trained")
    global pub_move
    pub_move = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
    create_world()
    rospy.Subscriber("/scan", LaserScan, laser_callback, queue_size=1000)
    rospy.spin()
    pub_reset = rospy.Publisher('/nonetto_resetter', EmptyMsg, queue_size=10)
    rospy.Subscriber("/nonetto_resetter", EmptyMsg, callback)


if __name__ == '__main__':
    initialize_robot()