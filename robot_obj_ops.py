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

get_model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)

# GET THE OBJECT'S POSITION-------------------------------------------------------------------------------------------------------------------------------------

def get_obj_pos(name):
    global get_model_coordinates
    try:
        state = get_model_coordinates(model_name=name)
        return state.pose
    except Exception, e:
        rospy.logerr('Error on calling service: %s', str(e))
        return


# GET ROBOT'S POSITION------------------------------------------------------------------------------------------------------------------------------------------

def get_robot_pos():
    global get_model_coordinates
    try:
        state = get_model_coordinates(model_name="mobile_base")
        return state.pose
    except Exception, e:
        rospy.logerr('Error on calling service: %s', str(e))
        return


# CHANGE ROBOT'S POSITION---------------------------------------------------------------------------------------------------------------------------------------

def set_robot_pos(pos_x, pos_y, pos_z):
    pose = Pose()
    pose.position.x = pos_x
    pose.position.y = pos_y
    pose.position.z = pos_z

    state = ModelState()
    state.model_name = "mobile_base"
    state.pose = pose

    try:
        global set_model_state
        ret = set_model_state(state)
    except Exception, e:
        print('\n')
        print("----------------------")
        rospy.logerr('Error on calling service: %s', str(e))
        print("----------------------")
        print('\n')


# CALCULATE VOLUME ---------------------------------------------------------------------------------------------------------------------------------------------

def get_volume(shape, radius, length, edgex, edgey, edgez):
    if shape == 0:
        return (4 * math.pi * (radius ** 3)) / 3
    elif shape == 1:
        return edgex * edgey * edgez
    elif shape == 2:
        return math.pi * (radius ** 2) * length


# WAIT UNTIL THE OBJECT STOPS ----------------------------------------------------------------------------------------------------------------------------------

def wait_till_obj_stops(name):
    global get_model_coordinates
    try:
        state = get_model_coordinates(model_name=name)

        initial_time = rospy.Time.now()
        # print "LINEAR: ", state.twist.linear.x
        while not (state.twist.linear.x < 0.00001) or not (state.twist.linear.x > -0.00001):
            state = get_model_coordinates(model_name=name)
            # print "WAITING"
        final_time = rospy.Time.now()
        # print "RETURNING"
        return

    except Exception, e:
        rospy.logerr('Error on calling service: %s', str(e))


def wait_for_robot(radius):
    global get_model_coordinates
    try:
        state = get_model_coordinates(model_name="mobile_base")

        while not (state.twist.linear.x < 0.00001) or not (state.twist.linear.x > -0.00001):
            state = get_model_coordinates(model_name="mobile_base")

        return

    except Exception, e:
        rospy.logerr('Error on calling service: %s', str(e))
