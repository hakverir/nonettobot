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

# CREATE THE SDF FILE TO SPECIFY SPHERE'S FEATURES--------------------------------------------------------------------------------------------------------------

def create_sdf_file_sphere(mass, radius):
  sdf_file = open("/home/nonetto/catkin_ws/src/nonettobot/src/sphere/model.sdf", "r+")
  file_parts = sdf_file.read()

  file_parts = file_parts.split("<mass>")
  part_two = file_parts[1].split("</mass>")
  new_file = file_parts[0] + "<mass>" + str(mass) + "</mass>" + part_two[1]

  file_parts = new_file.split("<radius>")
  part_two = file_parts[1].split("</radius>")
  part_one = file_parts[0] + "<radius>" + str(radius) + "</radius>" + part_two[1]
  part_two = file_parts[2].split("</radius>")
  new_file = part_one + "<radius>" + str(radius) + "</radius>" + part_two[1]

  sdf_file.seek(0)
  sdf_file.write(new_file)
  sdf_file.truncate()
  sdf_file.close()

# CREATE THE SDF FILE TO SPECIFY CUBE'S FEATURES--------------------------------------------------------------------------------------------------------------

def create_sdf_file_cube(mass, size_x, size_y, size_z):
  sdf_file = open("/home/nonetto/catkin_ws/src/nonettobot/src/cube/model.sdf", "r+")
  file_parts = sdf_file.read()

  file_parts = file_parts.split("<mass>")
  part_two = file_parts[1].split("</mass>")
  new_file = file_parts[0] + "<mass>" + str(mass) + "</mass>" + part_two[1]

  file_parts = new_file.split("<size>")
  part_two = file_parts[1].split("</size>")
  part_one = file_parts[0] + "<size>" + str(size_x) + " " + str(size_y) + " " + str(size_z) + "</size>" + part_two[1]
  part_two = file_parts[2].split("</size>")
  new_file = part_one + "<size>" + str(size_x) + " " + str(size_y) + " " + str(size_z) + "</size>" + part_two[1]

  sdf_file.seek(0)
  sdf_file.write(new_file)
  sdf_file.truncate()
  sdf_file.close()

# CREATE THE SDF FILE TO SPECIFY CYLINDER'S FEATURES--------------------------------------------------------------------------------------------------------------

def create_sdf_file_cylinder(mass, radius, length):
  sdf_file = open("/home/nonetto/catkin_ws/src/nonettobot/src/cylinder/model.sdf", "r+")
  file_parts = sdf_file.read()

  file_parts = file_parts.split("<mass>")
  part_two = file_parts[1].split("</mass>")
  new_file = file_parts[0] + "<mass>" + str(mass) + "</mass>" + part_two[1]

  file_parts = new_file.split("<radius>")
  part_two = file_parts[1].split("</radius>")
  part_one = file_parts[0] + "<radius>" + str(radius) + "</radius>" + part_two[1]
  part_two = file_parts[2].split("</radius>")
  new_file = part_one + "<radius>" + str(radius) + "</radius>" + part_two[1]

  file_parts = new_file.split("<length>")
  part_two = file_parts[1].split("</length>")
  part_one = file_parts[0] + "<length>" + str(length) + "</length>" + part_two[1]
  part_two = file_parts[2].split("</length>")
  new_file = part_one + "<length>" + str(length) + "</length>" + part_two[1]

  sdf_file.seek(0)
  sdf_file.write(new_file)
  sdf_file.truncate()
  sdf_file.close()

# CREATE OBJECT-------------------------------------------------------------------------------------------------------------------------------------------------

def create_obj(pos_x, pos_y, pos_z, shape, obj_counter):
  pose = Pose()
  pose.position.x = pos_x
  pose.position.y = pos_y
  pose.position.z = pos_z

  if shape == 0:
    model = open("/home/nonetto/catkin_ws/src/nonettobot/src/sphere/model.sdf")
  elif shape == 1:
    model = open("/home/nonetto/catkin_ws/src/nonettobot/src/cube/model.sdf")
  elif shape == 2:
    model = open("/home/nonetto/catkin_ws/src/nonettobot/src/cylinder/model.sdf")

  obj = SpawnModelRequest()

  obj.model_name = "obj_" + str(obj_counter)

  obj.model_xml = model.read()
  obj.robot_namespace = "/nonettobject"
  obj.initial_pose = pose

  try:
    global spawn_model
    ret = spawn_model(obj)
  except Exception, e:
    print('\n')
    print("----------------------")
    rospy.logerr('Error on calling service: %s',str(e))
    print("----------------------")
    print('\n')

# DELETE OBJECT-------------------------------------------------------------------------------------------------------------------------------------------------

def delete_obj(obj_name):
  req = DeleteModelRequest()
  req.model_name = obj_name
  exists = True
  try:
    res = delete_model(obj_name)
  except rospy.ServiceException, e:
    exists = False
    rospy.logdebug("Model %s does not exist in gazebo.", obj_name)