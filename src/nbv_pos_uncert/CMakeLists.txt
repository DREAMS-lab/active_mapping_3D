cmake_minimum_required(VERSION 3.8)
project(nbv_pos_uncert)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  PATTERN "__pycache__" EXCLUDE)

ament_package()
