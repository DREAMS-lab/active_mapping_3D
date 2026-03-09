cmake_minimum_required(VERSION 3.8)
project(gsplat)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
  launch
  config
  models
  worlds
  scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/gimbal_control.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
