cmake_minimum_required(VERSION 3.8)
project(mapping)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  PATTERN "__pycache__" EXCLUDE)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/orbslam3.sh")

ament_package()
