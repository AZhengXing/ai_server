cmake_minimum_required(VERSION 3.5)
project(ai_server)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies

# include_directories(/opt/tros/)
find_package(example_interfaces REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(dnn_node REQUIRED)
find_package(hobot_cv REQUIRED)
find_package(ai_service_msg REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(include
  /home/sunrise/xingzheng/ros_cv/install/ai_service_msg/include/
)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
link_directories(
  /usr/lib/hbbpu/
  /usr/lib/hbmedia/
  /usr/lib/
  /opt/tros/lib/
  /home/sunrise/xingzheng/ros_cv/install/ai_service_msg/lib/

)

add_executable(ai_server src/ai_server_main.cpp src/ai_server_node.cpp)
target_link_libraries(ai_server opencv_world dnn)
ament_target_dependencies(
  ai_server
  rclcpp
  dnn_node
  sensor_msgs
  std_msgs
  hobot_cv
  cv_bridge
  ai_service_msg
  example_interfaces
)

install(TARGETS ai_server
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
${PROJECT_SOURCE_DIR}/launch/
DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
