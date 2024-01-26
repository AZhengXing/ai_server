#include "ai_server_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  RCLCPP_WARN(rclcpp::get_logger("example"), "This is dnn node example!");
  rclcpp::executors::MultiThreadedExecutor ex;
  auto ai_server_node = std::make_shared<AIServer>("body_det");
  ex.add_node(ai_server_node);
  ex.spin();
  rclcpp::shutdown();
  return 0;
}