#include "dnn_node/dnn_node.h"
#include "rclcpp/rclcpp.hpp"
#include <string>
#include "sensor_msgs/msg/image.hpp"
#include "dnn_node/dnn_node_data.h"
#include "dnn_node/util/image_proc.h"
#include "ai_service_msg/srv/img_detection_srv.hpp"
#include "ai_service_msg/msg/img_detection_output.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"
#include <future> 
#include <vector>
using hobot::dnn_node::DnnNode;
using rclcpp::NodeOptions;
using hobot::dnn_node::DnnNodeOutput;
using IMGDetectionMsg = ai_service_msg::msg::IMGDetectionOutput;
using IMGDetectionSrv = ai_service_msg::srv::IMGDetectionSrv;
struct DnnExampleOutput : public DnnNodeOutput {
  // resize参数，用于算法检测结果的映射
  float ratio = 1.0;  //缩放比例系数，无需缩放为1

  // 算法推理使用的图像数据，用于本地渲染使用
  std::shared_ptr<hobot::easy_dnn::NV12PyramidInput> pyramid = nullptr;
  
  // 前处理的开始和结束时间，用于发布perf统计信息
  struct timespec preprocess_timespec_start;
  struct timespec preprocess_timespec_end;

  // 订阅到的图像数据和模型输入分辨率，unet算法后处理使用
  int img_w = 0;
  int img_h = 0;
  int model_w = 0;
  int model_h = 0;
  // async output
  std::promise<std::vector<IMGDetectionMsg>> prom;
  // img_name
  std::string img_name;

};

class AIServer : public DnnNode {
public:
  ~AIServer() override;
  AIServer(const std::string &node_name,
           const NodeOptions &opts = NodeOptions{});
private:
  void img_detection_service(const std::shared_ptr<IMGDetectionSrv::Request> req,
       std::shared_ptr<IMGDetectionSrv::Response> resp);
  void load_model_config();
  void load_ai_model();
  void load_ros_interfaces();
  void RosImgProcess(const sensor_msgs::msg::Image::ConstSharedPtr msg){
    RosImgProcess(*msg);
  }
  void RosImgProcess(const sensor_msgs::msg::Image &img,
                     std::shared_ptr<DnnExampleOutput> dnn_output = nullptr,
                     bool isSync = false);
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;
  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr
      ros_img_subscription_ = nullptr;

  rclcpp::Service<IMGDetectionSrv>::ConstSharedPtr ai_srv =
      nullptr;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  enum class DnnParserType {
    INVALID_PARSER = 0,
    YOLOV2_PARSER = 1,     
    YOLOV3_PARSER,          
    YOLOV5_PARSER,         
    YOLOV5X_PARSER,      
    CLASSIFICATION_PARSER,
    SSD_PARSER,
    EFFICIENTDET_PARSER,
    FCOS_PARSER,
    UNET_PARSER
    /*define more*/
  };
  struct AIServerConfig{
    size_t parallel_task_num = 4;
    DnnParserType parser = DnnParserType::INVALID_PARSER;
  } ai_server_config;

  struct AIModelConfig
  {
    std::string model_config_file_path{"/home/sunrise/xingzheng/ros_cv/install/ai_server/lib/ai_server/yolov5workconfig.json"};
    std::string model_file_path = "";
    std::string model_name = "";
    int model_input_w = 0;
    int model_input_h = 0;
  } ai_model_config;

  struct ImgMSGConfig{
    std::string topic_name = "/cam1/bgr8";
  } img_msg_config;//for test


};