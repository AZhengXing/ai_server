#include "ai_server_node.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/dnn_node_data.h"
#include "hobot_cv/hobotcv_imgproc.h"
#include "dnn_node/util/output_parser/perception_common.h"
#include "dnn_node/util/output_parser/detection/ptq_yolo5_output_parser.h"
#include "dnn_node/util/output_parser/detection/ptq_yolo2_output_parser.h"
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include <fstream>
using hobot::easy_dnn::DNNInput;
int ResizeNV12Img(const char *in_img_data, const int &in_img_height,
                  const int &in_img_width, const int &scaled_img_height,
                  const int &scaled_img_width, cv::Mat &out_img, float &ratio);

AIServer::AIServer(const std::string &node_name, const NodeOptions &opts)
    : DnnNode{node_name, opts} {
  load_model_config();
  load_ai_model();
  load_ros_interfaces();
}

AIServer::~AIServer() {}

void AIServer::load_model_config() {
  std::ifstream fs{ai_model_config.model_config_file_path};
  if (fs.is_open() == false) {
    rclcpp::shutdown();
    throw std::runtime_error{"open failed"};
  }
  rapidjson::IStreamWrapper isw(fs);
  rapidjson::Document document;
  document.ParseStream(isw);
  if (document.HasParseError()) {
    rclcpp::shutdown();
    throw std::runtime_error{"parese config file failed"};
  }
  if (document.HasMember("model_file")) {
    ai_model_config.model_file_path = document["model_file"].GetString();
  }
  if (document.HasMember("model_name")) {
    ai_model_config.model_name = document["model_name"].GetString();
  }
  ai_server_config.parser = DnnParserType::YOLOV5_PARSER;
  auto ret = hobot::dnn_node::parser_yolov5::LoadConfig(document);
  if (ret == -1) {
    throw std::runtime_error{"parese config file failed"};
  }
}

void AIServer::load_ai_model() {
  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("AI_Server"), "Init failed!");
    rclcpp::shutdown();
    return;
  }
  if (GetModelInputSize(0, ai_model_config.model_input_w,
                        ai_model_config.model_input_h) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("AI_Server"), "Get model input size fail!");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("AI_Server"),
                "The model input width is %d and height is %d",
                ai_model_config.model_input_w, ai_model_config.model_input_h);
  }
}

void AIServer::load_ros_interfaces() {
  RCLCPP_INFO(rclcpp::get_logger("AI_Server"), "build ros_img_subscription_");
  // auto qos_setting = rmw_qos_profile_services_default;
  // qos_setting.depth = 1;
  service_cb_group_ =
      create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  ai_srv = create_service<IMGDetectionSrv>(
      "ai_server",
      std::bind(&AIServer::img_detection_service, this, std::placeholders::_1,
                std::placeholders::_2),
      rmw_qos_profile_services_default, service_cb_group_);
}

void AIServer::img_detection_service(
    const std::shared_ptr<IMGDetectionSrv::Request> req,
    std::shared_ptr<IMGDetectionSrv::Response> resp) {
  RCLCPP_DEBUG(rclcpp::get_logger("AI_Server"), "get img");
  resp->ts_start_detection = rclcpp::Clock().now().seconds();
  const auto &img = req->img;
  std::shared_ptr<DnnExampleOutput> dnn_output =
      std::make_shared<DnnExampleOutput>();
  auto fu = dnn_output->prom.get_future();
  RosImgProcess(img, dnn_output);
  auto ret = fu.get();
  resp->ts_finish_detection = rclcpp::Clock().now().seconds();
  resp->goals = std::move(ret);
  RCLCPP_DEBUG(rclcpp::get_logger("AI_Server"), "goals size: %d",
               resp->goals.size());
}

void AIServer::RosImgProcess(const sensor_msgs::msg::Image &img,
                             std::shared_ptr<DnnExampleOutput> dnn_output,
                             bool isSync) {
  if (!rclcpp::ok()) {
    return;
  }
  std::shared_ptr<hobot::easy_dnn::NV12PyramidInput> pyramid = nullptr;
  if (img.encoding == "nv12") {
    cv::Mat out_img;
    const auto img_h = static_cast<int>(img.height);
    const auto img_w = static_cast<int>(img.width);
    if (img_h != ai_model_config.model_input_h ||
        img_w != ai_model_config.model_input_w) {
      auto ret = ResizeNV12Img(
          reinterpret_cast<const char *>(img.data.data()), img.height,
          img.width, ai_model_config.model_input_h,
          ai_model_config.model_input_w, out_img, dnn_output->ratio);

      if (ret < 0) {
        return;
      }
      RCLCPP_INFO(rclcpp::get_logger("AI_Server"), "Resize img OK");
    }
    uint32_t out_img_width = (uint32_t)out_img.cols;
    uint32_t out_img_height = (uint32_t)out_img.rows * 2 / 3;
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
        reinterpret_cast<const char *>(out_img.data), out_img_height, out_img_width,
        ai_model_config.model_input_h, ai_model_config.model_input_w);
  } else {
    return;
  }

  if (pyramid != nullptr) {
    RCLCPP_INFO(rclcpp::get_logger("AI_Server"), "Get pyramid OK");
    auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};
    dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
    dnn_output->msg_header->set__frame_id(img.header.frame_id);
    dnn_output->msg_header->set__stamp(img.header.stamp);
    if (Run(inputs, dnn_output, nullptr, isSync) != 0) {
      RCLCPP_INFO(rclcpp::get_logger("AI_Server"), "Run predict failed!");
      return;
    }
  } else {
    RCLCPP_INFO(rclcpp::get_logger("AI_Server"), "Get pyramid failed");
    return;
  }
}

int AIServer::PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) {
  if (!rclcpp::ok()) {
    return -1;
  }
  auto parser_output = std::dynamic_pointer_cast<DnnExampleOutput>(outputs);
  if (parser_output->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("PostProcessBase"),
                 "Invalid node_output->output_tensors");
    return -1;
  }
  RCLCPP_INFO(rclcpp::get_logger("AI_Server"), "Get output ok");
  using hobot::dnn_node::output_parser::DnnParserResult;
  std::shared_ptr<DnnParserResult> det_result = nullptr;
  auto parse_ret =
      hobot::dnn_node::parser_yolov5::Parse(parser_output, det_result);
  (void)parse_ret;
  RCLCPP_INFO(rclcpp::get_logger("PostProcessBase"), "out box size: %d",
              det_result->perception.det.size());
  std::vector<IMGDetectionMsg> ret;
  for (auto &rect : det_result->perception.det) {
    if (rect.bbox.xmin < 0)
      rect.bbox.xmin = 0;
    if (rect.bbox.ymin < 0)
      rect.bbox.ymin = 0;
    if (rect.bbox.xmax >= ai_model_config.model_input_w) {
      rect.bbox.xmax = ai_model_config.model_input_w - 1;
    }
    if (rect.bbox.ymax >= ai_model_config.model_input_h) {
      rect.bbox.ymax = ai_model_config.model_input_h - 1;
    }

    // outputs->outputs
    IMGDetectionMsg msg;
    const auto& ratio = parser_output->ratio;
    msg.position_x = ratio*rect.bbox.xmin;
    msg.position_x_offset = ratio*(rect.bbox.xmax - rect.bbox.xmin);
    msg.position_y = ratio*rect.bbox.ymin;
    msg.position_y_offset = ratio*(rect.bbox.ymax - rect.bbox.ymin);
    msg.name = rect.class_name;
    msg.conf = rect.score;
    ret.emplace_back(std::move(msg));
    std::stringstream ss;
    ss << "det rect: " <<  rect.bbox.xmin << " " << rect.bbox.ymin << " "
       << rect.bbox.xmax << " " << rect.bbox.ymax
       << ", det type: " << rect.class_name << ", ratio:" << ratio;
    RCLCPP_INFO(rclcpp::get_logger("PostProcessBase"), "%s", ss.str().c_str());
  }
  parser_output->prom.set_value(ret);
  return 0;
}

int AIServer::SetNodePara(){
  dnn_node_para_ptr_->model_file = ai_model_config.model_file_path;
  dnn_node_para_ptr_->model_name = ai_model_config.model_name;
  dnn_node_para_ptr_->model_task_type =
      hobot::dnn_node::ModelTaskType::ModelInferType;
  dnn_node_para_ptr_->task_num = ai_server_config.parallel_task_num;
  return 0;
}


// 使用hobotcv resize nv12格式图片，固定图片宽高比
int ResizeNV12Img(const char *in_img_data,
                  const int &in_img_height,
                  const int &in_img_width,
                  const int &scaled_img_height,
                  const int &scaled_img_width,
                  cv::Mat &out_img,
                  float &ratio) {
  cv::Mat src(
      in_img_height * 3 / 2, in_img_width, CV_8UC1, (void *)(in_img_data));
  float ratio_w =
      static_cast<float>(in_img_width) / static_cast<float>(scaled_img_width);
  float ratio_h =
      static_cast<float>(in_img_height) / static_cast<float>(scaled_img_height);
  float dst_ratio = std::max(ratio_w, ratio_h);
  int resized_width, resized_height;
  if (dst_ratio == ratio_w) {
    resized_width = scaled_img_width;
    resized_height = static_cast<float>(in_img_height) / dst_ratio;
  } else if (dst_ratio == ratio_h) {
    resized_width = static_cast<float>(in_img_width) / dst_ratio;
    resized_height = scaled_img_height;
  }

  // hobot_cv要求输出宽度为16的倍数
  int remain = resized_width % 16;
  if (remain != 0) {
    //向下取16倍数，重新计算缩放系数
    resized_width -= remain;
    dst_ratio = static_cast<float>(in_img_width) / resized_width;
    resized_height = static_cast<float>(in_img_height) / dst_ratio;
  }
  //高度向下取偶数
  resized_height =
      resized_height % 2 == 0 ? resized_height : resized_height - 1;
  ratio = dst_ratio;

  return hobot_cv::hobotcv_resize(
      src, in_img_height, in_img_width, out_img, resized_height, resized_width);
}
