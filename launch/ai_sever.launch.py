from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # DeclareLaunchArgument(
        #     'publish_image_source',
        #     default_value='./config/test1.nv12',
        #     description='image source'),
        # DeclareLaunchArgument(
        #     'publish_image_format',
        #     default_value='nv12',
        #     description='image format'),
        # DeclareLaunchArgument(
        #     'publish_message_topic_name',
        #     default_value='/test_msg',
        #     description='The topic name of message'),
        # DeclareLaunchArgument(
        #     'publish_output_image_w',
        #     default_value='0',
        #     description='output image width'),
        # DeclareLaunchArgument(
        #     'publish_output_image_h',
        #     default_value='0',
        #     description='output image height'),
        # DeclareLaunchArgument(
        #     'publish_source_image_w',
        #     default_value='960',
        #     description='source image width, nv12 format must set'),
        # DeclareLaunchArgument(
        #     'publish_source_image_h',
        #     default_value='544',
        #     description='source image height, nv12 format must set'),
        # DeclareLaunchArgument(
        #     'publish_fps',
        #     default_value='10',
        #     description='topic publish fps, 1~30'),
        # DeclareLaunchArgument(
        #     'publish_is_loop',
        #     default_value='True',
        #     description='loop publish or not'),
        # DeclareLaunchArgument(
        #     'publish_is_shared_mem',
        #     default_value='True',
        #     description='using zero copy or not'),
        # 启动图片发布pkg，output_image_w与output_image_h设置为0代表不改变图片的分辨率
        Node(
            package='ai_server',
            executable='ai_server',
            output='screen',
            # parameters=[
            #     {"image_source": LaunchConfiguration('publish_image_source')},
            #     {"image_format": LaunchConfiguration('publish_image_format')},
            #     {"msg_pub_topic_name": LaunchConfiguration(
            #         'publish_message_topic_name')},
            #     {"output_image_w": LaunchConfiguration(
            #         'publish_output_image_w')},
            #     {"output_image_h": LaunchConfiguration(
            #         'publish_output_image_h')},
            #     {"source_image_w": LaunchConfiguration(
            #         'publish_source_image_w')},
            #     {"source_image_h": LaunchConfiguration(
            #         'publish_source_image_h')},
            #     {"fps": LaunchConfiguration('publish_fps')},
            #     {"is_loop": LaunchConfiguration('publish_is_loop')},
            #     {"is_shared_mem": LaunchConfiguration('publish_is_shared_mem')}
            # ],
            arguments=['--ros-args', '--log-level', 'error']
        )
    ])
