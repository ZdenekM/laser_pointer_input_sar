#ifndef __LASER3DTRACKING_H_
#define __LASER3DTRACKING_H_

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>

#include <nn_laser_spot_tracking/KeypointImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Header.h>
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include <Eigen/Dense>

#include <utils/SecondOrderFilter.h>

#include <mutex>
#include <array>

namespace NNLST {

class Laser3DTracking {
    
public:
    Laser3DTracking(ros::NodeHandle* nh, const double& period);
    
    bool isReady();
    
    int run();
    
private:
    ros::NodeHandle* _nh;
    double _period;
    
    std::string _camera_frame;
    std::string _ref_frame ;
    std::string _laser_spot_frame;
    
    double _detection_confidence_threshold;
    double _cloud_detection_max_sec_diff;
    ros::Subscriber _keypoint_sub;
    void keypointSubClbk(const nn_laser_spot_tracking::KeypointImageConstPtr& msg);
    nn_laser_spot_tracking::KeypointImage _keypoint_image;
    
    tf2_ros::TransformBroadcaster _tf_broadcaster;
    std::vector<geometry_msgs::TransformStamped> _ref_T_spot; //one for raw, other for filtered. tf2_ros wants vector, cant use std::array

    ros::Subscriber _depth_sub;
    ros::Subscriber _info_sub;
    void depthClbk(const sensor_msgs::ImageConstPtr& msg);
    void infoClbk(const sensor_msgs::CameraInfoConstPtr& msg);
    cv::Mat _depth_image;
    std_msgs::Header _depth_header;
    std::mutex _depth_mutex;
    bool _has_depth = false;
    bool _has_info = false;
    image_geometry::PinholeCameraModel _cam_model;

    /***************************************************** */

    bool sendTransformFrom2D();
    bool updateTransform(const cv::Mat& depth_image, const std_msgs::Header& depth_header);
    
    /***************  FILTER    **********/
    NNLST::utils::FilterWrap<Eigen::Vector3d>::Ptr _laser_pos_filter;
    double _filter_damping, _filter_bw;
    
    std::unique_ptr<ddynamic_reconfigure::DDynamicReconfigure> _ddr_server;
    void ddr_callback_filter_damping(double new_value);
    void ddr_callback_filter_bw(double new_value);
    
};

}

#endif // __LASER3DTRACKING_H_
