#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>
#include <limits>

ros::Publisher circle_pub;
ros::Publisher velocity_pub;
ros::Publisher acceleration_pub;
ros::Publisher cloud_pub;
ros::Publisher cloud_pub2; 

pcl::PointXYZ prev_center;
ros::Time prev_time;
geometry_msgs::TwistStamped prev_velocity;

double computeDistance(double x, double y) {
    return sqrt(x * x + y * y);
}

cv::Mat pointCloudToImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int img_size = 500) {
    cv::Mat img = cv::Mat::zeros(img_size, img_size, CV_8UC1);

    for (const auto& point : cloud->points) {
        int x = static_cast<int>((point.x + 5.0) * (img_size / 10.0));
        int y = static_cast<int>((point.y + 5.0) * (img_size / 10.0));
        if (x >= 0 && x < img_size && y >= 0 && y < img_size) {
            img.at<uchar>(y, x) = 255;
        }
    }
    return img;
}

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");

    for (double z_min = 0; z_min <= 1.0; z_min += 0.01) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pass.setFilterLimits(z_min, z_min + 0.1);
        pass.filter(*filtered_cloud);
        *merged_cloud += *filtered_cloud;
    }

    sensor_msgs::PointCloud2 output_cloud_msg2;
    pcl::toROSMsg(*merged_cloud, output_cloud_msg2);
    output_cloud_msg2.header.frame_id = "livox_frame";
    output_cloud_msg2.header.stamp = ros::Time::now();
    cloud_pub2.publish(output_cloud_msg2);

    for (auto& point : merged_cloud->points) {
        point.z = 0.0;
    }

    sensor_msgs::PointCloud2 output_cloud_msg;
    pcl::toROSMsg(*merged_cloud, output_cloud_msg);
    output_cloud_msg.header.frame_id = "livox_frame";
    output_cloud_msg.header.stamp = ros::Time::now();
    cloud_pub.publish(output_cloud_msg);

    cv::Mat img = pointCloudToImage(merged_cloud);

    cv::imshow("PointCloud to Image", img);
    cv::waitKey(1);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(img, circles, cv::HOUGH_GRADIENT, 1, 20, 100, 30, 1, 50);

    if (circles.empty()) {
        ROS_WARN("Could not find any circles in the point cloud.");
        return;
    }
    
    double nearest_distance = std::numeric_limits<double>::max();
    cv::Vec3f nearest_circle;
    for (const auto& circle : circles) {
        double distance = computeDistance(circle[0], circle[1]);
        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_circle = circle;
        }
    }

    pcl::PointXYZ nearest_center;
    nearest_center.x = (nearest_circle[0] / 500.0) * 10.0 - 5.0;
    nearest_center.y = (nearest_circle[1] / 500.0) * 10.0 - 5.0;
    nearest_center.z = 0.0;

    cv::circle(img, cv::Point(cvRound(nearest_circle[0]), cvRound(nearest_circle[1])),
               cvRound(nearest_circle[2]), cv::Scalar(255, 0, 0), 2);
    cv::circle(img, cv::Point(cvRound(nearest_circle[0]), cvRound(nearest_circle[1])),
               2, cv::Scalar(0, 255, 0), 3);

    cv::imshow("Hough Circle Detection", img);
    cv::waitKey(1);

    geometry_msgs::PointStamped circle_center;
    circle_center.header.stamp = ros::Time::now();
    circle_center.point.x = nearest_center.x;
    circle_center.point.y = nearest_center.y;
    circle_center.point.z = nearest_center.z;
    circle_pub.publish(circle_center);

    ros::Time current_time = ros::Time::now();
    double dt = (current_time - prev_time).toSec();

    if (dt > 0) {
        geometry_msgs::TwistStamped velocity;
        velocity.header.stamp = current_time;
        velocity.twist.linear.x = (nearest_center.x - prev_center.x) / dt;
        velocity.twist.linear.y = (nearest_center.y - prev_center.y) / dt;
        velocity_pub.publish(velocity);

        geometry_msgs::TwistStamped acceleration;
        acceleration.header.stamp = current_time;
        acceleration.twist.linear.x = (velocity.twist.linear.x - prev_velocity.twist.linear.x) / dt;
        acceleration.twist.linear.y = (velocity.twist.linear.y - prev_velocity.twist.linear.y) / dt;
        acceleration_pub.publish(acceleration);

        prev_velocity = velocity;
    }

    prev_center = nearest_center;
    prev_time = current_time;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "circle_fitting_node");
    ros::NodeHandle nh;

    ros::Subscriber cloud_sub = nh.subscribe("/livox/lidar", 1, pointCloudCallback);

    circle_pub = nh.advertise<geometry_msgs::PointStamped>("circle_center", 1);
    velocity_pub = nh.advertise<geometry_msgs::TwistStamped>("circle_velocity", 1);
    acceleration_pub = nh.advertise<geometry_msgs::TwistStamped>("circle_acceleration", 1);
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
    cloud_pub2 = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud2", 1);

    prev_time = ros::Time::now();

    ros::spin();
    return 0;
}