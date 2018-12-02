
#include <ros/ros.h>
#include <cmath>
#include <string>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <tf2_ros/transform_listener.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PointStamped.h>
#include <obstacle_detection/Obstacle.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;


class DetectObstacleNode
{
public:



    DetectObstacleNode(){
        // constructor
        nh_ = ros::NodeHandle("~");
        dist_from_floor_ = 0.015;
        range_ = 0.06;
        point_counter_ = 0;
        point_threshold_ = 750;
        squared_radius_ = std::pow(0.23,2);

        avg_obstacle_x_ = 0.0;
        avg_obstacle_y_ = 0.0;

        min_y_ = 100.0;
        min_x_ = 0.0;
        max_y_ = -100.0;
        max_x_ = 0.0;

        obstacle_detected_.data = false;

        frame_skipper_ = 0;

        pointcloud_sub_ = nh_.subscribe<PointCloud>("/camera/depth/points",1,&DetectObstacleNode::pointcloud_callback, this);
        obstacle_detected_pub_ = nh_.advertise<std_msgs::Bool>("/wall_detected",1);
        //obstacle_detected_pub_ = nh_.advertise<std_msgs::Bool>("/obstacle_detected_perception",1);
        point_pos_pub_ = nh_.advertise<obstacle_detection::Obstacle>("/found_obstacle_perception",1);
    }

    ~DetectObstacleNode(){

    }

    void pointcloud_callback(const PointCloud::ConstPtr& msg){
        //Skip frames to reduce computational complexity: 2 frames/second
        if (frame_skipper_ == 0){
          frame_skipper_ = 0;
          min_y_ = 100.0;
          min_x_ = 0.0;
          max_y_ = -100.0;
          max_x_ = 0.0;
          //Transform the poitncloud such that the camera frame is alligned with the world frame.
          tf::StampedTransform transform;
          try
          {
              tf_listener_.lookupTransform("/base_link",msg->header.frame_id, ros::Time(0), transform);
              ROS_INFO("Waiting for Transform");
          }
          catch (tf::TransformException ex)
          {
              ROS_ERROR("%s",ex.what());
          }
          //ROS_INFO("Doing Transform");
          pcl_ros::transformPointCloud(*msg,transformed_pointcloud_,transform);

          obstacle_detected_.data = false;
          //ROS_INFO("New Pointcloud");
          //Iterate over Pointcloud
          point_counter_ = 0;
          for(PointCloud::iterator it = transformed_pointcloud_.begin(); it != transformed_pointcloud_.end(); it++)
          {
            //Remove points with value NAN
            if(!std::isnan(it->x)){
                //Check only the point in the right height range: dist_from_floor_ -> dist_from_floor_ + range_
                //ROS_INFO("Height:%f",it->z);
                if(it->z >= dist_from_floor_ && it->z < dist_from_floor_ + range_){
  		//ROS_INFO("X: %f , Y: %f", it->x, it->y);
                    //Check if point is in stopping radius_
                    if((std::pow(it->x,2)+std::pow(it->y,2))<squared_radius_){
                        //Increment number of counter points in radius and update the average x position and y position
                        point_counter_++;
                        if (point_counter_ >= point_threshold_){
                          break;
                        }
                        avg_obstacle_x_ += it->x;
                        avg_obstacle_y_ += it->y;
                        //Determine the span of the object in y direction
                        if (it->y <= min_y_){
                            min_y_ = it->y;
                            min_x_ = it->x;
                        }
                        if(it->y >= max_y_){
                          max_y_ = it->y;
                          max_x_ = it->x;
                        }
                    }
                }
            }
          }
          //ROS_INFO("Pointcounter:%i",point_counter_);
          //Check if enough point in radius were detected
          if(point_counter_ >= point_threshold_){
            //Publish obstacle_detected_ true
            obstacle_detected_.data = true;
            obstacle_detected_pub_.publish(obstacle_detected_);
            ROS_INFO("Obstacle detected with Perception");
            // Calculate AVG position of obstacle and publish
            if(point_counter_ != 0){

                point_pos_.header.frame_id = "/base_link";
                point_pos_.point.x = min_x_;
                point_pos_.point.y = min_y_;
                ROS_INFO("Minimum Robot Coordinates:\nx: %f , y: %f",point_pos_.point.x,point_pos_.point.y);
                point_pos_.point.z = 0.0;
                tf_listener_.transformPoint("/map",point_pos_,point_pos_map_);
                point_pos_map_.point.z = 0.0;
                obstacle_pos_.positions[0] = point_pos_map_;
		            ROS_INFO("Minimum Map Coordinates:\nx: %f , y: %f",point_pos_map_.point.x,point_pos_map_.point.y);


                point_pos_.header.frame_id = "/base_link";
                point_pos_.point.x = max_x_;
                point_pos_.point.y = max_y_;
                point_pos_.point.z = 0.0;
                ROS_INFO("Maximum Robot Coordinates:\nx: %f , y: %f",point_pos_.point.x,point_pos_.point.y);
                tf_listener_.transformPoint("/map",point_pos_,point_pos_map_);
                point_pos_map_.point.z = 0.0;
                obstacle_pos_.positions[1] = point_pos_map_;
		            ROS_INFO("Maximum Map Coordinates:\nx: %f , y: %f",point_pos_map_.point.x,point_pos_map_.point.y);


                avg_obstacle_x_ = avg_obstacle_x_/ ((float) point_counter_);
                avg_obstacle_y_ = avg_obstacle_y_/ ((float) point_counter_);
                point_pos_.header.frame_id = "/base_link";
                point_pos_.point.x = avg_obstacle_x_;
                point_pos_.point.y = avg_obstacle_y_;
                point_pos_.point.z = 0.0;
                ROS_INFO("Detected average robot coordinates\nx: %f , y: %f",point_pos_.point.x,point_pos_.point.y);
                tf_listener_.transformPoint("/map",point_pos_,point_pos_map_);
                point_pos_map_.point.z = 0.0;
                obstacle_pos_.positions[2] = point_pos_map_;
                ROS_INFO("Published average map coordinates:\nx: %f , y: %f",point_pos_map_.point.x,point_pos_map_.point.y);
                point_pos_pub_.publish(obstacle_pos_);

            }
          }
          else{
              obstacle_detected_.data = false;
              obstacle_detected_pub_.publish(obstacle_detected_);
          }
        }
        else{
          frame_skipper_ ++;
        }

    }

    void detect(){
        ros::spin();
    }

private:
  ros::NodeHandle nh_;
  std_msgs::Bool obstacle_detected_;
  geometry_msgs::PointStamped point_pos_;
  geometry_msgs::PointStamped point_pos_map_;
  obstacle_detection::Obstacle obstacle_pos_;

  double dist_from_floor_;
  double range_;
  double min_y_;
  double min_x_;
  double max_y_;
  double max_x_;
  double avg_obstacle_x_;
  double avg_obstacle_y_;
  double squared_radius_;
  int point_counter_;
  int point_threshold_;
  int frame_skipper_;

  PointCloud transformed_pointcloud_;

  ros::Subscriber pointcloud_sub_;
  ros::Publisher obstacle_detected_pub_;
  ros::Publisher point_pos_pub_;
  tf::TransformListener tf_listener_;
};



int main(int argc, char **argv)
{


  ros::init(argc, argv, "detect_obstacle");
  DetectObstacleNode detect_obstacle;

  detect_obstacle.detect();

  return 0;
}
