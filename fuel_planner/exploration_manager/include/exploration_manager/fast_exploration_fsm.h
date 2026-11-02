#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <thread>

using Eigen::Vector3d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace fast_planner
{
  class FastPlannerManager;
  class FastExplorationManager;
  class PlanningVisualization;
  struct FSMParam;
  struct FSMData;

  /************************************* 探索流程状态机 *******************************************/
  enum EXPL_STATE
  {
    INIT,         // 初始化
    WAIT_TRIGGER, // 等待探索开始
    PLAN_TRAJ,    // 规划探索轨迹
    PUB_TRAJ,     // 发布轨迹
    EXEC_TRAJ,    // 执行轨迹
    FINISH        // 完成探索
  };

  class FastExplorationFSM
  {
  private:
    /* planning utils */
    shared_ptr<FastPlannerManager> planner_manager_;  // 规划管理器:负责路径规划和轨迹优化
    shared_ptr<FastExplorationManager> expl_manager_; // 探索管理器:负责前沿搜索和探索策略
    shared_ptr<PlanningVisualization> visualization_; // 可视化工具:负责发布可视化消息到RViz

    shared_ptr<FSMParam> fp_; // 参数配置
    shared_ptr<FSMData> fd_;  // 状态数据
    EXPL_STATE state_;        // 当前状态

    bool classic_;

    /* ROS utils */
    ros::NodeHandle node_;                                              // ROS节点句柄
    ros::Timer exec_timer_, safety_timer_, vis_timer_, frontier_timer_; // 定时器，分别用于执行FSM主循环、检查安全、可视化和前沿搜索
    ros::Subscriber trigger_sub_, odom_sub_;                            // 订阅器，分别用于触发FSM状态转换和获取无人机里程计信息
    ros::Publisher replan_pub_, new_pub_, bspline_pub_, traj_path_pub_; // 发布器
    nav_msgs::Path traj_path_;  // [djg] 累积的探索轨迹

    /* helper functions */
    int callExplorationPlanner();                             // 规划调用
    void transitState(EXPL_STATE new_state, string pos_call); // 状态切换
    void resetExplorationStats();
    void openStatsLog();

    /* ROS functions */
    void FSMCallback(const ros::TimerEvent &e);                   // 状态机主循环
    void safetyCallback(const ros::TimerEvent &e);                // 碰撞检测
    void frontierCallback(const ros::TimerEvent &e);              // 前沿搜索
    void triggerCallback(const nav_msgs::PathConstPtr &msg);      // 触发回调
    void odometryCallback(const nav_msgs::OdometryConstPtr &msg); // 里程计回调
    void visualize();                                             // 可视化函数
    void clearVisMarker();                                        // 清除可视化标记

  public:
    FastExplorationFSM(/* args */)
    {
    }
    ~FastExplorationFSM()
    {
    }

    void init(ros::NodeHandle &nh);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

} // namespace fast_planner

#endif
