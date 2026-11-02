
#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <active_perception/grid_planner.h>
#include <traj_utils/planning_visualization.h>

#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/expl_data.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>

#include <ctime>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

using Eigen::Vector4d;

namespace fast_planner {
namespace {
string parentDir(const string& path) {
  const size_t pos = path.find_last_of('/');
  if (pos == string::npos) return "";
  if (pos == 0) return "/";
  return path.substr(0, pos);
}

bool ensureDirectory(const string& dir) {
  if (dir.empty() || dir == ".") return true;

  string cur;
  size_t pos = 0;
  if (dir[0] == '/') {
    cur = "/";
    pos = 1;
  }

  while (pos <= dir.size()) {
    const size_t slash = dir.find('/', pos);
    const string part = dir.substr(pos, slash == string::npos ? string::npos : slash - pos);
    if (!part.empty()) {
      if (!cur.empty() && cur.back() != '/') cur += "/";
      cur += part;
      if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
        ROS_ERROR("[StatsLog] Failed to create directory %s", cur.c_str());
        return false;
      }
    }
    if (slash == string::npos) break;
    pos = slash + 1;
  }
  return true;
}
}  // namespace

void FastExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("fsm/no_frontier_confirm_rounds", fp_->no_frontier_confirm_rounds_, 3);
  nh.param("exploration/enable_stats_log", fd_->enable_stats_log_, true);
  nh.param("exploration/stats_log_path", fd_->stats_log_path_, string(""));
  nh.param("exploration/enable_fsm_handoff_log", fd_->enable_handoff_log_, true);
  nh.param("exploration/fsm_handoff_log_path", fd_->handoff_log_path_,
           string("/home/djg/Data_debug/fsm_handoff_log.csv"));
  if (fd_->stats_log_path_.empty()) {
    string decision_trace_path;
    nh.param("exploration/decision_trace_log_path", decision_trace_path,
             string("/home/djg/Data_debug/decision_trace_log.csv"));
    const string log_dir = parentDir(decision_trace_path);
    fd_->stats_log_path_ = log_dir.empty() ? "/home/djg/Data_debug/exploration_stats.csv"
                                           : log_dir + "/exploration_stats.csv";
  }
  if (fd_->enable_handoff_log_) {
    const string handoff_dir = parentDir(fd_->handoff_log_path_);
    if (ensureDirectory(handoff_dir)) {
      fd_->handoff_log_file_.open(fd_->handoff_log_path_.c_str());
      if (fd_->handoff_log_file_.is_open()) {
        // [MODIFIED BY CODEX]: Event-only timing record; do not log 100 Hz commands here.
        fd_->handoff_log_file_
            << "round,event,plan_begin,plan_end,plan_duration,replan_time,"
            << "old_traj_id,old_start,old_duration,old_remaining,old_stop_deadline,"
            << "pred_start_x,pred_start_y,pred_start_z,pred_start_speed,pred_yaw,"
            << "result,new_traj_id,new_start,new_duration,finish_before_old_stop\n";
        fd_->handoff_log_file_.flush();
        ROS_WARN_STREAM("[FsmHandoffLog] Logging to " << fd_->handoff_log_path_);
      }
    }
  }

  /* Initialize main modules */
  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));

  planner_manager_ = expl_manager_->planner_manager_;
  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH" };
  fd_->static_state_ = true;
  fd_->trigger_ = false;

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &FastExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.5), &FastExplorationFSM::frontierCallback, this);

  trigger_sub_ =
      nh.subscribe("/waypoint_generator/waypoints", 1, &FastExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);

  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);
  traj_path_pub_ = nh.advertise<nav_msgs::Path>("/planning/trajectory_path", 1, true);
  traj_path_.header.frame_id = "world";
}

void FastExplorationFSM::resetExplorationStats() {
  fd_->expl_start_time_ = ros::Time::now();
  fd_->expl_active_ = true;
  fd_->total_traj_length_ = 0.0;
  fd_->has_last_odom_ = false;
  fd_->last_stats_log_time_ = ros::Time::now();
}

void FastExplorationFSM::openStatsLog() {
  if (!fd_->enable_stats_log_) return;

  if (fd_->stats_log_file_.is_open()) {
    fd_->stats_log_file_.close();
  }

  const string log_dir = parentDir(fd_->stats_log_path_);
  if (!ensureDirectory(log_dir)) {
    ROS_ERROR("[StatsLog] Disabled because log directory is unavailable: %s", log_dir.c_str());
    return;
  }

  fd_->stats_log_file_.open(fd_->stats_log_path_.c_str());
  if (fd_->stats_log_file_.is_open()) {
    fd_->stats_log_file_ << "elapsed_sec,coverage_ratio,explored_volume_m3,traj_length_m,speed_mps,"
                            "pos_x,pos_y,pos_z,grid_id,known_voxels,total_voxels" << std::endl;
    fd_->stats_log_file_.flush();
    ROS_WARN("[StatsLog] Logging to %s", fd_->stats_log_path_.c_str());
  } else {
    ROS_ERROR("[StatsLog] Failed to open %s", fd_->stats_log_path_.c_str());
  }
}

void FastExplorationFSM::FSMCallback(const ros::TimerEvent& e) {
  ROS_INFO_STREAM_THROTTLE(1.0, "[FSM]: state: " << fd_->state_str_[int(state_)]);

  // Periodic exploration stats logging (every 1 second)
  if (fd_->expl_active_ && fd_->stats_log_file_.is_open()) {
    ros::Time now = ros::Time::now();
    if ((now - fd_->last_stats_log_time_).toSec() >= 1.0) {
      fd_->last_stats_log_time_ = now;
      double elapsed = (now - fd_->expl_start_time_).toSec();

      // Direct voxel-count coverage from SDF map (cumulative, full-map).
      // Volume = coverage_ratio × 40×40×2 = 3200 m³ for cross-paper comparison.
      double coverage_ratio = 0.0;
      double explored_volume_m3 = 0.0;
      int known_voxels = 0, total_voxels = 0;

      if (expl_manager_ && expl_manager_->getSDFMap()) {
        auto& sdf = *expl_manager_->getSDFMap();
        int known = 0, total = 0;
        sdf.countKnownInBox(known, total);
        known_voxels = known;
        total_voxels = total;
        coverage_ratio = total > 0 ? static_cast<double>(known) / static_cast<double>(total) : 0.0;
        explored_volume_m3 = coverage_ratio * 3200.0;
      }

      double speed_mps = fd_->odom_vel_.norm();

      // [ABLATION] 位置与所在网格：回访航程占比、区域驻留、后期时间占比都依赖这几列
      int cur_grid_id = -1;
      if (expl_manager_ && expl_manager_->grid_planner_) {
        cur_grid_id = expl_manager_->grid_planner_->getGridContainingPoint(fd_->odom_pos_);
      }

      fd_->stats_log_file_ << elapsed << ","
                           << coverage_ratio << ","
                           << explored_volume_m3 << ","
                           << fd_->total_traj_length_ << ","
                           << speed_mps << ","
                           << fd_->odom_pos_.x() << ","
                           << fd_->odom_pos_.y() << ","
                           << fd_->odom_pos_.z() << ","
                           << cur_grid_id << ","
                           << known_voxels << ","
                           << total_voxels << std::endl;
      fd_->stats_log_file_.flush();
    }
  }

  switch (state_) {
    case INIT: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "no odom.");
        return;
      }
      // Go to wait trigger when odom is ok
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    case WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "wait for trigger.");
      break;
    }

    case FINISH: {
      ROS_INFO_THROTTLE(1.0, "finish exploration.");
      // Close stats log file
      if (fd_->stats_log_file_.is_open()) {
        fd_->stats_log_file_.close();
        fd_->expl_active_ = false;
        ROS_WARN("[StatsLog] Exploration finished, log saved.");
      }
      if (fd_->handoff_log_file_.is_open()) {
        fd_->handoff_log_file_.close();
      }
      break;
    }

    case PLAN_TRAJ: {
      // Fallback: initialize stats logging if triggerCallback didn't
      // (e.g. trigger message arrived before WAIT_TRIGGER state).
      if (!fd_->expl_active_) {
        resetExplorationStats();
        openStatsLog();
      }
      const ros::Time plan_begin = ros::Time::now();
      int old_traj_id = -1;
      double old_start = -1.0, old_duration = -1.0, old_remaining = -1.0;
      double old_stop_deadline = -1.0;
      if (!fd_->static_state_) {
        LocalTrajData* old_info = &planner_manager_->local_data_;
        old_traj_id = old_info->traj_id_;
        old_start = old_info->start_time_.toSec();
        old_duration = old_info->duration_;
        old_remaining = old_duration - (plan_begin - old_info->start_time_).toSec();
        // traj_server uses this same 0.3 s grace interval after /planning/replan.
        old_stop_deadline = plan_begin.toSec() + 0.3 + fp_->replan_time_;
      }
      if (fd_->static_state_) {
        // Plan from static state (hover)
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_acc_.setZero();

        fd_->start_yaw_(0) = fd_->odom_yaw_;
        fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
      } else {
        // Replan from non-static state, starting from 'replan_time' seconds later
        LocalTrajData* info = &planner_manager_->local_data_;
        double t_r = (ros::Time::now() - info->start_time_).toSec() + fp_->replan_time_;

        fd_->start_pt_ = info->position_traj_.evaluateDeBoorT(t_r);
        fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
        fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
        fd_->start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
      }

      // [MODIFIED BY CODEX] Inform traj_server the replanning BEFORE planning.
      // Root cause of the "segment stop + velocity zero" bug: the active spline
      // must be truncated (0.3s + replan_time grace period) BEFORE planning so
      // the successor can cut in at a non-zero velocity.  Delaying replan_pub_
      // until after SUCCEED let the old spline run to its natural zero-velocity
      // end, producing the stop-and-go behaviour between segments.
      replan_pub_.publish(std_msgs::Empty());
      int res = callExplorationPlanner();
      const ros::Time plan_end = ros::Time::now();
      if (fd_->handoff_log_file_.is_open()) {
        const LocalTrajData* new_info = &planner_manager_->local_data_;
        const bool has_new = res == SUCCEED;
        const double new_start = has_new ? new_info->start_time_.toSec() : -1.0;
        const double new_duration = has_new ? new_info->duration_ : -1.0;
        const int new_id = has_new ? new_info->traj_id_ : -1;
        const int finish_before_old_stop = old_stop_deadline < 0.0 ||
                                           plan_end.toSec() <= old_stop_deadline ? 1 : 0;
        fd_->handoff_log_file_
            << ++fd_->handoff_log_round_ << ",plan_complete," << plan_begin.toSec() << ","
            << plan_end.toSec() << "," << (plan_end - plan_begin).toSec() << ","
            << fp_->replan_time_ << "," << old_traj_id << "," << old_start << ","
            << old_duration << "," << old_remaining << "," << old_stop_deadline << ","
            << fd_->start_pt_.x() << "," << fd_->start_pt_.y() << "," << fd_->start_pt_.z()
            << "," << fd_->start_vel_.norm() << "," << fd_->start_yaw_(0) << ","
            << res << "," << new_id << "," << new_start << "," << new_duration << ","
            << finish_before_old_stop << "\n";
        fd_->handoff_log_file_.flush();
      }
      if (res == SUCCEED) {
        fd_->no_frontier_rounds_ = 0;
        fd_->fail_rounds_ = 0;
        transitState(PUB_TRAJ, "FSM");
      } else if (res == NO_FRONTIER) {
        ROS_WARN("[FSM] Keep active trajectory: planning returned NO_FRONTIER");
        ++fd_->no_frontier_rounds_;
        const int confirm_rounds = std::max(1, fp_->no_frontier_confirm_rounds_);
        if (fd_->no_frontier_rounds_ >= confirm_rounds) {
          transitState(FINISH, "FSM");
          fd_->static_state_ = true;
          clearVisMarker();
        } else {
          ROS_WARN("[FSM] NO_FRONTIER %d/%d, keep replanning before FINISH",
                   fd_->no_frontier_rounds_, confirm_rounds);
          fd_->static_state_ = true;
        }
      } else if (res == FAIL) {
        // Transient local-planner failure must not terminate exploration.
        // The active trajectory was already preempted above, so retry from the
        // latest odometry state until a successor is produced.
        ROS_WARN("plan fail, keep replanning");
        fd_->static_state_ = true;
      }
      break;
    }

    case PUB_TRAJ: {
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {
        bspline_pub_.publish(fd_->newest_traj_);
        // [djg] 发布累积轨迹
        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "world";
        pose.header.stamp = ros::Time::now();
        pose.pose.position.x = fd_->start_pt_.x();
        pose.pose.position.y = fd_->start_pt_.y();
        pose.pose.position.z = fd_->start_pt_.z();
        pose.pose.orientation.w = 1.0;
        traj_path_.poses.push_back(pose);
        traj_path_.header.stamp = ros::Time::now();
        traj_path_pub_.publish(traj_path_);
        fd_->static_state_ = false;
        transitState(EXEC_TRAJ, "FSM");

        thread vis_thread(&FastExplorationFSM::visualize, this);
        vis_thread.detach();
      }
      break;
    }

    case EXEC_TRAJ: {
      LocalTrajData* info = &planner_manager_->local_data_;
      double t_cur = (ros::Time::now() - info->start_time_).toSec();

      // Replan if traj is almost fully executed
      double time_to_end = info->duration_ - t_cur;
      if (time_to_end < fp_->replan_thresh1_) {
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN("Replan: traj fully executed=================================");
        return;
      }
      // Keep FUEL's original online handoff: while the current trajectory is
      // still being executed, map updates that substantially change a
      // frontier immediately start planning the successor trajectory from the
      // predicted state at now + replan_time.  This is deliberately not a
      // "wait until the selected frontier is finished" condition.
      if (t_cur > fp_->replan_thresh2_ && expl_manager_->frontier_finder_->isFrontierCovered()) {
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN("Replan: cluster covered=====================================");
        return;
      }
      // Replan after some time
      if (t_cur > fp_->replan_thresh3_ && !classic_) {
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN("Replan: periodic call=======================================");
      }
      break;
    }
  }
}

int FastExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);

  int res = expl_manager_->planExploreMotion(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_,
                                             fd_->start_yaw_);
  classic_ = false;

  // int res = expl_manager_->classicFrontier(fd_->start_pt_, fd_->start_yaw_[0]);
  // classic_ = true;

  // int res = expl_manager_->rapidFrontier(fd_->start_pt_, fd_->start_vel_, fd_->start_yaw_[0],
  // classic_);

  if (res == SUCCEED) {
    auto info = &planner_manager_->local_data_;
    info->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;

    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;
    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }
    Eigen::VectorXd knots = info->position_traj_.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
    for (int i = 0; i < yaw_pts.rows(); ++i) {
      double yaw = yaw_pts(i, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = info->yaw_traj_.getKnotSpan();
    fd_->newest_traj_ = bspline;
  }
  return res;
}

void FastExplorationFSM::visualize() {
  auto info = &planner_manager_->local_data_;
  auto plan_data = &planner_manager_->plan_data_;
  auto ed_ptr = expl_manager_->ed_;

  // RACER-style: pass active grid ID to GridPlanner for highlighting
  expl_manager_->grid_planner_->setActiveGridIdForVis(expl_manager_->getActiveGridId());

  // Set drone's current physical grid for visualization (white border)
  expl_manager_->grid_planner_->setDroneGridIdForVis(
      expl_manager_->grid_planner_->getGridContainingPoint(fd_->odom_pos_));

  // Set selected frontier position for visualization (yellow sphere)
  if (!ed_ptr->refined_points_.empty())
    expl_manager_->grid_planner_->setSelectedFrontierForVis(ed_ptr->refined_points_[0]);
  else if (ed_ptr->points_.size() > 0)
    expl_manager_->grid_planner_->setSelectedFrontierForVis(ed_ptr->points_[0]);

  // Draw grid cells with active grid highlight (green border)
  expl_manager_->grid_planner_->publishGridMarkers();

  // Draw global grid tour with A* collision-free paths (not direct lines)
  if (!ed_ptr->grid_tour_.empty()) {
    expl_manager_->grid_planner_->publishGlobalTour(
        ed_ptr->grid_tour_, ed_ptr->grid_tour_waypoints_,
        ed_ptr->grid_tour_ids_, ed_ptr->grid_ids_);
  }

  // Draw each frontier cluster with the original FUEL color palette.  Component
  // locks are no longer used by planning, so coloring by component state makes
  // every cluster fall back to the same orange marker.
  static int last_ftr_num = 0;
  for (int i = 0; i < ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(
        ed_ptr->frontiers_[i], 0.1,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 0.8),
        "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr_num; ++i) {
    visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr_num = ed_ptr->frontiers_.size();

  // Draw local frontier tour (A* path between frontiers within active grid)
  if (!ed_ptr->global_tour_.empty()) {
    visualization_->drawLines(ed_ptr->global_tour_, 0.08,
                              Vector4d(0.0, 0.9, 0.2, 0.9),
                              "local_frontier_tour", 0, 6);
  }

  // Draw local refined viewpoints (yellow/gold)
  visualization_->drawSpheres(ed_ptr->refined_points_, 0.25, Vector4d(1.0, 0.85, 0.1, 0.9), "refined_pts", 0, 6);
  visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->refined_views_, 0.04,
                            Vector4d(1.0, 0.7, 0.1, 0.6), "refined_view", 0, 6);
  visualization_->drawLines(ed_ptr->refined_tour_, 0.06, Vector4d(1.0, 0.75, 0.15, 0.8), "refined_tour", 0, 6);
  visualization_->drawLines(ed_ptr->refined_views1_, ed_ptr->refined_views2_, 0.03,
                            Vector4d(0.9, 0.6, 0.1, 0.5), "refined_fov", 0, 6);
  // for (int i = 0; i < ed_ptr->n_points_.size(); ++i)
  //   visualization_->drawSpheres(ed_ptr->n_points_[i], 0.1,
  //                               visualization_->getColor(double(ed_ptr->refined_ids_[i]) /
  //                               ed_ptr->frontiers_.size()),
  //                               "n_points", i, 6);
  // for (int i = ed_ptr->n_points_.size(); i < 15; ++i)
  //   visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);

  // Draw trajectory
  // visualization_->drawSpheres({ ed_ptr->next_goal_ }, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
  visualization_->drawBspline(info->position_traj_, 0.1, Vector4d(1.0, 0.0, 0.0, 1), false, 0.15,
                              Vector4d(1, 1, 0, 1));
  // visualization_->drawSpheres(plan_data->kino_path_, 0.1, Vector4d(1, 0, 1, 1), "kino_path", 0, 0);
  // visualization_->drawLines(ed_ptr->path_next_goal_, 0.05, Vector4d(0, 1, 1, 1), "next_goal", 1, 6);
}

void FastExplorationFSM::clearVisMarker() {
  visualization_->drawSpheres({}, 0.25, Vector4d(1, 0.85, 0.1, 0.9), "refined_pts", 0, 6);
  visualization_->drawLines({}, {}, 0.04, Vector4d(1, 0.7, 0.1, 0.6), "refined_view", 0, 6);
  visualization_->drawLines({}, 0.06, Vector4d(1, 0.75, 0.15, 0.8), "refined_tour", 0, 6);
  visualization_->drawLines({}, {}, 0.03, Vector4d(0.9, 0.6, 0.1, 0.5), "refined_fov", 0, 6);
}

void FastExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  static int delay = 0;
  if (++delay < 5) return;

  if (state_ == WAIT_TRIGGER || state_ == FINISH) {
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;
    ft->searchFrontiers();
    ft->computeFrontiersToVisit();
    ft->updateFrontierCostMatrix();

    ft->getFrontiers(ed->frontiers_);
    ft->getFrontierBoxes(ed->frontier_boxes_);

    // Draw frontier and bounding box
    for (int i = 0; i < ed->frontiers_.size(); ++i) {
      visualization_->drawCubes(ed->frontiers_[i], 0.1,
                                visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4),
                                "frontier", i, 4);
      // visualization_->drawBox(ed->frontier_boxes_[i].first, ed->frontier_boxes_[i].second,
      // Vector4d(0.5, 0, 1, 0.3),
      //                         "frontier_boxes", i, 4);
    }
    for (int i = ed->frontiers_.size(); i < 50; ++i) {
      visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
      // "frontier_boxes", i, 4);
    }
  }

  // if (!fd_->static_state_)
  // {
  //   static double astar_time = 0.0;
  //   static int astar_num = 0;
  //   auto t1 = ros::Time::now();

  //   planner_manager_->path_finder_->reset();
  //   planner_manager_->path_finder_->setResolution(0.4);
  //   if (planner_manager_->path_finder_->search(fd_->odom_pos_, Vector3d(-5, 0, 1)))
  //   {
  //     auto path = planner_manager_->path_finder_->getPath();
  //     visualization_->drawLines(path, 0.05, Vector4d(1, 0, 0, 1), "astar", 0, 6);
  //     auto visit = planner_manager_->path_finder_->getVisited();
  //     visualization_->drawCubes(visit, 0.3, Vector4d(0, 0, 1, 0.4), "astar-visit", 0, 6);
  //   }
  //   astar_num += 1;
  //   astar_time = (ros::Time::now() - t1).toSec();
  //   ROS_WARN("Average astar time: %lf", astar_time);
  // }
}

void FastExplorationFSM::triggerCallback(const nav_msgs::PathConstPtr& msg) {
  if (msg->poses[0].pose.position.z < -0.1) return;
  if (state_ != WAIT_TRIGGER) return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;

  // Initialize exploration stats logging
  resetExplorationStats();
  openStatsLog();

  transitState(PLAN_TRAJ, "triggerCallback");
}

void FastExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ) {
    // Check safety and trigger replan if necessary
    double dist;
    bool safe = planner_manager_->checkTrajCollision(dist);
    if (!safe) {
      ROS_WARN("Replan: collision detected==================================");
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;

  // Accumulate trajectory length during exploration
  if (fd_->expl_active_ && fd_->has_last_odom_) {
    double delta = (fd_->odom_pos_ - fd_->last_odom_pos_).norm();
    if (delta < 2.0) {  // reject teleportation / large jumps
      fd_->total_traj_length_ += delta;
    }
  }
  fd_->last_odom_pos_ = fd_->odom_pos_;
  fd_->has_last_odom_ = true;
}

void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[" + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]
       << endl;
}
}  // namespace fast_planner
