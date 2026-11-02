#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <vector>
#include <fstream>
#include <string>
#include <bspline/Bspline.h>

using std::vector;
using std::string;
using Eigen::Vector3d;

namespace fast_planner {
struct FSMData {
  // FSM data
  bool trigger_, have_odom_, static_state_;
  int no_frontier_rounds_ = 0;
  int fail_rounds_ = 0;
  vector<string> state_str_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_;  // start state
  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;

  // Exploration stats logging (10-second periodic)
  bool enable_stats_log_ = true;
  string stats_log_path_ = "/home/djg/Data_debug/exploration_stats.csv";
  ros::Time expl_start_time_;
  bool expl_active_ = false;
  double total_traj_length_ = 0.0;
  Eigen::Vector3d last_odom_pos_ = Eigen::Vector3d::Zero();
  bool has_last_odom_ = false;
  ros::Time last_stats_log_time_;
  std::ofstream stats_log_file_;
  // [MODIFIED BY CODEX]: Low-rate planner-side trajectory handoff diagnostics.
  bool enable_handoff_log_ = true;
  string handoff_log_path_ = "/home/djg/Data_debug/fsm_handoff_log.csv";
  std::ofstream handoff_log_file_;
  int handoff_log_round_ = 0;
};

struct FSMParam {
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_time_;  // second
  int no_frontier_confirm_rounds_ = 3;
};

struct ExplorationData {
  vector<vector<Vector3d>> frontiers_;
  vector<vector<Vector3d>> dead_frontiers_;
  vector<pair<Vector3d, Vector3d>> frontier_boxes_;
  vector<Vector3d> points_;
  vector<Vector3d> averages_;
  vector<Vector3d> views_;
  vector<double> yaws_;
  vector<Vector3d> global_tour_;

  // [MODIFIED BY CODEX]: Complete local LKH order is separate from the
  // Euclidean short-range viewpoint-refinement prefix.  refined_exit_id_ is
  // the first successor outside that prefix and enters Dijkstra as a fixed
  // representative viewpoint; it is look-ahead only, never a queued command.
  vector<int> local_tsp_ids_;
  vector<int> refined_ids_;
  int refined_exit_id_ = -1;
  vector<vector<Vector3d>> n_points_;
  vector<Vector3d> unrefined_points_;
  vector<Vector3d> refined_points_;
  vector<Vector3d> refined_views_;  // points + dir(yaw)
  vector<Vector3d> refined_views1_, refined_views2_;
  vector<Vector3d> refined_tour_;

  Vector3d next_goal_;
  vector<Vector3d> path_next_goal_;

  // viewpoint planning
  // vector<Vector4d> views_;
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;

  // Grid-based global guidance
  vector<int> grid_ids_;               // active grid IDs
  vector<Vector3d> grid_centers_;      // active grid centers
  vector<int> grid_tour_ids_;          // grid IDs in TSP order
  vector<Vector3d> grid_tour_;         // centers in TSP order
  vector<Vector3d> grid_tour_waypoints_; // grid visit waypoints in TSP order
  vector<int> grid_frontier_ids_;      // frontier IDs belonging to next grids in tour
  // Frontier component tracking: maps frontier ID -> true if in the currently active
  // same-side component under the grid-component gate.
  std::unordered_map<int, bool> frontier_component_active_;
};

struct ExplorationParam {
  // params
  bool refine_local_;
  int refined_num_;
  double refined_radius_;
  int top_view_num_;
  double max_decay_;
  string tsp_dir_;  // resource dir of tsp solver
  double relax_time_;

  // Grid switching policy:
  // select grid by frontier motion cost in a short lookahead window
  int grid_switch_lookahead_ = 4;
  double grid_switch_order_penalty_ = 2.0;
  double grid_switch_cost_margin_ = 0.8;
  double grid_switch_unreachable_penalty_ = 8.0;
  double grid_switch_max_path_length_ = 14.0;
  double grid_switch_max_path_stretch_ = 2.4;
  int grid_neighbor_hops_ = 2;
  int grid_target_hold_rounds_ = 4;
  double grid_target_switch_margin_ = 1.5;
  double grid_pick_w_info_ = 3.0;
  double grid_pick_w_motion_ = 1.0;
  double grid_pick_w_neighbor_ = 3.0;
  double local_unfinished_grid_priority_margin_ = 3.0;
  double local_unfinished_grid_max_path_ = 7.0;
  int recent_path_grid_limit_ = 8;
  double fine_grid_switch_margin_ = 2.0;
  double current_grid_priority_max_path_ = 8.0;
  double current_grid_priority_margin_ = 6.0;
  double current_grid_priority_path_margin_ = 2.0;
  double root_coverage_priority_weight_ = 0.8;
  double local_neighbor_frontier_max_path_ = 10.0;
  double local_reobserve_radius_ = 6.5;
  double local_reobserve_max_path_ = 9.0;
  double local_recent_grid_reobserve_max_path_ = 11.0;
  // Time-equivalent reward for a verified bounded residual region.
  // [MODIFIED BY CODEX]: 恢复独立小区域(口袋)收尾奖励权重为 6.0
  double local_pocket_weight_ = 6.0;
  double local_view_uncovered_overlap_thresh_ = 0.35;
  double local_view_covered_overlap_thresh_ = 0.35;
  bool local_view_block_cross_grid_ = false;
  bool root_hard_lock_ = false;
  int root_hard_lock_max_unreachable_rounds_ = 6;
  int root_complete_confirm_rounds_ = 3;
  bool enable_root_soft_release_ = false;
  double root_soft_release_coverage_ratio_ = 0.90;
  int root_soft_release_min_rounds_ = 12;
  int root_soft_release_cooldown_rounds_ = 25;
  int lead_frontier_stall_rounds_ = 4;
  double lead_frontier_stall_dist_ = 1.2;
  int residual_defer_pass_cnt_ = 10;
  double residual_defer_max_coverage_ = 0.08;
  double residual_defer_max_path_ = 7.5;

  // Coarse-grid stickiness: prevent fine-grid oscillation within same root
  double same_root_fine_grid_margin_ = 3.5;
  double same_root_backtrack_penalty_ = 7.0;
  double fine_grid_cov_complete_thresh_ = 0.88;
  double coarse_dir_momentum_weight_ = 4.0;
};

}  // namespace fast_planner

#endif
