#include <active_perception/frontier_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
// #include <path_searching/astar2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <sstream>
#include <iomanip>
#include <pcl/filters/voxel_grid.h>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>


namespace fast_planner {
FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = vector<char>(voxel_num, 0);
  fill(frontier_flag_.begin(), frontier_flag_.end(), 0);
  coverage_flag_ = vector<char>(voxel_num, 0);
  fill(coverage_flag_.begin(), coverage_flag_.end(), 0);
  coverage_ever_flag_ = vector<char>(voxel_num, 0);
  fill(coverage_ever_flag_.begin(), coverage_ever_flag_.end(), 0);
  coverage_last_round_.assign(voxel_num, -1000000);
  coverage_round_ = 0;

  nh.param("frontier/cluster_min", cluster_min_, -1);
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);
  nh.param("frontier/cluster_size_z", cluster_size_z_, -1.0);
  nh.param("frontier/min_candidate_dist", min_candidate_dist_, -1.0);
  nh.param("frontier/min_candidate_clearance", min_candidate_clearance_, -1.0);
  nh.param("frontier/candidate_dphi", candidate_dphi_, -1.0);
  nh.param("frontier/candidate_rmax", candidate_rmax_, -1.0);
  nh.param("frontier/candidate_rmin", candidate_rmin_, -1.0);
  nh.param("frontier/candidate_rnum", candidate_rnum_, -1);
  nh.param("frontier/down_sample", down_sample_, -1);
  nh.param("frontier/min_visib_num", min_visib_num_, -1);
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);
  ///////////////////////////////////////////////////////////////////////////////////
  nh.param("frontier/feature_radius", feature_radius_, 1.0);
  nh.param("frontier/feature_dir_num", feature_dir_num_, 16);
  ///////////////////////////////////////////////////////////////////////////////////

  regret_lambda1_ = 0.8;  // 凹陷区域权重，固定在代码内
  regret_lambda2_ = 0.2;  // 狭窄通道权重，固定在代码内

  // [MODIFIED BY CODEX]: 结构评价主开关（消融实验用），XML 中 frontier/enable_structure_eval 可切换
  nh.param("frontier/enable_structure_eval", enable_structure_eval_, true);
  // [ABLATION]: 射线项权重 alpha，默认 0.7 保持与改动前硬编码 0.70/0.30 完全一致
  nh.param("frontier/alpha", structure_alpha_, 0.7);

  // ==================== 局部 coverage + structure closeout 参数 ====================
  nh.param("frontier/purge_min_cells", purge_min_cells_, 10);
  nh.param("frontier/forget_coverage_thresh", forget_coverage_thresh_, 0.06);

  nh.param("frontier/local_closeout_radius", local_closeout_radius_, 4.5);
  nh.param("frontier/force_local_sweep", force_local_sweep_, true);
  nh.param("frontier/local_sweep_min_frontiers", local_sweep_min_frontiers_, 1);
  nh.param("frontier/local_sweep_max_frontiers", local_sweep_max_frontiers_, 2);
  nh.param("frontier/local_sweep_hold_rounds", local_sweep_hold_rounds_, 0);
  nh.param("frontier/local_sweep_max_path_length", local_sweep_max_path_length_, 6.0);
  nh.param("frontier/local_sweep_max_path_stretch", local_sweep_max_path_stretch_, 1.8);
  nh.param("frontier/local_sweep_max_yaw_jump", local_sweep_max_yaw_jump_, 1.75);
  nh.param("frontier/local_sweep_nonlocal_penalty", local_sweep_nonlocal_penalty_, 0.0);
  nh.param("frontier/local_sweep_bonus_max", local_sweep_bonus_max_, 0.8);
  nh.param("frontier/force_local_cover_all", force_local_cover_all_, false);
  nh.param("frontier/local_corner_boost", local_corner_boost_, 0.18);
  nh.param("frontier/local_pocket_boost", local_pocket_boost_, 0.10);
  nh.param("frontier/local_target_hold_rounds", local_target_hold_rounds_, 3);
  nh.param("frontier/local_switch_margin", local_switch_margin_, 0.24);
  nh.param("frontier/local_repeat_path_penalty", local_repeat_path_penalty_, 0.22);
  nh.param("frontier/local_repeat_pass_penalty", local_repeat_pass_penalty_, 0.10);
  nh.param("frontier/local_strong_hold_rounds", local_strong_hold_rounds_, 6);
  nh.param("frontier/local_strong_switch_margin", local_strong_switch_margin_, 0.30);
  nh.param("frontier/enable_local_branch_lock", enable_local_branch_lock_, true);
  nh.param("frontier/local_branch_entry_gap", local_branch_entry_gap_, 1.4);
  nh.param("frontier/local_branch_hold_rounds", local_branch_hold_rounds_, 4);
  nh.param("frontier/local_branch_switch_margin", local_branch_switch_margin_, 0.22);
  nh.param("frontier/local_min_wall_clearance", local_min_wall_clearance_, 0.45);
  nh.param("frontier/enable_global_chain_bias", enable_global_chain_bias_, true);
  nh.param("frontier/global_chain_bonus_gain", global_chain_bonus_gain_, 1.2);
  nh.param("frontier/global_chain_gap_gate", global_chain_gap_gate_, 2.0);
  nh.param("frontier/global_target_hold_rounds", global_target_hold_rounds_, 6);
  nh.param("frontier/global_switch_margin", global_switch_margin_, 1.6);
  nh.param("frontier/global_pass_pocket_bonus_gain", global_pass_pocket_bonus_gain_, 2.4);
  nh.param("frontier/global_pass_pocket_gap_gate", global_pass_pocket_gap_gate_, 9.0);
  nh.param("frontier/global_pass_pocket_near_radius", global_pass_pocket_near_radius_, 9.0);
  nh.param("frontier/global_pass_pocket_max_overlap", global_pass_pocket_max_overlap_, 0.90);
  nh.param("frontier/global_pass_pocket_min_debt", global_pass_pocket_min_debt_, 0.22);
  nh.param("frontier/global_pass_pocket_min_bounded", global_pass_pocket_min_bounded_, 0.35);

  // ========= 参数收口：只保留少量“强度参数”，其余使用稳定缺省，便于收敛实验 =========
  double global_boundary_strength = 1.0;
  double global_local_completion_strength = 1.0;
  nh.param("frontier/enable_global_boundary_bias", enable_global_boundary_bias_, true);
  nh.param("frontier/global_boundary_strength", global_boundary_strength, 1.0);
  nh.param("frontier/global_boundary_band", global_boundary_band_, 4.0);
  nh.param("frontier/enable_global_local_completion_gate", enable_global_local_completion_gate_, true);
  nh.param("frontier/global_local_completion_strength", global_local_completion_strength, 1.0);
  nh.param("frontier/global_local_completion_radius", global_local_completion_radius_, 5.5);

  // 边界引导固定策略
  global_boundary_bonus_gain_ = 4.0 * global_boundary_strength;
  global_boundary_gap_gate_ = 18.0;
  global_boundary_max_overlap_ = 0.92;
  global_boundary_min_score_ = 0.35;
  global_boundary_side_hold_rounds_ = 10;
  global_boundary_side_switch_margin_ = 0.45;
  global_boundary_cross_penalty_ = 0.8 * global_boundary_strength;
  global_boundary_nonboundary_penalty_ = 0.0;
  global_boundary_same_side_bonus_ = 0.4 * global_boundary_strength;

  // 局部清空固定策略
  global_local_completion_gap_gate_ = 8.0;
  global_local_completion_overlap_max_ = 0.35;
  global_local_completion_min_count_ = 2;
  global_local_completion_max_cells_ = 450;
  global_local_completion_allow_single_ = true;
  global_local_completion_single_score_gate_ = 0.62;
  global_local_completion_single_pass_gate_ = 2;
  global_local_completion_bonus_ = 1.6 * global_local_completion_strength;
  global_local_completion_penalty_ = 1.4 * global_local_completion_strength;
  nh.param("frontier/cooldown_rounds", cooldown_rounds_, 12);
  nh.param("frontier/cooldown_progress_thresh", cooldown_progress_thresh_, 0.02);
  nh.param("frontier/closeout_min_view_dist", closeout_min_view_dist_, 0.8);
  nh.param("frontier/adaptive_voxel_thresh", adaptive_voxel_thresh_, 0.45);

  // 路径膨胀覆盖参数
  nh.param("frontier/inflate_radius", inflate_radius_, 4.5);
  nh.param("frontier/coverage_memory_rounds", coverage_memory_rounds_, 15);
  nh.param("frontier/coverage_mark_only_new_segment", coverage_mark_only_new_segment_, true);

  nh.param("frontier/enable_frontier_track_log", enable_frontier_track_log_, false);
  nh.param("frontier/frontier_track_log_path", frontier_track_log_path_,
          std::string("/home/djg/Data_debug/frontier_track_log.csv"));

  nh.param("frontier/enable_local_view_log", enable_local_view_log_, false);
  nh.param("frontier/local_view_log_path", local_view_log_path_,
          std::string("/home/djg/Data_debug/local_view_log.csv"));

  frontier_track_round_ = 0;

  // ============================================================

  // ==================== 稳定前沿跟踪日志 ====================

  if (enable_frontier_track_log_) {
    frontier_track_log_file_.open(frontier_track_log_path_, std::ios::out);
    if (frontier_track_log_file_.is_open()) {
        frontier_track_log_file_
          << "round,stamp,frontier_id,sig,"
          << "avg_x,avg_y,avg_z,"
          << "box_min_x,box_min_y,box_min_z,"
          << "box_max_x,box_max_y,box_max_z,"
          << "cell_num,filtered_cell_num,dist_to_uav,in_local_band,"
          << "persist_cnt,first_seen_round,last_seen_round,"
          << "forget_flag,stale_fraction,coverage_overlap,forget_reason,"
          << "persistent_coverage_overlap,closeout_overlap,"
          << "concavity,narrowness,top_visib,filtered_coverage_overlap,"
          << "debt_score,progress_delta,local_pass_cnt,attempt_cnt,cooldown_until_round,"
          << "path_overlap,last_start_path_size,last_start_path_length,last_start_raw_cost,last_start_final_cost,"
          << "min_pass_dist,near_pass_cnt,last_near_round,pass_debt_score,"
          << "back_unknown_num,back_obstacle_ratio,back_leak_ratio,"
          << "bounded_pocket_score,bounded_pocket_flag,"
          << "adaptive_voxel_score,invalid_closeout,tsp_layer,closeout_type,"
          << "frontier_stage,root_lock,root_unreachable_rounds,"
          << "active_grid_id,active_root_grid_id\n";
      frontier_track_log_file_.flush();
      ROS_WARN_STREAM("[FrontierTrackLog] Logging to " << frontier_track_log_path_);
    } else {
      ROS_ERROR_STREAM("[FrontierTrackLog] Failed to open file: " << frontier_track_log_path_);
    }
  }

  if (enable_local_view_log_) {
    local_view_log_file_.open(local_view_log_path_, std::ios::out);
    if (local_view_log_file_.is_open()) {
        local_view_log_file_
          << "round,stamp,frontier_id,sig,local_order,"
          << "candidate_num,selected_num,best_visib,best_view_dist,best_unknown,"
          << "dist_to_uav,cell_num,coverage_overlap,closeout_overlap,"
          << "forget_flag,stale_fraction,debt_score,progress_delta,"
          << "local_pass_cnt,attempt_cnt,cooldown_until_round,"
          << "in_local_band,min_pass_dist,near_pass_cnt,last_near_round,"
          << "adaptive_voxel_score,invalid_closeout,closeout_type,"
          << "frontier_stage,root_lock,root_unreachable_rounds,"
          << "active_grid_id,active_root_grid_id,"
          << "lkh_sequence,start_astar_length,dijkstra_prefix_order,"
          << "in_dijkstra,dijkstra_reason\n";
      local_view_log_file_.flush();
      ROS_WARN_STREAM("[LocalViewLog] Logging to " << local_view_log_path_);
    } else {
      ROS_ERROR_STREAM("[LocalViewLog] Failed to open file: " << local_view_log_path_);
    }
  }
  // ================================================================



  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils(nh));


  // 遗忘前沿可视化
  forget_vis_pub_ = nh.advertise<visualization_msgs::MarkerArray>(
      "/planning_vis/forget_frontiers", 10);
  local_order_vis_pub_ = nh.advertise<visualization_msgs::MarkerArray>(
      "/planning_vis/local_explore_order", 10);

}

FrontierFinder::~FrontierFinder() {
  if (frontier_track_log_file_.is_open()) {
    frontier_track_log_file_.close();
  }

  if (local_view_log_file_.is_open()) {
    local_view_log_file_.close();
  }
}

void FrontierFinder::setGridPlanningLogContext(
    int frontier_stage, bool root_lock, int root_unreachable_rounds,
    int active_grid_id, int active_root_grid_id) {
  log_frontier_stage_ = frontier_stage;
  log_root_lock_ = root_lock;
  log_root_unreachable_rounds_ = root_unreachable_rounds;
  log_active_grid_id_ = active_grid_id;
  log_active_root_grid_id_ = active_root_grid_id;
}

void FrontierFinder::setLocalViewRefinementLogContext(
    const vector<int>& lkh_ids, const std::unordered_map<int, double>& start_path_lengths,
    const std::unordered_map<int, int>& dijkstra_orders,
    const std::unordered_map<int, std::string>& dijkstra_reasons) {
  local_view_lkh_sequence_.clear();
  for (size_t i = 0; i < lkh_ids.size(); ++i) {
    if (i > 0) local_view_lkh_sequence_ += ";";
    local_view_lkh_sequence_ += std::to_string(lkh_ids[i]);
  }
  local_view_start_path_lengths_ = start_path_lengths;
  local_view_dijkstra_orders_ = dijkstra_orders;
  local_view_dijkstra_reasons_ = dijkstra_reasons;
}

void FrontierFinder::searchFrontiers() {
  ros::Time t1 = ros::Time::now();
  tmp_frontiers_.clear();

  // Bounding box of updated region
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max, true);

  // Removed changed frontiers in updated map
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers.erase(iter);
  };

  std::cout << "Before remove: " << frontiers_.size() << std::endl;

  removed_ids_.clear();
  int rmv_idx = 0;
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter)) {
      // Keep the index in the matrix before any row is removed.  The cached
      // columns are erased in descending order in updateFrontierCostMatrix().
      removed_ids_.push_back(rmv_idx);
      resetFlag(iter, frontiers_);
    } else {
      ++iter;
    }
    ++rmv_idx;
  }
  std::cout << "After remove: " << frontiers_.size() << std::endl;
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }

  // Search new frontier within box slightly inflated from updated box
  Vector3d search_min = update_min - Vector3d(1, 1, 0.5);
  Vector3d search_max = update_max + Vector3d(1, 1, 0.5);
  Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);
  for (int k = 0; k < 3; ++k) {
    search_min[k] = max(search_min[k], box_min[k]);
    search_max[k] = min(search_max[k], box_max[k]);
  }
  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);
  edt_env_->sdf_map_->posToIndex(search_max, max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        // Scanning the updated region to find seeds of frontiers
        Eigen::Vector3i cur(x, y, z);
        if (frontier_flag_[toadr(cur)] == 0 && knownfree(cur) && isNeighborUnknown(cur)) {
          // Expand from the seed cell to find a complete frontier cluster
          expandFrontier(cur);
        }
      }
  splitLargeFrontiers(tmp_frontiers_);

  ROS_WARN_THROTTLE(5.0, "Frontier t: %lf", (ros::Time::now() - t1).toSec());
}

void FrontierFinder::expandFrontier(
    const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */) {
  // std::cout << "depth: " << depth << std::endl;
  auto t1 = ros::Time::now();

  // Data for clustering
  queue<Eigen::Vector3i> cell_queue;
  vector<Eigen::Vector3d> expanded;
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toadr(first)] = 1;

  // Search frontier cluster based on region growing (distance clustering)
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (auto nbr : nbrs) {
      // Qualified cell should be inside bounding box and frontier cell not clustered
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) ||
          !(knownfree(nbr) && isNeighborUnknown(nbr)))
        continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      if (pos[2] < 0.4) continue;  // Remove noise close to ground
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }
  if (expanded.size() > cluster_min_) {
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers) {
  list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // Check if each frontier needs to be split horizontally
    if (splitHorizontally(*it, splits)) {
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);
  }
  frontiers = tmps;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<2>();
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_pc = vectors.col(max_idx);
  std::cout << "max idx: " << max_idx << std::endl;
  std::cout << "mean: " << mean.transpose() << ", first pc: " << first_pc.transpose() << std::endl;

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::isInBoxes(
    const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx) {
  Vector3d pt;
  edt_env_->sdf_map_->indexToPos(idx, pt);
  for (auto box : boxes) {
    // Check if contained by a box
    bool inbox = true;
    for (int i = 0; i < 3; ++i) {
      inbox = inbox && pt[i] > box.first[i] && pt[i] < box.second[i];
      if (!inbox) break;
    }
    if (inbox) return true;
  }
  return false;
}

void FrontierFinder::updateFrontierCostMatrix() {
  auto compute_pair = [](const list<Frontier>::iterator &from,
                         const list<Frontier>::iterator &to,
                         double &cost, vector<Vector3d> &path) {
    if (from->viewpoints_.empty() || to->viewpoints_.empty()) {
      cost = 1000.0;
      path.clear();
      return;
    }
    const Viewpoint &vf = from->viewpoints_.front();
    const Viewpoint &vt = to->viewpoints_.front();
    cost = ViewNode::computeCost(vf.pos_, vt.pos_, vf.yaw_, vt.yaw_,
                                 Vector3d::Zero(), 0.0, path);
    if (!std::isfinite(cost)) {
      cost = 1000.0;
      path.clear();
    }
  };

  auto rebuild_all = [&]() {
    vector<list<Frontier>::iterator> ftrs;
    ftrs.reserve(frontiers_.size());
    for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it) {
      it->costs_.clear();
      it->paths_.clear();
      ftrs.push_back(it);
    }

    const int n = static_cast<int>(ftrs.size());
    vector<vector<double>> costs(n, vector<double>(n, 0.0));
    vector<vector<vector<Vector3d>>> paths(n, vector<vector<Vector3d>>(n));
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        compute_pair(ftrs[i], ftrs[j], costs[i][j], paths[i][j]);
        costs[j][i] = costs[i][j];
        paths[j][i] = paths[i][j];
        std::reverse(paths[j][i].begin(), paths[j][i].end());
      }
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        ftrs[i]->costs_.push_back(costs[i][j]);
        ftrs[i]->paths_.push_back(paths[i][j]);
      }
    }
    removed_ids_.clear();
    first_new_ftr_ = frontiers_.end();
    ROS_WARN("[FrontierCost] cache_rebuild: %d x %d", n, n);
  };

  const int total_num = static_cast<int>(frontiers_.size());
  const int new_num = static_cast<int>(std::distance(first_new_ftr_, frontiers_.end()));
  const int old_num = total_num - new_num;
  const int previous_num = old_num + static_cast<int>(removed_ids_.size());

  bool cache_valid = old_num >= 0;
  int checked_old = 0;
  for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it, ++checked_old) {
    cache_valid = cache_valid &&
                  static_cast<int>(it->costs_.size()) == previous_num &&
                  static_cast<int>(it->paths_.size()) == previous_num;
  }
  cache_valid = cache_valid && checked_old == old_num;
  for (int id : removed_ids_)
    cache_valid = cache_valid && id >= 0 && id < previous_num;

  if (!cache_valid) {
    ROS_WARN("[FrontierCost] incremental cache mismatch, using safe rebuild");
    rebuild_all();
    return;
  }

  std::sort(removed_ids_.begin(), removed_ids_.end(), std::greater<int>());
  removed_ids_.erase(std::unique(removed_ids_.begin(), removed_ids_.end()),
                     removed_ids_.end());
  for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it) {
    for (int id : removed_ids_) {
      auto cost_it = it->costs_.begin();
      auto path_it = it->paths_.begin();
      std::advance(cost_it, id);
      std::advance(path_it, id);
      it->costs_.erase(cost_it);
      it->paths_.erase(path_it);
    }
  }

  // Append only old-new and new-new entries.  Unchanged old-old entries stay
  // in the incremental frontier information structure.
  for (auto old_it = frontiers_.begin(); old_it != first_new_ftr_; ++old_it) {
    for (auto new_it = first_new_ftr_; new_it != frontiers_.end(); ++new_it) {
      double cost = 1000.0;
      vector<Vector3d> forward_path;
      compute_pair(old_it, new_it, cost, forward_path);
      old_it->costs_.push_back(cost);
      old_it->paths_.push_back(forward_path);
      std::reverse(forward_path.begin(), forward_path.end());
      new_it->costs_.push_back(cost);
      new_it->paths_.push_back(forward_path);
    }
  }
  for (auto from = first_new_ftr_; from != frontiers_.end(); ++from) {
    for (auto to = from; to != frontiers_.end(); ++to) {
      if (from == to) {
        from->costs_.push_back(0.0);
        from->paths_.push_back({});
        continue;
      }
      double cost = 1000.0;
      vector<Vector3d> forward_path;
      compute_pair(from, to, cost, forward_path);
      from->costs_.push_back(cost);
      from->paths_.push_back(forward_path);
      std::reverse(forward_path.begin(), forward_path.end());
      to->costs_.push_back(cost);
      to->paths_.push_back(forward_path);
    }
  }

  for (const auto &frontier : frontiers_) {
    if (static_cast<int>(frontier.costs_.size()) != total_num ||
        static_cast<int>(frontier.paths_.size()) != total_num) {
      ROS_WARN("[FrontierCost] incremental result mismatch, using safe rebuild");
      rebuild_all();
      return;
    }
  }
  removed_ids_.clear();
  first_new_ftr_ = frontiers_.end();
  ROS_INFO("[FrontierCost] cache_incremental: %d old, %d new, %d total",
           old_num, new_num, total_num);
}

bool FrontierFinder::getFrontierPairCost(int from_id, int to_id, double &cost,
                                         vector<Vector3d> &path) const {
  if (from_id < 0 || to_id < 0 || from_id >= static_cast<int>(frontiers_.size()) ||
      to_id >= static_cast<int>(frontiers_.size()))
    return false;

  auto from = frontiers_.begin();
  std::advance(from, from_id);
  if (from->id_ != from_id ||
      from->costs_.size() != frontiers_.size() ||
      from->paths_.size() != frontiers_.size())
    return false;

  auto cost_it = from->costs_.begin();
  auto path_it = from->paths_.begin();
  std::advance(cost_it, to_id);
  std::advance(path_it, to_id);
  cost = *cost_it;
  path = *path_it;
  return std::isfinite(cost);
}



void FrontierFinder::mergeFrontiers(Frontier& ftr1, const Frontier& ftr2) {
  // Merge ftr2 into ftr1
  ftr1.average_ =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  ftr1.cells_.insert(ftr1.cells_.end(), ftr2.cells_.begin(), ftr2.cells_.end());
  computeFrontierInfo(ftr1);
}

bool FrontierFinder::canBeMerged(const Frontier& ftr1, const Frontier& ftr2) {
  Vector3d merged_avg =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  // Check if it can merge two frontier without exceeding size limit
  for (auto c1 : ftr1.cells_) {
    auto diff = c1 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  for (auto c2 : ftr2.cells_) {
    auto diff = c2 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  return true;
}

bool FrontierFinder::haveOverlap(
    const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // Check if two box have overlap part
  Vector3d bmin, bmax;
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3) return false;
  }
  return true;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx))) return true;
  }
  return false;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  // ==================== 1. 计算前沿簇平均位置与包围盒 ====================
  // average_：用于表征该前沿簇的大致几何中心
  // box_min_ / box_max_：用于描述该簇的轴对齐包围盒（AABB）
  ftr.average_.setZero();
  if (ftr.cells_.empty()) return;  // V11: guard against empty cells
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();

  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= double(ftr.cells_.size());

  // ==================== 2. 对前沿簇做降采样 ====================
  // filtered_cells_ 用于减少后续视点评估和结构分析的计算量
  downsample(ftr.cells_, ftr.filtered_cells_);

  // ==================== 3. 计算窄通道性 narrowness_（基于PCA主轴尺度） ====================
  // 设计思想：
  // 不再使用AABB的 dx / dy 长宽比，因为它会受到坐标轴方向影响。
  // 这里改为在 XY 平面上对 frontier.filtered_cells_ 做 PCA（主成分分析），
  // 用主轴和次轴的尺度差异来刻画"狭长程度"。
  //
  // 若前沿簇像走廊边界、门洞边缘、狭长入口，则：
  //   major_scale_ >> minor_scale_
  //   narrowness_ 接近 1
  //
  // 若前沿簇更接近团状/块状，则：
  //   major_scale_ ≈ minor_scale_
  //   narrowness_ 接近 0

  // 取当前前沿簇平均位置在 XY 平面的投影作为均值
  Eigen::Vector2d mean_xy = ftr.average_.head<2>();

  // 计算 XY 平面协方差矩阵
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (const auto& cell : ftr.filtered_cells_) {
    Eigen::Vector2d d = cell.head<2>() - mean_xy;
    cov += d * d.transpose();
  }

  // 避免空点集时除零
  if (!ftr.filtered_cells_.empty()) {
    cov /= double(ftr.filtered_cells_.size());
  }

  // 对称矩阵特征分解，得到两个特征值（升序）
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
  Eigen::Vector2d eig_vals = es.eigenvalues();

  // 特征值不能为负，给一个极小正数做数值保护
  double lambda_minor = std::max(eig_vals[0], 1e-6);
  double lambda_major = std::max(eig_vals[1], 1e-6);

  // 用标准差（sqrt(lambda)）而不是方差，更符合"尺度"直觉
  ftr.minor_scale_ = std::sqrt(lambda_minor);
  ftr.major_scale_ = std::sqrt(lambda_major);

  // 窄通道性定义：
  // 次轴越小、主轴越大，则该值越大
  ftr.narrowness_ =
      1.0 - std::sqrt((lambda_minor + 1e-6) / (lambda_major + 1e-6));

  // 限制到 [0, 1]
  ftr.narrowness_ = std::min(std::max(ftr.narrowness_, 0.0), 1.0);


  // ==================== 4. 计算凹陷度 concavity_（邻域障碍分布不对称性） ====================
  // 设计思想：
  // 不再围绕 frontier.average_ 重新做一轮视点采样，因为原 FUEL 已经做了 viewpoint sampling。
  // 这里直接利用前沿簇周围已有地图结构，统计"障碍物在各个方向上的分布是否集中"。
  //
  // 若障碍物主要集中在少数方向，通常说明该区域更像：
  //   - 墙角
  //   - 凹陷区域
  //   - 单入口袋状区域
  //
  // 若障碍物在周围方向分布较均匀，则更像普通边界，concavity_ 较小。

  // 方向分桶数，至少为 8
  int K = std::max(feature_dir_num_, 8);

  // 统计每个方向扇区内的障碍体素数量
  std::vector<double> occ_hist(K, 0.0);

  // 对每个降采样后的前沿点，在其邻域内统计障碍方向分布
  for (const auto& cell : ftr.filtered_cells_) {
    Eigen::Vector3i center_idx;
    edt_env_->sdf_map_->posToIndex(cell, center_idx);

    // 将 feature_radius_ 换算成体素步数
    int step = std::max(1, int(std::round(feature_radius_ / resolution_)));

    // 在 XY 平面邻域内遍历
    for (int dx = -step; dx <= step; ++dx) {
      for (int dy = -step; dy <= step; ++dy) {
        if (dx == 0 && dy == 0) continue;

        Eigen::Vector3i idx = center_idx + Eigen::Vector3i(dx, dy, 0);

        // 越界则跳过
        if (!edt_env_->sdf_map_->isInMap(idx)) continue;

        Eigen::Vector3d pos_nb;
        edt_env_->sdf_map_->indexToPos(idx, pos_nb);

        // 超出半径则跳过
        double dist = (pos_nb.head<2>() - cell.head<2>()).norm();
        if (dist > feature_radius_) continue;

        // 只统计膨胀障碍
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) != 1) continue;

        // 计算障碍相对当前前沿点的方向角
        double theta = std::atan2(pos_nb[1] - cell[1], pos_nb[0] - cell[0]);

        // 映射到 [0, 2pi)
        if (theta < 0.0) theta += 2.0 * M_PI;

        // 分配到对应方向桶
        int bin = std::min(K - 1, int(std::floor(theta / (2.0 * M_PI) * K)));
        occ_hist[bin] += 1.0;
      }
    }
  }

  // 统计障碍总量
  double occ_sum = 0.0;
  for (double v : occ_hist) occ_sum += v;

  // 若邻域几乎没有障碍，则认为该前沿簇不具备明显凹陷风险
  if (occ_sum < 1e-6) {
    ftr.obstacle_anisotropy_ = 0.0;
    ftr.obstacle_density_ = 0.0;
    ftr.concavity_ = 0.0;
  } else {
    // 平均每个方向桶的障碍量，可作为邻域拥挤程度参考
    ftr.obstacle_density_ = occ_sum / double(K);

    // 计算归一化二阶矩：
    // 分布越集中在少数方向，值越大
    double sq_sum = 0.0;
    for (double v : occ_hist) {
      double p = v / occ_sum;
      sq_sum += p * p;
    }

    // 归一化到 [0, 1]
    // - 均匀分布 -> 0
    // - 极端集中 -> 1
    ftr.obstacle_anisotropy_ =
        (sq_sum - 1.0 / double(K)) / (1.0 - 1.0 / double(K));
    ftr.obstacle_anisotropy_ =
        std::min(std::max(ftr.obstacle_anisotropy_, 0.0), 1.0);

    // 当前第一版：直接把障碍分布不对称性作为凹陷度
    ftr.concavity_ = ftr.obstacle_anisotropy_;
  }

  // ==================== 5. 调试输出 ====================
  // 后续跑迷宫时，可以直接观察：
  // - major/minor 是否合理
  // - narrowness_ 是否对狭长结构更敏感
  // - concavity_ 是否对角落/凹陷区域更敏感
  ROS_WARN_STREAM("[Feature] avg = " << ftr.average_.transpose()
                  << " | major = " << ftr.major_scale_
                  << " | minor = " << ftr.minor_scale_
                  << " | narrowness = " << ftr.narrowness_
                  << " | obs_anisotropy = " << ftr.obstacle_anisotropy_
                  << " | obs_density = " << ftr.obstacle_density_
                  << " | concavity = " << ftr.concavity_);
}

void FrontierFinder::computeFrontiersToVisit() {
  // forget_flag 在 refreshFrontierTrackingForPlanning() 中计算，
  // 在 updateCoverage() 之后执行。

  first_new_ftr_ = frontiers_.end();
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    sampleViewpoints(tmp_ftr);

    tmp_ftr.forget_flag_ = false;
    tmp_ftr.stale_fraction_ = 0.0;
    tmp_ftr.coverage_overlap_ = 0.0;
    tmp_ftr.filtered_coverage_overlap_ = 0.0;
    tmp_ftr.debt_score_ = 0.0;
    tmp_ftr.progress_delta_ = 0.0;
    tmp_ftr.path_overlap_ = 0.0;
    tmp_ftr.forget_reason_ = 0;
    tmp_ftr.region_id_ = -1;
    tmp_ftr.local_pass_cnt_ = 0;
    tmp_ftr.attempt_cnt_ = 0;
    tmp_ftr.cooldown_until_round_ = 0;
    tmp_ftr.back_unknown_num_ = 0;
    tmp_ftr.back_obstacle_ratio_ = 0.0;
    tmp_ftr.back_leak_ratio_ = 0.0;
    tmp_ftr.bounded_pocket_score_ = 0.0;
    tmp_ftr.bounded_pocket_flag_ = false;
    tmp_ftr.persistent_coverage_overlap_ = 0.0;
    tmp_ftr.closeout_overlap_ = 0.0;
    tmp_ftr.adaptive_voxel_score_ = 0.0;
    tmp_ftr.invalid_closeout_ = false;
    tmp_ftr.tsp_layer_ = 3;
    tmp_ftr.closeout_type_ = 0;

    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      tmp_ftr.id_ = static_cast<int>(frontiers_.size());
      list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
      // Sort the viewpoints by coverage fraction, best view in front
      
      // ==================== 调试输出：打印新前沿簇信息 ====================
      ROS_WARN_STREAM("[Frontier] new frontier inserted"
                << ", id = " << inserted->id_
                << ", avg = " << inserted->average_.transpose()
                << ", concavity = " << inserted->concavity_
                << ", narrowness = " << inserted->narrowness_
                << ", cell_num = " << inserted->cells_.size()
                << ", filtered_cell_num = " << inserted->filtered_cells_.size());
      sort(
          inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) {
            if (v1.visib_num_ != v2.visib_num_) return v1.visib_num_ > v2.visib_num_;
            return v1.visible_unknown_num_ > v2.visible_unknown_num_;
          });
      if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
    } else {
      // Find no viewpoint, move cluster to dormant list
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  // Reset indices of frontiers
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
    ft.stable_sig_ = makeFrontierSignature(ft);
    std::cout << ft.id_ << ", ";
  }
  std::cout << "\nnew num: " << new_num << ", new dormant: " << new_dormant_num << std::endl;
  std::cout << "to visit: " << frontiers_.size() << ", dormant: " << dormant_frontiers_.size()
            << std::endl;

}

std::string FrontierFinder::makeFrontierSignature(const Frontier& frontier) const {
  // 仅用于日志分析，不参与规划决策
  // 用较粗的网格量化中心和包围盒尺寸，避免 id 每轮重排导致无法追踪
  const double sig_res = std::max(0.5, 4.0 * resolution_);

  auto quantize = [&](double v) -> int {
    return static_cast<int>(std::floor(v / sig_res + 0.5));
  };

  Eigen::Vector3d center = 0.5 * (frontier.box_min_ + frontier.box_max_);
  Eigen::Vector3d size = frontier.box_max_ - frontier.box_min_;

  std::ostringstream ss;
  ss << quantize(center(0)) << "_"
     << quantize(center(1)) << "_"
     << quantize(center(2)) << "_"
     << quantize(size(0)) << "_"
     << quantize(size(1)) << "_"
     << quantize(size(2));

  return ss.str();
}


void FrontierFinder::updateFrontierTrackingStats(const Vector3d& cur_pos) {
  ++frontier_track_round_;

  const double LOCAL_BAND_RADIUS = local_closeout_radius_;

  for (auto& frontier : frontiers_) {
    // Step 1) 做稳定签名与跨轮次记忆：用于追踪“是不是同一个前沿簇”
    std::string sig = makeFrontierSignature(frontier);
    frontier.stable_sig_ = sig;

    auto last_it = sig_last_seen_round_.find(sig);
    if (last_it == sig_last_seen_round_.end()) {
      sig_persist_cnt_[sig] = 1;
      sig_first_seen_round_[sig] = frontier_track_round_;
    } else {
      if (last_it->second == frontier_track_round_ - 1) {
        sig_persist_cnt_[sig] += 1;
      } else {
        sig_persist_cnt_[sig] = 1;
      }
    }

    sig_last_seen_round_[sig] = frontier_track_round_;

    frontier.persist_cnt_ = sig_persist_cnt_[sig];
    frontier.first_seen_round_ = sig_first_seen_round_[sig];
    frontier.last_seen_round_ = sig_last_seen_round_[sig];

    // Step 2) 计算局部性：只有落在本地半径内的前沿才参与近场补观测
    double dist_to_uav = (frontier.average_ - cur_pos).norm();
    frontier.in_local_band_ = (dist_to_uav <= LOCAL_BAND_RADIUS);

    if (frontier.in_local_band_) {
      sig_local_pass_cnt_[sig] += 1;
      frontier.local_pass_cnt_ = sig_local_pass_cnt_[sig];
    }

    // Step 3) 更新覆盖相关统计：当前覆盖、持久覆盖、退化比例
    double cov_all = computeCoverageOverlap(frontier);
    frontier.filtered_coverage_overlap_ = computeFilteredCoverageOverlap(frontier);
    frontier.coverage_overlap_ = std::max(cov_all, frontier.filtered_coverage_overlap_);
    frontier.persistent_coverage_overlap_ = computePersistentCoverageOverlap(frontier);
    frontier.closeout_overlap_ =
        std::max(frontier.coverage_overlap_, frontier.persistent_coverage_overlap_);

    frontier.stale_fraction_ = computeStaleFraction(frontier);

    // Step 4) 更新进展量：前沿 cell 数下降越多，说明该簇被清理进展越大
    auto pc_it = sig_prev_cell_num_.find(sig);
    if (pc_it != sig_prev_cell_num_.end() && pc_it->second > 0) {
      int cur_cells = static_cast<int>(frontier.cells_.size());
      frontier.progress_delta_ =
          std::max(0.0, double(pc_it->second - cur_cells) / double(pc_it->second));
      sig_prev_cell_num_[sig] = cur_cells;
    } else {
      sig_prev_cell_num_[sig] = static_cast<int>(frontier.cells_.size());
      frontier.progress_delta_ = 0.0;
    }

    auto ac_it = sig_attempt_cnt_.find(sig);
    frontier.attempt_cnt_ =
        (ac_it != sig_attempt_cnt_.end()) ? ac_it->second : 0;

    auto cd_it = sig_cooldown_until_round_.find(sig);
    frontier.cooldown_until_round_ =
        (cd_it != sig_cooldown_until_round_.end()) ? cd_it->second : 0;

    // Step 5) 重置本轮证据容器，后面重新计算（避免沿用旧值）
    frontier.debt_score_ = 0.0;
    frontier.region_id_ = 0;

    frontier.back_unknown_num_ = 0;
    frontier.back_obstacle_ratio_ = 0.0;
    frontier.back_leak_ratio_ = 0.0;
    frontier.bounded_pocket_score_ = 0.0;
    frontier.bounded_pocket_flag_ = false;

    frontier.invalid_closeout_ = false;
    frontier.tsp_layer_ = 3;
    frontier.closeout_type_ = 0;

    // FAEP-style local residual evidence:
    // rear unknown leakage + obstacle-bounded pocket + pass debt.
    computeBackLeakPocketEvidence(frontier);
    computePassDebtScore(frontier, cur_pos);
    frontier.adaptive_voxel_score_ =
        computeAdaptiveVoxelForgetScore(frontier, cur_pos);
  }
}

void FrontierFinder::writeFrontierTrackingLogOnly(const Vector3d& cur_pos) {
  if (!enable_frontier_track_log_ || !frontier_track_log_file_.is_open()) return;
  for (auto& frontier : frontiers_) {
    std::string sig = frontier.stable_sig_;
    if (sig.empty()) sig = makeFrontierSignature(frontier);
    double dist_to_uav = (frontier.average_ - cur_pos).norm();
    int top_visib = frontier.viewpoints_.empty() ? 0 : frontier.viewpoints_.front().visib_num_;
    frontier_track_log_file_
        << frontier_track_round_ << "," << ros::Time::now().toSec() << ","
        << frontier.id_ << ","
        << sig << ","
        << frontier.average_(0) << "," << frontier.average_(1) << "," << frontier.average_(2) << ","
        << frontier.box_min_(0) << "," << frontier.box_min_(1) << "," << frontier.box_min_(2) << ","
        << frontier.box_max_(0) << "," << frontier.box_max_(1) << "," << frontier.box_max_(2) << ","
        << frontier.cells_.size() << "," << frontier.filtered_cells_.size() << ","
        << dist_to_uav << "," << frontier.in_local_band_ << ","
        << frontier.persist_cnt_ << "," << frontier.first_seen_round_ << "," << frontier.last_seen_round_ << ","
        << frontier.forget_flag_ << "," << frontier.stale_fraction_ << ","
        << frontier.coverage_overlap_ << "," << frontier.forget_reason_ << ","
        << frontier.persistent_coverage_overlap_ << "," << frontier.closeout_overlap_ << ","
        << frontier.concavity_ << "," << frontier.narrowness_ << "," << top_visib << ","
        << frontier.filtered_coverage_overlap_ << ","
        << frontier.debt_score_ << "," << frontier.progress_delta_ << ","
        << frontier.local_pass_cnt_ << "," << frontier.attempt_cnt_ << "," << frontier.cooldown_until_round_ << ","
        << frontier.path_overlap_ << ","
        << frontier.last_start_path_size_ << "," << frontier.last_path_length_ << ","
        << frontier.last_start_travel_cost_ << "," << frontier.last_start_final_cost_ << ","
        << frontier.min_pass_dist_ << ","
        << frontier.near_pass_cnt_ << ","
        << frontier.last_near_round_ << ","
        << frontier.pass_debt_score_ << ","
        << frontier.back_unknown_num_ << "," << frontier.back_obstacle_ratio_ << "," << frontier.back_leak_ratio_ << ","
        << frontier.bounded_pocket_score_ << "," << frontier.bounded_pocket_flag_ << ","
        << frontier.adaptive_voxel_score_ << "," << frontier.invalid_closeout_ << ","
        << frontier.tsp_layer_ << "," << frontier.closeout_type_ << ","
        << log_frontier_stage_ << "," << log_root_lock_ << ","
        << log_root_unreachable_rounds_ << ","
        << log_active_grid_id_ << "," << log_active_root_grid_id_ << "\n";
  }
  frontier_track_log_file_.flush();
}

void FrontierFinder::getTopViewpointsInfo(
    const Vector3d& cur_pos,
    vector<Eigen::Vector3d>& points,
    vector<double>& yaws,
    vector<Eigen::Vector3d>& averages) {

  points.clear();
  yaws.clear();
  averages.clear();

  // Viewpoint quality follows the paper: N_front is primary and N_unknown
  // breaks ties. This does not alter frontier-level priority computation.
  for (const auto& frontier : frontiers_) {
    if (frontier.viewpoints_.empty()) continue;

    bool found_valid_view = false;
    Viewpoint best_view = frontier.viewpoints_.front();

    for (const auto& view : frontier.viewpoints_) {
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;

      const bool better_quality =
          view.visib_num_ > best_view.visib_num_ ||
          (view.visib_num_ == best_view.visib_num_ &&
           view.visible_unknown_num_ > best_view.visible_unknown_num_);
      if (!found_valid_view || better_quality) {
        best_view = view;
        found_valid_view = true;
      }
    }

    // fallback：如果全部被距离阈值过滤掉，就退回簇内质量最高的视点
    if (!found_valid_view) {
      int fallback_best_score = -1;
      for (const auto& view : frontier.viewpoints_) {
        if (view.visib_num_ > fallback_best_score ||
            (view.visib_num_ == fallback_best_score &&
             view.visible_unknown_num_ > best_view.visible_unknown_num_)) {
          fallback_best_score = view.visib_num_;
          best_view = view;
          found_valid_view = true;
        }
      }
    }

    if (found_valid_view) {
      points.push_back(best_view.pos_);
      yaws.push_back(best_view.yaw_);
      averages.push_back(frontier.average_);
    }
  }
}

void FrontierFinder::getViewpointsInfo(
    const Vector3d& cur_pos,
    const double& cur_yaw,
    const vector<int>& ids,
    const int& view_num,
    const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points,
    vector<vector<double>>& yaws) {

  points.clear();
  yaws.clear();
  // Flight time/yaw is evaluated on the multi-frontier viewpoint graph.
  // Do not let it discard a high-quality view during candidate screening.
  (void)cur_yaw;

  for (int local_order = 0; local_order < static_cast<int>(ids.size()); ++local_order) {
    const int id = ids[local_order];
    Frontier* frontier_ptr = getFrontierById(id);
    if (frontier_ptr == nullptr || frontier_ptr->viewpoints_.empty()) continue;
    const Frontier& frontier = *frontier_ptr;

    struct ViewScore {
      int visib_num_ = 0;
      int visible_unknown_num_ = 0;
      Vector3d pos_ = Vector3d::Zero();
      double yaw_ = 0.0;
    };
    vector<ViewScore> scored_views;
    int max_view_num = view_num > 0 ? view_num : (int)frontier.viewpoints_.size();
    int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;

    for (const auto& view : frontier.viewpoints_) {
      if (view.visib_num_ <= visib_thresh) break;
      const double dist = (view.pos_ - cur_pos).norm();
      if (dist < min_candidate_dist_) continue;
      scored_views.push_back(
          {view.visib_num_, view.visible_unknown_num_, view.pos_, view.yaw_});
    }

    if (scored_views.empty()) {
      for (const auto& view : frontier.viewpoints_) {
        if (view.visib_num_ <= visib_thresh) break;
        scored_views.push_back(
            {view.visib_num_, view.visible_unknown_num_, view.pos_, view.yaw_});
      }
    }

    std::sort(scored_views.begin(), scored_views.end(),
              [](const ViewScore& a, const ViewScore& b) {
                // View quality: N_front is primary, N_unknown is secondary.
                if (a.visib_num_ != b.visib_num_) return a.visib_num_ > b.visib_num_;
                if (a.visible_unknown_num_ != b.visible_unknown_num_)
                  return a.visible_unknown_num_ > b.visible_unknown_num_;
                if (std::fabs(a.pos_.x() - b.pos_.x()) > 1e-9) return a.pos_.x() < b.pos_.x();
                if (std::fabs(a.pos_.y() - b.pos_.y()) > 1e-9) return a.pos_.y() < b.pos_.y();
                return a.yaw_ < b.yaw_;
              });

    vector<Eigen::Vector3d> pts;
    vector<double> ys;
    for (const auto& sv : scored_views) {
      if ((int)pts.size() >= max_view_num) break;
      pts.push_back(sv.pos_);
      ys.push_back(sv.yaw_);
    }

    if (!pts.empty()) {
      points.push_back(pts);
      yaws.push_back(ys);
    }

    if (enable_local_view_log_ && local_view_log_file_.is_open()) {
      const std::string sig = frontier.stable_sig_.empty() ? makeFrontierSignature(frontier)
                                                           : frontier.stable_sig_;
      const double dist_to_uav = (frontier.average_ - cur_pos).norm();
      const int candidate_num = static_cast<int>(frontier.viewpoints_.size());
      const int selected_num = static_cast<int>(pts.size());
      const int best_visib = frontier.viewpoints_.empty() ? 0 : frontier.viewpoints_.front().visib_num_;
      const double best_view_dist =
          frontier.viewpoints_.empty() ? -1.0 : (frontier.viewpoints_.front().pos_ - cur_pos).norm();
      const int best_unknown =
          frontier.viewpoints_.empty() ? 0 : frontier.viewpoints_.front().unknown_num_;
      const auto path_it = local_view_start_path_lengths_.find(frontier.id_);
      const auto order_it = local_view_dijkstra_orders_.find(frontier.id_);
      const auto reason_it = local_view_dijkstra_reasons_.find(frontier.id_);
      const double start_astar_length =
          path_it == local_view_start_path_lengths_.end() ? -1.0 : path_it->second;
      const int dijkstra_order =
          order_it == local_view_dijkstra_orders_.end() ? -1 : order_it->second;
      const int in_dijkstra = dijkstra_order >= 0 ? 1 : 0;
      const std::string dijkstra_reason =
          reason_it == local_view_dijkstra_reasons_.end() ? "unknown" : reason_it->second;

      local_view_log_file_
          << frontier_track_round_ << "," << ros::Time::now().toSec() << ","
          << frontier.id_ << "," << sig << "," << local_order << ","
          << candidate_num << "," << selected_num << ","
          << best_visib << "," << best_view_dist << "," << best_unknown << ","
          << dist_to_uav << "," << frontier.cells_.size() << ","
          << frontier.coverage_overlap_ << "," << frontier.closeout_overlap_ << ","
          << frontier.forget_flag_ << "," << frontier.stale_fraction_ << ","
          << frontier.debt_score_ << "," << frontier.progress_delta_ << ","
          << frontier.local_pass_cnt_ << "," << frontier.attempt_cnt_ << ","
          << frontier.cooldown_until_round_ << ","
          << frontier.in_local_band_ << "," << frontier.min_pass_dist_ << ","
          << frontier.near_pass_cnt_ << "," << frontier.last_near_round_ << ","
          << frontier.adaptive_voxel_score_ << "," << frontier.invalid_closeout_ << ","
          << frontier.closeout_type_ << ","
          << log_frontier_stage_ << "," << log_root_lock_ << ","
          << log_root_unreachable_rounds_ << ","
          << log_active_grid_id_ << "," << log_active_root_grid_id_ << ","
          << local_view_lkh_sequence_ << "," << start_astar_length << ","
          << dijkstra_order << "," << in_dijkstra << "," << dijkstra_reason << "\n";
    }
  }

  if (enable_local_view_log_ && local_view_log_file_.is_open()) {
    local_view_log_file_.flush();
  }

  ROS_WARN("[LocalView] frontier_order=%zu, viewpoint_groups=%zu", ids.size(), points.size());
  publishLocalExploreOrderViz(cur_pos, ids);
}

void FrontierFinder::publishLocalExploreOrderViz(
    const Vector3d &cur_pos,
    const vector<int> &ordered_ids) {
  if (local_order_vis_pub_.getNumSubscribers() <= 0) return;

  visualization_msgs::MarkerArray msg;

  visualization_msgs::Marker del;
  del.header.frame_id = "world";
  del.header.stamp = ros::Time::now();
  del.ns = "local_explore_order";
  del.id = 0;
  del.action = visualization_msgs::Marker::DELETEALL;
  msg.markers.push_back(del);

  visualization_msgs::Marker frontier_pts;
  frontier_pts.header.frame_id = "world";
  frontier_pts.header.stamp = ros::Time::now();
  frontier_pts.ns = "local_explore_order_frontiers";
  frontier_pts.id = 1;
  frontier_pts.type = visualization_msgs::Marker::SPHERE_LIST;
  frontier_pts.action = visualization_msgs::Marker::ADD;
  frontier_pts.pose.orientation.w = 1.0;
  frontier_pts.scale.x = 0.22;
  frontier_pts.scale.y = 0.22;
  frontier_pts.scale.z = 0.22;
  frontier_pts.color.r = 1.0;
  frontier_pts.color.g = 0.0;
  frontier_pts.color.b = 0.0;
  frontier_pts.color.a = 1.0;

  for (int id : ordered_ids) {
    const Frontier *f = getFrontierById(id);
    if (f == nullptr) continue;
    geometry_msgs::Point p;
    p.x = f->average_.x();
    p.y = f->average_.y();
    p.z = f->average_.z();
    frontier_pts.points.push_back(p);
  }
  msg.markers.push_back(frontier_pts);

  visualization_msgs::Marker nearby_views;
  nearby_views.header.frame_id = "world";
  nearby_views.header.stamp = ros::Time::now();
  nearby_views.ns = "local_explore_order_nearby_views";
  nearby_views.id = 3;
  nearby_views.type = visualization_msgs::Marker::SPHERE_LIST;
  nearby_views.action = visualization_msgs::Marker::ADD;
  nearby_views.pose.orientation.w = 1.0;
  nearby_views.scale.x = 0.12;
  nearby_views.scale.y = 0.12;
  nearby_views.scale.z = 0.12;
  nearby_views.color.r = 0.1;
  nearby_views.color.g = 0.4;
  nearby_views.color.b = 1.0;
  nearby_views.color.a = 0.9;

  const double nearby_radius = 5.0;
  for (const auto &f : frontiers_) {
    if ((f.average_ - cur_pos).norm() > nearby_radius) continue;
    for (const auto &vp : f.viewpoints_) {
      geometry_msgs::Point p;
      p.x = vp.pos_.x();
      p.y = vp.pos_.y();
      p.z = vp.pos_.z();
      nearby_views.points.push_back(p);
    }
  }
  msg.markers.push_back(nearby_views);

  local_order_vis_pub_.publish(msg);
}


// 当前方案不再进行局部强制插队。
// 自适应体素遗忘只通过 TSP 第一行有界代价修正参与排序。
Frontier* FrontierFinder::getFrontierById(int id) {
  for (auto& f : frontiers_) {
    if (f.id_ == id) return &f;
  }
  return nullptr;
}

// ================================================================
// 基于当前地图重新评估closeout视点真实可见性
// ================================================================
bool FrontierFinder::selectBestCurrentCloseoutViewpoint(
    const Frontier& f,
    const Eigen::Vector3d& cur_pos,
    Viewpoint& best_view) {

  if (f.viewpoints_.empty() || percep_utils_ == nullptr) return false;

  std::vector<Eigen::Vector3d> live_cells;
  live_cells.reserve(f.filtered_cells_.size());
  const std::vector<Eigen::Vector3d>& src_cells =
      f.filtered_cells_.empty() ? f.cells_ : f.filtered_cells_;
  for (const auto& cell : src_cells) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!edt_env_->sdf_map_->isInMap(idx)) continue;
    if (!(knownfree(idx) && isNeighborUnknown(idx))) continue;
    live_cells.push_back(cell);
  }
  if (live_cells.empty()) return false;

  bool found = false;
  double best_score = -1e9;

  // yaw-only已禁用：cur_pos作waypoint导致轨迹退化

  for (const auto& v : f.viewpoints_) {
    double dist = (v.pos_ - cur_pos).norm();
    if (dist < std::max(0.8, closeout_min_view_dist_)) continue;
    if (dist > local_closeout_radius_ + 0.5) continue;
    if (!edt_env_->sdf_map_->isInBox(v.pos_)) continue;
    if (edt_env_->sdf_map_->getInflateOccupancy(v.pos_) == 1) continue;

    int cur_visib = countVisibleCells(v.pos_, v.yaw_, live_cells);
    int min_need = std::max(min_visib_num_, int(0.12 * double(live_cells.size())));
    if (cur_visib < min_need) continue;

    int cur_unknown = 0;
    percep_utils_->setPose(v.pos_, v.yaw_);
    for (const auto& cell : live_cells) {
      if (!percep_utils_->insideFOV(cell)) continue;
      Eigen::Vector3i idx;
      raycaster_->input(cell, v.pos_);
      bool visible = true;
      while (raycaster_->nextId(idx)) {
        if (!edt_env_->sdf_map_->isInMap(idx)) { visible = false; break; }
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) { visible = false; break; }
      }
      if (!visible) continue;
      Eigen::Vector3d dir = (cell - v.pos_).normalized();
      Eigen::Vector3d behind = cell + 2.0 * dir;
      raycaster_->input(cell, behind);
      while (raycaster_->nextId(idx)) {
        if (!edt_env_->sdf_map_->isInMap(idx)) break;
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1) break;
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) { ++cur_unknown; break; }
      }
    }

    double visib_norm = double(cur_visib) / std::max(1.0, double(live_cells.size()));
    double dist_norm = std::min(dist / std::max(local_closeout_radius_, 1.0), 1.0);
    double score = 0.75 * visib_norm + 0.10 * double(cur_unknown) / std::max(1.0, double(live_cells.size())) - 0.15 * dist_norm;

    if (!found || score > best_score) {
      found = true; best_score = score; best_view = v;
      best_view.visib_num_ = cur_visib;
      best_view.unknown_num_ = cur_unknown;
    }
  }
  return found;
}


void FrontierFinder::markCoverageSegment(
    const Vector3d& p0,
    const Vector3d& p1,
    int stamp_round) {

  if (coverage_last_round_.size() != coverage_flag_.size()) {
    coverage_last_round_.assign(coverage_flag_.size(), -1000000);
  }

  const double R = inflate_radius_;
  const double sample_step = resolution_;
  const int step_xy = std::max(1, static_cast<int>(std::ceil(inflate_radius_ / resolution_)));
  const int step_z = step_xy;
  const int stride = 1;
  const double r2 = R * R;
  auto mark_disk = [&](const Vector3d& p) {
    if (!edt_env_->sdf_map_->isInBox(p)) return;
    Vector3i cidx;
    edt_env_->sdf_map_->posToIndex(p, cidx);
    for (int dx = -step_xy; dx <= step_xy; dx += stride) {
      for (int dy = -step_xy; dy <= step_xy; dy += stride) {
        const double xy2 = double(dx * dx + dy * dy) * resolution_ * resolution_;
        if (xy2 > r2) continue;
        for (int dz = -step_z; dz <= step_z; dz += stride) {
          Eigen::Vector3i idx = cidx + Eigen::Vector3i(dx, dy, dz);
          if (!edt_env_->sdf_map_->isInBox(idx)) continue;
          if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1) continue;
          if (edt_env_->sdf_map_->getOccupancy(idx) != SDFMap::FREE) continue;
          const int adr = toadr(idx);
          if (adr < 0 || adr >= static_cast<int>(coverage_flag_.size())) continue;
          coverage_last_round_[adr] = stamp_round;
          coverage_flag_[adr] = 1;
          coverage_ever_flag_[adr] = 1;
        }
      }
    }
  };
  auto segment_clear = [&](const Vector3d& a, const Vector3d& b) {
    if (!raycaster_) return true;
    if (!edt_env_->sdf_map_->isInBox(a) || !edt_env_->sdf_map_->isInBox(b)) return false;
    Vector3i idx;
    raycaster_->input(a, b);
    while (raycaster_->nextId(idx)) {
      if (!edt_env_->sdf_map_->isInMap(idx)) return false;
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        return false;
      }
    }
    return true;
  };

  Eigen::Vector2d tangent = (p1 - p0).head<2>();
  double seg_len = tangent.norm();
  if (seg_len < 1e-6) return;

  tangent /= seg_len;
  Eigen::Vector2d normal(-tangent.y(), tangent.x());

  // Truncate coverage at the first obstacle/unknown intersection.
  Vector3d last_safe_center = p0;
  mark_disk(last_safe_center);

  for (double s = sample_step; s < seg_len + sample_step; s += sample_step) {
    double frac = std::min(s / seg_len, 1.0);
    Vector3d center = p0 + frac * (p1 - p0);
    if (!segment_clear(last_safe_center, center)) break;
    last_safe_center = center;

    int max_steps = static_cast<int>(R / resolution_) + 1;

    for (int side = -1; side <= 1; side += 2) {
      Eigen::Vector2d dir_xy = normal * static_cast<double>(side);

      for (int k = 0; k < max_steps; ++k) {
        double dist = k * resolution_;
        Vector3d probe = center + dist * Vector3d(dir_xy.x(), dir_xy.y(), 0.0);

        Eigen::Vector3i probe_idx;
        edt_env_->sdf_map_->posToIndex(probe, probe_idx);

        if (!edt_env_->sdf_map_->isInMap(probe_idx)) break;

        int adr = toadr(probe_idx);
        if (adr < 0 || adr >= static_cast<int>(coverage_flag_.size())) break;

        if (edt_env_->sdf_map_->getInflateOccupancy(probe_idx) == 1) break;

        // 标记当前体素及沿航向 ±1 的相邻体素
        for (int dt = -1; dt <= 1; ++dt) {
          Vector3d filled =
              probe + dt * resolution_ * Vector3d(tangent.x(), tangent.y(), 0.0);

          Eigen::Vector3i filled_idx;
          edt_env_->sdf_map_->posToIndex(filled, filled_idx);

          if (!edt_env_->sdf_map_->isInMap(filled_idx)) continue;

          int fadr = toadr(filled_idx);
          if (fadr < 0 || fadr >= static_cast<int>(coverage_flag_.size())) continue;

          if (edt_env_->sdf_map_->getInflateOccupancy(filled_idx) == 1) continue;

          // recent coverage stamp
          coverage_last_round_[fadr] = stamp_round;

          // 保留 coverage_flag_，避免旧日志或可视化函数直接失效
          coverage_flag_[fadr] = 1;
    coverage_ever_flag_[fadr] = 1;
        }

        coverage_last_round_[adr] = stamp_round;
        coverage_flag_[adr] = 1;
    coverage_ever_flag_[adr] = 1;
      }
    }
  }
}

void FrontierFinder::updateCoverage(const Vector3d& cur_pos, const Vector3d& cur_yaw) {
  (void)cur_yaw;

  if (coverage_last_round_.size() != coverage_flag_.size()) {
    coverage_last_round_.assign(coverage_flag_.size(), -1000000);
  }

  // 第一次只记录起点
  if (path_history_.empty()) {
    path_history_.push_back(cur_pos);
    return;
  }

  // 每移动 0.5m 就更新一次 coverage segment
  // 这样更容易捕捉拐角、凹陷、窄门附近的顺路残留
  const double COVERAGE_UPDATE_STEP = 0.30;

  if ((cur_pos - path_history_.back()).norm() <= COVERAGE_UPDATE_STEP) {
    return;
  }

  Vector3d p0 = path_history_.back();
  path_history_.push_back(cur_pos);

  while (static_cast<int>(path_history_.size()) > PATH_HISTORY_MAX) {
    path_history_.pop_front();
  }

  ++coverage_round_;

  if (coverage_mark_only_new_segment_) {
    // 方案一：只标记新增航段，不再每轮重刷完整历史路径
    markCoverageSegment(p0, cur_pos, coverage_round_);
  } else {
    // 兼容旧逻辑：不推荐使用
    for (size_t i = 1; i < path_history_.size(); ++i) {
      markCoverageSegment(path_history_[i - 1], path_history_[i], coverage_round_);
    }
  }
}

// 计算前沿簇中被覆盖地图标记的 cell 占比
// 返回值 ∈ [0, 1]：值越高说明该前沿位于无人机已观测路径走廊内
double FrontierFinder::computeCoverageOverlap(const Frontier& frontier) const {
  if (frontier.cells_.empty()) return 0.0;
  if (coverage_last_round_.empty()) return 0.0;

  int covered_count = 0;
  int valid_round = coverage_round_ - coverage_memory_rounds_;

  for (const auto& cell : frontier.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);

    if (!edt_env_->sdf_map_->isInMap(idx)) continue;

    int adr = toadr(idx);
    if (adr < 0 || adr >= static_cast<int>(coverage_last_round_.size())) continue;

    if (coverage_last_round_[adr] >= valid_round) {
      ++covered_count;
    }
  }

  return static_cast<double>(covered_count) / frontier.cells_.size();
}

double FrontierFinder::computeFilteredCoverageOverlap(const Frontier& frontier) const {
  const vector<Vector3d>& cells =
      frontier.filtered_cells_.empty() ? frontier.cells_ : frontier.filtered_cells_;

  if (cells.empty()) return 0.0;
  if (coverage_last_round_.empty()) return 0.0;

  int covered_count = 0;
  int valid_round = coverage_round_ - coverage_memory_rounds_;

  for (const auto& cell : cells) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);

    if (!edt_env_->sdf_map_->isInMap(idx)) continue;

    int adr = toadr(idx);
    if (adr < 0 || adr >= static_cast<int>(coverage_last_round_.size())) continue;

    if (coverage_last_round_[adr] >= valid_round) {
      ++covered_count;
    }
  }

  return static_cast<double>(covered_count) / cells.size();
}

double FrontierFinder::computePathOverlap(const vector<Vector3d>& path) const {
  if (path.empty()) return 0.0;
  if (coverage_last_round_.empty()) return 0.0;

  int sample_num = 0;
  int covered_num = 0;
  int valid_round = coverage_round_ - coverage_memory_rounds_;

  const double sample_step = std::max(0.2, 2.0 * resolution_);

  auto addSample = [&](const Vector3d& pt) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(pt, idx);

    if (!edt_env_->sdf_map_->isInMap(idx)) return;

    int adr = toadr(idx);
    if (adr < 0 || adr >= static_cast<int>(coverage_last_round_.size())) return;

    ++sample_num;

    if (coverage_last_round_[adr] >= valid_round) {
      ++covered_num;
    }
  };

  if (path.size() == 1) {
    addSample(path.front());
  } else {
    for (size_t i = 1; i < path.size(); ++i) {
      const Vector3d& p0 = path[i - 1];
      const Vector3d& p1 = path[i];

      double len = (p1 - p0).norm();
      int steps = std::max(1, static_cast<int>(std::ceil(len / sample_step)));

      for (int k = 0; k <= steps; ++k) {
        double t = static_cast<double>(k) / static_cast<double>(steps);
        addSample(p0 + t * (p1 - p0));
      }
    }
  }

  if (sample_num == 0) return 0.0;
  return static_cast<double>(covered_num) / static_cast<double>(sample_num);
}


double FrontierFinder::computeStaleFraction(const Frontier& frontier) {
  if (frontier.cells_.empty()) return 0.0;

  int stale_cells = 0;
  for (const auto& cell : frontier.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!edt_env_->sdf_map_->isInMap(idx) ||
        edt_env_->sdf_map_->getOccupancy(idx) != SDFMap::FREE ||
        !isNeighborUnknown(idx)) {
      ++stale_cells;
    }
  }
  return static_cast<double>(stale_cells) / static_cast<double>(frontier.cells_.size());
}

void FrontierFinder::getFrontiers(vector<vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (auto frontier : frontiers_)
    clusters.push_back(frontier.cells_);
  // clusters.push_back(frontier.filtered_cells_);
}

void FrontierFinder::getDormantFrontiers(vector<vector<Vector3d>>& clusters) {
  clusters.clear();
  for (auto ft : dormant_frontiers_)
    clusters.push_back(ft.cells_);
}

void FrontierFinder::getFrontierBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  boxes.clear();
  for (auto frontier : frontiers_) {
    Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector3d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, scale));
  }
}

void FrontierFinder::getPathForTour(
    const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path) {
  // Make an frontier_indexer to access the frontier list easier
  vector<list<Frontier>::iterator> frontier_indexer;
  for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it)
    frontier_indexer.push_back(it);

  // V11: bounds-check all frontier_ids indices
  const int n_ftrs = static_cast<int>(frontier_indexer.size());
  if (frontier_ids.empty() || frontier_ids[0] < 0 || frontier_ids[0] >= n_ftrs) return;
  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  ViewNode::searchPath(pos, frontier_indexer[frontier_ids[0]]->viewpoints_.front().pos_, segment);
  path.insert(path.end(), segment.begin(), segment.end());

  // Get paths of tour passing all clusters
  for (int i = 0; i < static_cast<int>(frontier_ids.size()) - 1; ++i) {
    if (frontier_ids[i] < 0 || frontier_ids[i] >= n_ftrs) continue;
    int next_idx = frontier_ids[i + 1];
    if (next_idx < 0 || next_idx >= n_ftrs) continue;
    // Move to path to next cluster
    auto path_iter = frontier_indexer[frontier_ids[i]]->paths_.begin();
    for (int j = 0; j < next_idx && path_iter != frontier_indexer[frontier_ids[i]]->paths_.end(); ++j)
      ++path_iter;
    if (path_iter != frontier_indexer[frontier_ids[i]]->paths_.end())
      path.insert(path.end(), path_iter->begin(), path_iter->end());
  }
}



void FrontierFinder::refreshFrontierTrackingForPlanning(
    const Vector3d& cur_pos,
    const Vector3d cur_yaw) {
  updateCoverage(cur_pos, cur_yaw);
  updateFrontierTrackingStats(cur_pos);
  computeForgetFlags();
  writeFrontierTrackingLogOnly(cur_pos);
}

void FrontierFinder::findViewpoints(
    const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps) {
  if (!edt_env_->sdf_map_->isInBox(sample) ||
      edt_env_->sdf_map_->getInflateOccupancy(sample) == 1 || isNearUnknown(sample))
    return;

  double right_angle_ = percep_utils_->getRightAngle();
  double vertical_angle_ = percep_utils_->getTopAngle();
  double ray_length_ = percep_utils_->getMaxDist();

  // Central yaw is determined by frontier's average position and sample
  auto dir = ftr_avg - sample;
  double hc = atan2(dir[1], dir[0]);

  vector<int> slice_gains;
  // Evaluate info gain of different slices
  for (double phi_h = -M_PI_2; phi_h <= M_PI_2 + 1e-3; phi_h += M_PI / 18) {
    // Compute gain of one slice
    int gain = 0;
    for (double phi_v = -vertical_angle_; phi_v <= vertical_angle_; phi_v += vertical_angle_ / 3) {
      // Find endpoint of a ray
      Vector3d end;
      end[0] = sample[0] + ray_length_ * cos(phi_v) * cos(hc + phi_h);
      end[1] = sample[1] + ray_length_ * cos(phi_v) * sin(hc + phi_h);
      end[2] = sample[2] + ray_length_ * sin(phi_v);

      // Do raycasting to check info gain
      Vector3i idx;
      raycaster_->input(sample, end);
      while (raycaster_->nextId(idx)) {
        // Hit obstacle, stop the ray
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || !edt_env_->sdf_map_->isInBox(idx))
          break;
        // Count number of unknown cells
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) ++gain;
      }
    }
    slice_gains.push_back(gain);
  }

  // Sum up slices' gain to get different yaw's gain
  vector<pair<double, int>> yaw_gains;
  for (int i = 0; i < 6; ++i)  // [-90,-10]-> [10,90], delta_yaw = 20, 6 groups
  {
    double yaw = hc - M_PI_2 + M_PI / 9.0 * i + right_angle_;
    int gain = 0;
    for (int j = 2 * i; j < 2 * i + 9; ++j)  // 80 degree hFOV, 9 slices
      gain += slice_gains[j];
    yaw_gains.push_back(make_pair(yaw, gain));
  }

  // Get several yaws with highest gain
  vps.clear();
  sort(
      yaw_gains.begin(), yaw_gains.end(),
      [](const pair<double, int>& p1, const pair<double, int>& p2) {
        return p1.second > p2.second;
      });
  for (int i = 0; i < 3; ++i) {
    if (yaw_gains[i].second < min_visib_num_) break;
    Viewpoint vp = { sample, yaw_gains[i].first, yaw_gains[i].second };
    while (vp.yaw_ < -M_PI)
      vp.yaw_ += 2 * M_PI;
    while (vp.yaw_ > M_PI)
      vp.yaw_ -= 2 * M_PI;
    vps.push_back(vp);
  }
}

// Sample viewpoints around frontier's average position, check coverage to the frontier cells
void FrontierFinder::sampleViewpoints(Frontier& frontier) {
  // Evaluate sample viewpoints on circles, find ones that cover most cells
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_;
       rc <= candidate_rmax_ + 1e-3; rc += dr)
    for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
      const Vector3d sample_pos = frontier.average_ + rc * Vector3d(cos(phi), sin(phi), 0);

      // Qualified viewpoint is in bounding box and in safe region
      if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
          edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos))
        continue;

      // Compute average yaw
      auto& cells = frontier.filtered_cells_;
      Eigen::Vector3d ref_dir = (cells.front() - sample_pos).normalized();
      double avg_yaw = 0.0;
      for (int i = 1; i < cells.size(); ++i) {
        Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
        double yaw = acos(dir.dot(ref_dir));
        if (ref_dir.cross(dir)[2] < 0) yaw = -yaw;
        avg_yaw += yaw;
      }
      avg_yaw = avg_yaw / cells.size() + atan2(ref_dir[1], ref_dir[0]);
      wrapYaw(avg_yaw);
      // Keep FUEL's sampling / safety filters unchanged, but evaluate the
      // paper's two viewpoint-quality metrics on the accepted sample:
      // N_front and the visible unknown voxels behind those frontiers.
      int visib_num = 0;
      int visible_unknown_num = 0;
      countVisibleFrontierAndUnknown(sample_pos, avg_yaw, cells, visib_num,
                                     visible_unknown_num);

      // [djg] FAEP Bottom Ray (论文 Section III-B):
      // 方向向量: 视点位置 → 前沿平均位置
      // 射线起点: 前沿平均位置
      // 射线沿同方向继续向后方投射, hk = 触碰障碍物/地图边界的距离
      // cs(k) = (hmax - hk) / hmax → hk越小越可能是独立小区域
      // 地图边界和障碍物性质相同——都表示后方空间有限
      const double hmax = 5.0;
      double hk = hmax;
      {
        Vector3d dir = (frontier.average_ - sample_pos).normalized();
        Vector3d ray_start = frontier.average_;
        Vector3d ray_end = frontier.average_ + hmax * dir;
        raycaster_->input(ray_start, ray_end);
        Eigen::Vector3i idx, last_valid_idx;
        bool has_last = false;
        while (raycaster_->nextId(idx)) {
          if (!edt_env_->sdf_map_->isInMap(idx)) {
            if (has_last) {
              Eigen::Vector3d boundary_pos;
              edt_env_->sdf_map_->indexToPos(last_valid_idx, boundary_pos);
              hk = std::max(0.05, (boundary_pos - ray_start).norm());
            }
            break;
          }
          last_valid_idx = idx;
          has_last = true;
          int occ = edt_env_->sdf_map_->getOccupancy(idx);
          if (occ == SDFMap::OCCUPIED ||
              edt_env_->sdf_map_->getInflateOccupancy(idx) == 1) {
            Eigen::Vector3d hit_pos;
            edt_env_->sdf_map_->indexToPos(idx, hit_pos);
            hk = std::max(0.05, (hit_pos - ray_start).norm());
            break;
          }
        }
      }
      int pocket_depth = static_cast<int>(hk / edt_env_->sdf_map_->getResolution());

      if (visib_num > min_visib_num_) {
        // unknown_num_ remains the legacy rear-ray depth used by the
        // frontier-level missed-area logic. The new metric is independent.
        Viewpoint vp = { sample_pos, avg_yaw, visib_num, pocket_depth, visible_unknown_num };
        frontier.viewpoints_.push_back(vp);
        // int gain = findMaxGainYaw(sample_pos, frontier, sample_yaw);
      }
      // }
    }
}



double FrontierFinder::computePersistentCoverageOverlap(const Frontier& frontier) const {
  if (frontier.cells_.empty()) return 0.0;
  if (coverage_ever_flag_.empty()) return 0.0;
  int covered = 0, total = 0;
  for (const auto& cell : frontier.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!edt_env_->sdf_map_->isInMap(idx)) continue;
    int adr = toadr(idx);
    if (adr < 0 || adr >= (int)coverage_ever_flag_.size()) continue;
    ++total;
    if (coverage_ever_flag_[adr]) ++covered;
  }
  return total == 0 ? 0.0 : (double)covered / (double)total;
}

void FrontierFinder::computeForgetFlags() {
  const int MIN_FLAG_CELLS = purge_min_cells_;
  const int MAX_CLOSEOUT_CELLS = 300;

  for (auto& frontier : frontiers_) {
    frontier.forget_flag_ = false;
    frontier.forget_reason_ = 0;
    frontier.invalid_closeout_ = false;
    frontier.tsp_layer_ = 3;
    frontier.closeout_type_ = 0;

    int total_cells = static_cast<int>(frontier.cells_.size());

    // forget_flag 的语义：该前沿是“局部、近期、可修复”的补观测候选
    bool size_ok =
        total_cells >= MIN_FLAG_CELLS &&
        total_cells <= MAX_CLOSEOUT_CELLS;

    bool valid =
        frontier.cooldown_until_round_ <= frontier_track_round_;

    bool currently_local =
        frontier.in_local_band_;

    bool history_relevant =
        frontier.min_pass_dist_ <= local_closeout_radius_ + 0.5 &&
        frontier.last_near_round_ > 0 &&
        frontier_track_round_ - frontier.last_near_round_ <= coverage_memory_rounds_;

    // 即时局部残留：不依赖“历史上路过”，当前就在局部且未知残留证据强
    bool instant_local_relevant =
        currently_local &&
        std::max(frontier.coverage_overlap_, frontier.filtered_coverage_overlap_) <= 0.20 &&
        frontier.bounded_pocket_score_ >= 0.70 &&
        frontier.back_unknown_num_ >= 8;

    // 这里只做候选标记，不要卡 persist_cnt / near_pass_cnt / progress_delta
    // 否则很多本来该优先的局部残留进不了 TSP 第一行修正。
    // pocket 证据收紧：避免长期“全真”导致区分度下降
    bool strong_pocket =
        frontier.bounded_pocket_score_ >= 0.70 &&
        frontier.back_unknown_num_ >= 8;

    bool score_ok =
        frontier.adaptive_voxel_score_ >= adaptive_voxel_thresh_ ||
        frontier.pass_debt_score_ >= 0.40 ||
        strong_pocket;

    // 这里仅做候选标记，不直接改 TSP 顺序；
    // 排序由局部 TSP 的 start_cost + coverage/component 偏置决定
    if (size_ok &&
        valid &&
        currently_local &&
        (history_relevant || instant_local_relevant) &&
        score_ok) {
      frontier.forget_flag_ = true;
      frontier.forget_reason_ |= 256;
      if (instant_local_relevant && !history_relevant) {
        frontier.forget_reason_ |= 512;  // 即时局部补观测触发
      }
      frontier.closeout_type_ = 6;
    }

    if (frontier.cooldown_until_round_ > frontier_track_round_) {
      frontier.forget_reason_ |= 8;
    }
  }
}

// ====================  全量前沿清理（保守版） ====================
// 设计思想：
// 原始 isFrontierCovered() 和 searchFrontiers() 仅依赖 getUpdatedBox()
// 来判断前沿是否已被观测覆盖，可能存在遗漏。
//
// 本函数直接遍历所有前沿 cell，对比地图当前实际状态，仅移除
// 确实已经大面积失效（>80% cells 不再是前沿）的簇。
//
// 注意：阈值设得保守（purge_fraction_ = 0.8），避免误删仍有探索
//       价值的"半覆盖"前沿。proximity_bonus 才是解决回退问题的核心。

// ====================  空间位置评分 ====================
double FrontierFinder::computeSpatialPositionScore(
    const Frontier& f,
    const Eigen::Vector3d& cur_pos) const {

  double d = (f.average_ - cur_pos).norm();
  const double rp = 5.0;
  double near_score = 0.0;
  if (d <= 6.0) {
    near_score = std::exp(-(d * d) / (rp * rp));
  }

  Eigen::Vector3d map_origin, map_size;
  edt_env_->sdf_map_->getRegion(map_origin, map_size);
  Eigen::Vector3d map_max = map_origin + map_size;

  double boundary_score = 0.0;
  if (map_size.x() > 1e-3 && map_size.y() > 1e-3 &&
      f.average_.x() >= map_origin.x() && f.average_.x() <= map_max.x() &&
      f.average_.y() >= map_origin.y() && f.average_.y() <= map_max.y()) {
    double edge_dist = std::min(
        std::min(f.average_.x() - map_origin.x(), map_max.x() - f.average_.x()),
        std::min(f.average_.y() - map_origin.y(), map_max.y() - f.average_.y()));
    double band = std::max(std::max(global_boundary_band_, local_closeout_radius_ + 0.5), 3.5);
    boundary_score =
        1.0 - std::min(std::max(edge_dist / std::max(band, 1e-3), 0.0), 1.0);
  }

  double score = 0.65 * near_score + 0.35 * boundary_score;
  return std::min(std::max(score, 0.0), 1.0);
}

// ====================  观测不足评分 ====================
double FrontierFinder::computeObservationDeficitScore(
    const Frontier& f) const {

  double recent_cov = f.coverage_overlap_;
  double persistent_cov = f.persistent_coverage_overlap_;
  double coverage_evidence =
      std::min(std::max(0.7 * recent_cov + 0.3 * persistent_cov, 0.0), 1.0);

  // 覆盖历史 + 近距离途经残留 + 后方泄漏/口袋证据
  double pass_evidence =
      std::min(std::max(f.pass_debt_score_, 0.0), 1.0);
  double pocket_evidence =
      std::min(std::max(f.bounded_pocket_score_, 0.0), 1.0);
  double leak_evidence =
      std::min(std::max(f.back_leak_ratio_, 0.0), 1.0);

  // 解释：
  // coverage_evidence: 历史上确实经过/覆盖过
  // pass_evidence:     经过后仍残留（补观测债务）
  // pocket/leak:       侧后方未知泄漏与口袋形态
  double score =
      0.35 * coverage_evidence +
      0.35 * pass_evidence +
      0.20 * pocket_evidence +
      0.10 * leak_evidence;

  return std::min(std::max(score, 0.0), 1.0);
}

// ====================  体素簇大小评分 ====================
double FrontierFinder::computeVoxelClusterSizeScore(
    const Frontier& f) const {

  int n = static_cast<int>(f.cells_.size());

  if (n < purge_min_cells_) {
    return 0.0;
  }

  if (n > 250) {
    return 0.0;
  }

  double score = 1.0 - std::min(double(n) / 250.0, 1.0);
  return std::min(std::max(score, 0.0), 1.0);
}

// ====================  距离当前路径间隔评分 ====================
double FrontierFinder::computeDistanceToRecentPath(
    const Eigen::Vector3d& pos) const {

  if (path_history_.empty()) {
    return 1e9;
  }

  if (path_history_.size() == 1) {
    return (pos - path_history_.front()).norm();
  }

  double best_dist = 1e9;

  for (size_t i = 1; i < path_history_.size(); ++i) {
    Eigen::Vector3d a = path_history_[i - 1];
    Eigen::Vector3d b = path_history_[i];

    Eigen::Vector3d ab = b - a;
    double len2 = ab.squaredNorm();

    if (len2 < 1e-6) {
      best_dist = std::min(best_dist, (pos - a).norm());
      continue;
    }

    double t = (pos - a).dot(ab) / len2;
    t = std::min(std::max(t, 0.0), 1.0);

    Eigen::Vector3d proj = a + t * ab;
    best_dist = std::min(best_dist, (pos - proj).norm());
  }

  return best_dist;
}

// ====================  路径间隔评分 ====================
double FrontierFinder::computePathGapScore(
    const Frontier& f) const {

  double d_path = computeDistanceToRecentPath(f.average_);

  if (d_path > inflate_radius_) {
    return 0.0;
  }

  double score =
      std::exp(-(d_path * d_path) / (inflate_radius_ * inflate_radius_));

  return std::min(std::max(score, 0.0), 1.0);
}

void FrontierFinder::computeBackLeakPocketEvidence(Frontier& f) {
  f.back_unknown_num_ = 0;
  f.back_obstacle_ratio_ = 0.0;
  f.back_leak_ratio_ = 0.0;
  f.bounded_pocket_score_ = 0.0;
  f.bounded_pocket_flag_ = false;

  // [MODIFIED BY CODEX]: 消融实验开关——关闭时跳过全部结构证据计算（δ_k 标记 + 结构证据项 + 差异化绕行代价均失效）
  if (!enable_structure_eval_) return;

  if (f.viewpoints_.empty()) return;

  // ============================================================
  // FAEP Bottom Ray: cs(k) = (hmax - hk) / hmax
  // 方向: 最佳视点 → 前沿平均位置
  // 起点: 前沿平均位置，向后沿同方向投射射线
  // hk = 射线触碰障碍物(occ/inflated)的距离
  // cs高 = hk小 = 后方近距离有墙 → 独立小区域(口袋)
  // ============================================================
  const double hmax_m = 5.0;
  const double res = edt_env_->sdf_map_->getResolution();
  const double hmax_voxels = hmax_m / res;
  const int K = std::min(6, static_cast<int>(f.viewpoints_.size()));

  double best_cs = 0.0;
  int best_hk_voxels = static_cast<int>(hmax_voxels);

  for (int i = 0; i < K; ++i) {
    double hk_vox = static_cast<double>(std::max(f.viewpoints_[i].unknown_num_, 1));
    double cs = std::min(std::max((hmax_voxels - hk_vox) / hmax_voxels, 0.0), 1.0);
    if (cs > best_cs) { best_cs = cs; best_hk_voxels = static_cast<int>(hk_vox); }
  }

  f.back_unknown_num_ = best_hk_voxels;
  f.back_leak_ratio_ = best_cs;

  // 障碍密度: frontier周边已观测障碍物比例 (辅助, 当Bottom Ray因墙未映射而失效时提供信号)
  f.back_obstacle_ratio_ = std::min(std::max(f.obstacle_density_, 0.0), 1.0);

  // [ABLATION]: 式(11) Cr = alpha*ray + (1-alpha)*density
  // alpha=0.7 时与改动前的 0.70/0.30 数值完全一致；alpha=1 只保留后向射线，
  // alpha=0 只保留邻域障碍密度。
  const double alpha = std::min(std::max(structure_alpha_, 0.0), 1.0);
  const double w_ray = alpha;
  const double w_density = 1.0 - alpha;

  // 综合口袋得分: 后向射线 + 邻域障碍密度
  f.bounded_pocket_score_ = std::min(std::max(
      w_ray * best_cs + w_density * f.back_obstacle_ratio_, 0.0), 1.0);

  // 口袋标记: 两个判据都跟随权重，保证 alpha=1 时密度项完全不参与、
  // alpha=0 时后向射线项完全不参与（否则单项消融不彻底）。
  const bool ray_flag = (w_ray > 1e-6) && (best_cs >= 0.40);
  const bool density_flag = (w_density > 1e-6) && (f.back_obstacle_ratio_ >= 0.50);
  f.bounded_pocket_flag_ = ray_flag || density_flag;
}

// ====================  体素簇几何特征评分 ====================
double FrontierFinder::computeGeometryFeatureScore(
    const Frontier& f) const {

  double score = 0.6 * f.concavity_ + 0.4 * f.narrowness_;
  return std::min(std::max(score, 0.0), 1.0);
}

// ====================  综合自适应体素遗忘评分 ====================
double FrontierFinder::computeAdaptiveVoxelForgetScore(
    const Frontier& f,
    const Eigen::Vector3d& cur_pos) const {

  // 分项语义：
  // P (Position):   当前空间位置是否“顺路”
  // O (Observation):观测不足证据（coverage + pass_debt + pocket/leak）
  // S (Size):       规模是否适合局部补观测
  // R (PathGap):    与近期轨迹的邻近程度
  // G (Geometry):   几何形态是否容易形成残留
  double P = computeSpatialPositionScore(f, cur_pos);
  double O = computeObservationDeficitScore(f);
  double S = computeVoxelClusterSizeScore(f);
  double R = computePathGapScore(f);
  double G = computeGeometryFeatureScore(f);

  // 权重设计意图：
  // - O 权重最高：补观测动作的本质是“确实存在观测债务”
  // - P/S/R 作为可执行性与顺路性约束，避免把全局目标硬拉回局部
  // - G 用于对细长/凹陷等高残留形态做轻微偏置
  double score =
      0.25 * P +
      0.30 * O +
      0.20 * S +
      0.15 * R +
      0.10 * G;

  // 评分被夹紧在 [0,1]，便于：
  // 1) 与 adaptive_voxel_thresh_ 直接比较
  // 2) 在起点降价中按比例映射到 bonus 上限
  return std::min(std::max(score, 0.0), 1.0);
}

double FrontierFinder::computePassDebtScore(
    Frontier& f,
    const Eigen::Vector3d& cur_pos) {

  std::string sig = f.stable_sig_;
  if (sig.empty()) sig = makeFrontierSignature(f);

  double cur_dist = (f.average_ - cur_pos).norm();

  // 用 signature 维持跨轮次统计，不因 frontier 容器重排丢历史：
  // - min_pass_dist: 历史最小距离
  // - near_pass_cnt: 进入局部半径次数
  // - last_near_round: 最近一次靠近轮次
  auto it = sig_min_pass_dist_.find(sig);
  if (it == sig_min_pass_dist_.end()) {
    sig_min_pass_dist_[sig] = cur_dist;
  } else {
    sig_min_pass_dist_[sig] = std::min(sig_min_pass_dist_[sig], cur_dist);
  }

  const double near_radius = local_closeout_radius_;

  if (cur_dist <= near_radius) {
    sig_near_pass_cnt_[sig] += 1;
    sig_last_near_round_[sig] = frontier_track_round_;
  }

  f.min_pass_dist_ = sig_min_pass_dist_[sig];
  f.near_pass_cnt_ = sig_near_pass_cnt_[sig];
  f.last_near_round_ = sig_last_near_round_[sig];

  // 从未真正靠近过，pass_debt 直接为 0（不属于“途经未清理”）。
  if (f.min_pass_dist_ > near_radius + 0.5) {
    f.pass_debt_score_ = 0.0;
    return 0.0;
  }

  // dist_score: 历史最小距离越近越高（指数衰减）
  double dist_score =
      std::exp(-(f.min_pass_dist_ * f.min_pass_dist_) /
               (near_radius * near_radius));

  // pass_score: 靠近次数越多越高（最多按 3 次封顶）
  double pass_score =
      std::min(double(f.near_pass_cnt_) / 3.0, 1.0);

  int cell_num = static_cast<int>(f.cells_.size());
  // size_score: 小簇更适合“顺路清理”，大簇应交给主探索流程
  double size_score = 0.0;

  if (cell_num >= purge_min_cells_ && cell_num <= 250) {
    size_score = 1.0 - std::min(double(cell_num) / 250.0, 1.0);
  }

  // geom_score: 结合凹性/狭长性，估计“容易残留”的几何形态
  double geom_score =
      std::min(std::max(0.6 * f.concavity_ + 0.4 * f.narrowness_, 0.0), 1.0);

  // 组合为 pass_debt_score（0~1），作为综合评分与门控的重要证据项
  f.pass_debt_score_ =
      0.45 * dist_score +
      0.25 * pass_score +
      0.20 * size_score +
      0.10 * geom_score;

  f.pass_debt_score_ =
      std::min(std::max(f.pass_debt_score_, 0.0), 1.0);

  return f.pass_debt_score_;
}

bool FrontierFinder::isFrontierCovered() {

  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  auto checkChanges = [&](const list<Frontier>& frontiers) {
    for (auto ftr : frontiers) {
      if (!haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max)) continue;
      const int change_thresh = min_view_finish_fraction_ * ftr.cells_.size();
      int change_num = 0;
      for (auto cell : ftr.cells_) {
        Eigen::Vector3i idx;
        edt_env_->sdf_map_->posToIndex(cell, idx);
        if (!(knownfree(idx) && isNeighborUnknown(idx)) && ++change_num >= change_thresh)
          return true;
      }
    }
    return false;
  };

  if (checkChanges(frontiers_) || checkChanges(dormant_frontiers_)) return true;

  return false;
}

bool FrontierFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(
    const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& cluster) {
  percep_utils_->setPose(pos, yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) visib_num += 1;
  }
  return visib_num;
}

void FrontierFinder::countVisibleFrontierAndUnknown(
    const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& cluster,
    int& frontier_num, int& visible_unknown_num) {
  percep_utils_->setPose(pos, yaw);
  frontier_num = 0;
  visible_unknown_num = 0;

  // Different frontier rays can pass through the same unknown voxel. Count a
  // voxel once so N_unknown remains a geometric observation-gain measure,
  // rather than a function of frontier downsampling density.
  std::unordered_set<int> counted_unknown;
  Eigen::Vector3i idx;
  const double max_dist = percep_utils_->getMaxDist();

  for (const auto& cell : cluster) {
    if (!percep_utils_->insideFOV(cell)) continue;

    raycaster_->input(cell, pos);
    bool visible = true;
    while (raycaster_->nextId(idx)) {
      if (!edt_env_->sdf_map_->isInMap(idx) ||
          edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visible = false;
        break;
      }
    }
    if (!visible) continue;

    ++frontier_num;

    // Algorithm 1 in the cited viewpoint paper continues the visible ray from
    // the frontier to the sensor-range endpoint and counts unknown voxels until
    // an obstacle or the map boundary is reached.
    const Vector3d ray_dir = (cell - pos).normalized();
    const Vector3d ray_end = pos + max_dist * ray_dir;
    raycaster_->input(cell, ray_end);
    while (raycaster_->nextId(idx)) {
      if (!edt_env_->sdf_map_->isInMap(idx)) break;
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::OCCUPIED) {
        break;
      }
      if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN &&
          counted_unknown.insert(toadr(idx)).second) {
        ++visible_unknown_num;
      }
    }
  }
}

void FrontierFinder::downsample(
    const vector<Eigen::Vector3d>& cluster_in, vector<Eigen::Vector3d>& cluster_out) {
  // downsamping cluster
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : cluster_in)
    cloud->points.emplace_back(cell[0], cell[1], cell[2]);

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);

  cluster_out.clear();
  for (auto pt : cloudf->points)
    cluster_out.emplace_back(pt.x, pt.y, pt.z);
}

void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI)
    yaw += 2 * M_PI;
  while (yaw > M_PI)
    yaw -= 2 * M_PI;
}

Eigen::Vector3i FrontierFinder::searchClearVoxel(const Eigen::Vector3i& pt) {
  queue<Eigen::Vector3i> init_que;
  vector<Eigen::Vector3i> nbrs;
  Eigen::Vector3i cur, start_idx;
  init_que.push(pt);
  // visited_flag_[toadr(pt)] = 1;

  while (!init_que.empty()) {
    cur = init_que.front();
    init_que.pop();
    if (knownfree(cur)) {
      start_idx = cur;
      break;
    }

    nbrs = sixNeighbors(cur);
    for (auto nbr : nbrs) {
      // int adr = toadr(nbr);
      // if (visited_flag_[adr] == 0)
      // {
      //   init_que.push(nbr);
      //   visited_flag_[adr] = 1;
      // }
    }
  }
  return start_idx;
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel) {
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN) return true;
  }
  return false;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) const {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

}  // namespace fast_planner
