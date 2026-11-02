#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <lkh_tsp_solver/lkh_interface.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <active_perception/frontier_finder.h>
#include <active_perception/grid_planner.h>
#include <plan_manage/planner_manager.h>

#include <exploration_manager/expl_data.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <utility>

using namespace Eigen;

namespace fast_planner {
// SECTION interfaces for setup and query

/**
 * FastExplorationManager 类 - 负责自主探索规划
 * 主要功能：
 * 1. 搜索前沿点并聚类
 * 2. 计算每个前沿簇的最佳观测点
 * 3. 规划全局探索路径（使用 TSP 算法）
 * 4. 优化局部探索路径
 * 5. 生成轨迹并执行探索任务
 */
FastExplorationManager::FastExplorationManager() {
}

/**
 * 析构函数 - 释放资源
 */
FastExplorationManager::~FastExplorationManager() {
  if (grid_select_log_file_.is_open()) {
    grid_select_log_file_.close();
  }
  if (local_tsp_cost_log_file_.is_open()) {
    local_tsp_cost_log_file_.close();
  }
  if (decision_trace_log_file_.is_open()) {
    decision_trace_log_file_.close();
  }
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

/**
 * 初始化函数 - 设置参数和创建模块实例
 * @param nh ROS 节点句柄
 */
void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  // 初始化规划管理器
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);
  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;
  
  // 初始化前沿发现器
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));

  // 初始化网格规划器（全局层）
  grid_planner_.reset(new GridPlanner(edt_environment_, nh));

  // 初始化探索数据和参数
  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  // 读取探索相关参数
  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);
  nh.param("exploration/grid_switch_lookahead", ep_->grid_switch_lookahead_, 4);
  nh.param(
      "exploration/grid_switch_order_penalty", ep_->grid_switch_order_penalty_, 2.0);
  nh.param("exploration/grid_switch_cost_margin", ep_->grid_switch_cost_margin_, 0.8);
  nh.param(
      "exploration/grid_switch_unreachable_penalty",
      ep_->grid_switch_unreachable_penalty_, 8.0);
  nh.param(
      "exploration/grid_switch_max_path_length",
      ep_->grid_switch_max_path_length_, 14.0);
  nh.param(
      "exploration/grid_switch_max_path_stretch",
      ep_->grid_switch_max_path_stretch_, 2.4);
  nh.param("exploration/grid_neighbor_hops", ep_->grid_neighbor_hops_, 2);
  nh.param("exploration/grid_target_hold_rounds", ep_->grid_target_hold_rounds_, 8);
  nh.param("exploration/grid_target_switch_margin", ep_->grid_target_switch_margin_, 1.5);
  nh.param("exploration/grid_pick_w_info", ep_->grid_pick_w_info_, 3.0);
  nh.param("exploration/grid_pick_w_motion", ep_->grid_pick_w_motion_, 1.0);
  nh.param("exploration/grid_pick_w_neighbor", ep_->grid_pick_w_neighbor_, 3.0);
  nh.param("exploration/fine_grid_switch_margin", ep_->fine_grid_switch_margin_, 2.0);
  nh.param("exploration/current_grid_priority_max_path",
           ep_->current_grid_priority_max_path_, 8.0);
  nh.param("exploration/current_grid_priority_margin",
           ep_->current_grid_priority_margin_, 6.0);
  nh.param("exploration/current_grid_priority_path_margin",
           ep_->current_grid_priority_path_margin_, 2.0);
  nh.param("exploration/root_coverage_priority_weight",
           ep_->root_coverage_priority_weight_, 0.8);
  nh.param("exploration/local_neighbor_frontier_max_path",
           ep_->local_neighbor_frontier_max_path_, 10.0);
  nh.param("exploration/local_reobserve_radius",
           ep_->local_reobserve_radius_, 6.5);
  nh.param("exploration/local_reobserve_max_path",
           ep_->local_reobserve_max_path_, 9.0);
  nh.param("exploration/local_recent_grid_reobserve_max_path",
           ep_->local_recent_grid_reobserve_max_path_, 11.0);
  nh.param("exploration/local_pocket_weight", ep_->local_pocket_weight_, 6.0);  // [MODIFIED BY CODEX]: 恢复独立小区域(口袋)收尾奖励权重为 6.0
  nh.param("exploration/local_view_uncovered_overlap_thresh",
           ep_->local_view_uncovered_overlap_thresh_, 0.35);
  nh.param("exploration/local_view_covered_overlap_thresh",
           ep_->local_view_covered_overlap_thresh_, 0.35);
  nh.param("exploration/local_view_block_cross_grid",
           ep_->local_view_block_cross_grid_, true);
  ep_->root_hard_lock_ = false;
  nh.param("exploration/root_hard_lock_max_unreachable_rounds",
           ep_->root_hard_lock_max_unreachable_rounds_, 6);
  nh.param("exploration/root_complete_confirm_rounds",
           ep_->root_complete_confirm_rounds_, 3);
  nh.param("exploration/enable_root_soft_release",
           ep_->enable_root_soft_release_, false);
  nh.param("exploration/root_soft_release_coverage_ratio",
           ep_->root_soft_release_coverage_ratio_, 0.90);
  nh.param("exploration/root_soft_release_min_rounds",
           ep_->root_soft_release_min_rounds_, 12);

  // Dual-threshold grid switching + timeout + batch cleanup
  nh.param("exploration/root_switch_frontier_ratio",
           root_switch_frontier_ratio_, 0.12);
  nh.param("exploration/root_switch_coverage_ratio",
           root_switch_coverage_ratio_, 0.95);
  nh.param("exploration/root_component_release_ratio",
           root_component_release_ratio_, 0.06);
  nh.param("exploration/active_component_release_ratio",
           active_component_release_ratio_, 0.15);
  nh.param("exploration/root_timeout_seconds",
           root_timeout_seconds_, 150.0);
  nh.param("exploration/batch_cleanup_interval",
           batch_cleanup_interval_, 3);
  nh.param("exploration/fine_grid_cleanup_interval",
           fine_grid_cleanup_interval_, 3);
  nh.param("exploration/fine_grid_cleanup_hold_rounds",
           fine_grid_cleanup_hold_rounds_, 0);
  nh.param("exploration/fine_grid_cleanup_cov_thresh",
           fine_grid_cleanup_cov_thresh_, 0.90);
  nh.param("exploration/root_soft_release_cooldown_rounds",
           ep_->root_soft_release_cooldown_rounds_, 25);
  nh.param("exploration/lead_frontier_stall_rounds", ep_->lead_frontier_stall_rounds_, 4);
  nh.param("exploration/lead_frontier_stall_dist", ep_->lead_frontier_stall_dist_, 1.2);
  nh.param("exploration/residual_defer_pass_cnt",
           ep_->residual_defer_pass_cnt_, 3);
  nh.param("exploration/residual_defer_max_coverage",
           ep_->residual_defer_max_coverage_, 0.30);
  nh.param("exploration/residual_defer_max_path",
           ep_->residual_defer_max_path_, 7.5);
  nh.param("exploration/local_unfinished_grid_priority_margin",
           ep_->local_unfinished_grid_priority_margin_, 3.0);
  nh.param("exploration/local_unfinished_grid_max_path",
           ep_->local_unfinished_grid_max_path_, 7.0);
  nh.param("exploration/recent_path_grid_limit", ep_->recent_path_grid_limit_, 8);
  recent_path_grid_limit_ = std::max(1, ep_->recent_path_grid_limit_);
  // Coarse-grid stickiness: prevent oscillation between fine grids in same root
  nh.param("exploration/same_root_fine_grid_margin",
           ep_->same_root_fine_grid_margin_, 3.5);
  nh.param("exploration/same_root_backtrack_penalty",
           ep_->same_root_backtrack_penalty_, 7.0);
  nh.param("exploration/fine_grid_cov_complete_thresh",
           ep_->fine_grid_cov_complete_thresh_, 0.90);
  nh.param("exploration/coarse_dir_momentum_weight",
           ep_->coarse_dir_momentum_weight_, 4.0);
  nh.param("exploration/enable_grid_select_log", enable_grid_select_log_, true);
  nh.param("exploration/grid_select_log_path", grid_select_log_path_,
           std::string("/home/djg/Data_debug/grid_select_log.csv"));
  nh.param("exploration/enable_local_tsp_cost_log", enable_local_tsp_cost_log_, true);
  nh.param("exploration/local_tsp_cost_log_path", local_tsp_cost_log_path_,
           std::string("/home/djg/Data_debug/local_tsp_cost_log.csv"));
  nh.param("exploration/enable_decision_trace_log", enable_decision_trace_log_, true);
  nh.param("exploration/decision_trace_log_path", decision_trace_log_path_,
           std::string("/home/djg/Data_debug/decision_trace_log.csv"));
  nh.param("exploration/local_tsp_block_cost", local_tsp_block_cost_, 5000.0);
  nh.param("exploration/local_tsp_path_sample_step", local_tsp_path_sample_step_, 0.2);
  nh.param("exploration/local_tsp_first_cost_ratio",
           local_tsp_first_cost_ratio_, 1.3);
  nh.param("exploration/local_tsp_first_cost_margin",
           local_tsp_first_cost_margin_, 1.0);
  // [MODIFIED BY CODEX]: Local look-ahead is deliberately small.  It is a
  // planning aid, never a queued flight command.
  nh.param("exploration/local_tsp_candidate_num", local_tsp_candidate_num_, 10);
  // [ABLATION]: 下一待访问网格方向引导总开关（去掉方向引导配置用）
  nh.param("exploration/use_grid_guidance", use_grid_guidance_, true);

  if (enable_grid_select_log_) {
    grid_select_log_file_.open(grid_select_log_path_, std::ios::out);
    if (grid_select_log_file_.is_open()) {
      grid_select_log_file_
          << "round,stamp,candidate_order,grid_idx,grid_id,expanded_grid_count,"
          << "frontier_count,reachable_frontier_count,best_motion_cost,best_path_len,"
          << "center_dist,score,grid_status,unknown_voxels,frontier_voxels,"
          << "coverage_ratio,path_covered,selected_order,selected,"
          << "agg_unknown_voxels,agg_frontier_voxels,adjacency_hops,"
          << "score_motion,score_info_norm,score_neighbor,"
          << "own_reachable_frontier_count,own_best_motion_cost,own_best_path_len,"
          << "is_current_grid,is_active_grid,frontier_stage,root_lock,root_unreachable_rounds,"
          << "grid_anchor_path_len,base_order,base_grid_id,base_score,"
          << "best_order,best_grid_id,best_score,chosen_reason\n";
      grid_select_log_file_.flush();
      ROS_WARN_STREAM("[GridSelectLog] Logging to " << grid_select_log_path_);
    } else {
      ROS_ERROR_STREAM("[GridSelectLog] Failed to open file: " << grid_select_log_path_);
    }
  }

  if (enable_local_tsp_cost_log_) {
    local_tsp_cost_log_file_.open(local_tsp_cost_log_path_, std::ios::out);
    if (local_tsp_cost_log_file_.is_open()) {
      local_tsp_cost_log_file_
          << "round,stamp,edge_type,from_frontier_id,to_frontier_id,"
          << "from_grid_id,to_grid_id,raw_cost,final_cost,path_length,"
          << "euclidean_distance,path_stretch,path_excess,yaw_delta,"
          << "astar_reachable,local_path_ok,blocked_reason,path_reachable,"
          << "path_inside_active_grid,active_grid_id,active_root_grid_id,tour_order,selected_first,"
          << "path_overlap,coverage_overlap,filtered_coverage_overlap,"
          << "min_pass_dist,near_pass_cnt,in_local_band,allow_closeout,"
          << "lkh_sequence,start_astar_length,dijkstra_prefix_order,"
          << "in_dijkstra,dijkstra_reason\n";
      local_tsp_cost_log_file_.flush();
      ROS_WARN_STREAM("[LocalTspCostLog] Logging to " << local_tsp_cost_log_path_);
    } else {
      ROS_ERROR_STREAM("[LocalTspCostLog] Failed to open file: "
                       << local_tsp_cost_log_path_);
    }
  }

  if (enable_decision_trace_log_) {
    decision_trace_log_file_.open(decision_trace_log_path_, std::ios::out);
    if (decision_trace_log_file_.is_open()) {
      decision_trace_log_file_
          << "round,stamp,stage,result,active_grid_id,active_root_grid_id,"
          << "active_component_grid_id,active_component_local_id,active_component_locked,"
          << "root_complete,root_timeout,root_ready_to_switch,root_unreachable_rounds,pending_root_count,"
          << "soft_released_root_id,grid_count,grid_frontier_count,refined_id_count,"
          << "refined_point_count,backup_goal_count,selected_source,selected_frontier_id,"
          << "selected_frontier_grid_id,selected_frontier_root_id,next_goal_x,next_goal_y,"
          << "next_goal_z,next_goal_yaw,path_len,time_lb,reachable_view_count,"
          << "sel_motion_cost,sel_detour_cost,sel_pocket_score,sel_gradient,sel_backward_penalty,"
          << "sel_total_score,candidate_count,"
          // [ABLATION] L1/L2 机制证据 + 阶段耗时
          << "straight_dist,path_stretch,crosses_wall,same_grid_other_component,"
          << "ccl_candidates_before,ccl_filtered_num,"
          << "ray_term,density_term,structure_score,pocket_flag,"
          << "gradient_term,has_next_grid,tsp_order_selected,"
          << "alt_changed_by_structure,alt_changed_by_guidance,"
          << "plan_ms_total,plan_ms_grid,plan_ms_reach,plan_ms_rank\n";
      decision_trace_log_file_.flush();
      ROS_WARN_STREAM("[DecisionTraceLog] Logging to " << decision_trace_log_path_);
    } else {
      ROS_ERROR_STREAM("[DecisionTraceLog] Failed to open file: "
                       << decision_trace_log_path_);
    }
  }

  // 读取视图节点相关参数
  nh.param("exploration/vm", ViewNode::vm_, -1.0);
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);

  // 初始化 A* 搜索和射线投射
  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, edt_environment_);
  ViewNode::map_ = sdf_map_;

  double resolution_ = sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  sdf_map_->getRegion(origin, size);
  ViewNode::caster_.reset(new RayCaster);
  ViewNode::caster_->setParams(resolution_, origin);

  // 设置路径搜索参数
  planner_manager_->path_finder_->lambda_heu_ = 1.0;
  planner_manager_->path_finder_->max_search_time_ = 1.0;

  // 初始化 TSP 参数文件
  ofstream par_file(ep_->tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 1\n";

  // [MODIFIED BY CODEX]: The local frontier LKH instance must not overwrite
  // the global grid TSP instance in the same planning cycle.
  ofstream local_par_file(ep_->tsp_dir_ + "/local_frontier.par");
  local_par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/local_frontier.tsp\n";
  local_par_file << "GAIN23 = NO\n";
  local_par_file << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/local_frontier.txt\n";
  local_par_file << "RUNS = 1\n";
  local_par_file.close();

  // [ABLATION] 把本配置的开关状态落盘，供“开关是否真的进入运行”的核对
  {
    std::string dir = decision_trace_log_path_;
    const size_t slash = dir.find_last_of('/');
    dir = (slash == std::string::npos) ? std::string(".") : dir.substr(0, slash);
    const std::string flags_path = dir + "/ablation_flags.json";
    std::ofstream flags(flags_path.c_str());
    if (flags.is_open()) {
      flags << "{\n"
            << "  \"frontier_alpha\": " << frontier_finder_->structureAlpha() << ",\n"
            << "  \"use_ccl_for_filter\": "
            << (grid_planner_->cclFilterEnabled() ? "true" : "false") << ",\n"
            << "  \"use_ccl_for_center\": "
            << (grid_planner_->cclCenterEnabled() ? "true" : "false") << ",\n"
            << "  \"slice_dim\": \"" << grid_planner_->sliceDim() << "\",\n"
            << "  \"use_grid_guidance\": " << (use_grid_guidance_ ? "true" : "false") << "\n"
            << "}\n";
      flags.close();
      ROS_WARN_STREAM("[Ablation] flags -> " << flags_path);
    }
  }
}

/**
 * 规划探索运动
 * @param pos 当前位置
 * @param vel 当前速度
 * @param acc 当前加速度
 * @param yaw 当前偏航角
 * @return 探索结果状态
 */
int FastExplorationManager::planExploreMotion(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw) {
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;
  // [ABLATION] 整轮规划起点，用于 decision_trace 的 plan_ms_total
  const ros::Time plan_begin_time = t1;
  // [ABLATION] 阶段耗时与“是否存在下一网格”，在下方赋值后由 trace 写出
  double plan_ms_grid_ = -1.0;
  double plan_ms_reach_ = -1.0;
  double plan_ms_rank_ = -1.0;
  int has_next_grid_flag_ = 0;
  ed_->views_.clear();
  ed_->global_tour_.clear();

  std::cout << "start pos: " << pos.transpose() << ", vel: " << vel.transpose()
            << ", acc: " << acc.transpose() << std::endl;

  // 1、搜索前沿点并聚类
  frontier_finder_->searchFrontiers();

  double frontier_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // 2、计算需要访问的前沿簇并获取信息
  frontier_finder_->computeFrontiersToVisit();
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);

  // 3、获取每个前沿簇的最佳视点
  if (!ed_->frontiers_.empty()) {
    frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_->yaws_, ed_->averages_);
    for (int i = 0; i < ed_->points_.size(); ++i)
      ed_->views_.push_back(
          ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));
  }

  // [MODIFIED BY CODEX]: Keep FUEL-original's ordering: first expose the
  // current top viewpoints, then update the incremental frontier graph used
  // by the ATSP.  Old-old edges remain cached; only old-new/new-new edges use
  // the 1 ms bounded estimator in this planning round.
  if (!ed_->frontiers_.empty()) frontier_finder_->updateFrontierCostMatrix();

  // 将前沿面映射到网格
  // [ABLATION] 网格层（含 CCL）耗时
  const ros::Time t_grid_begin = ros::Time::now();
  grid_planner_->updateGrids(pos);
  grid_planner_->inputFrontiersDetailed(ed_->averages_, ed_->points_);
  ed_->grid_ids_.clear();
  ed_->grid_centers_.clear();
  // [MODIFIED BY CODEX]: The global TSP is a predicted coverage order.  It
  // must contain active unknown grids even before they expose a frontier.
  grid_planner_->getGlobalTspGridCenters(ed_->grid_centers_, ed_->grid_ids_);

  // RUSH: update subregion states and compute subregion TSP tour
  grid_planner_->updateSubRegionStates(pos);
  grid_planner_->computeSubRegionTSP(pos);
  plan_ms_grid_ = (ros::Time::now() - t_grid_begin).toSec() * 1000.0;

  const int cur_gid_initial = grid_planner_->getGridContainingPoint(pos);
  if (active_root_grid_id_ < 0 && cur_gid_initial >= 0 &&
      cur_gid_initial < static_cast<int>(grid_planner_->all_grids_.size())) {
    // RUSH: use subregion ID (depth-1), not depth-0 root ID
    active_root_grid_id_ = grid_planner_->getSubRegionId(cur_gid_initial);
    active_root_enter_time_ = ros::Time::now();
    active_root_timed_out_ = false;
    active_root_initial_frontier_count_ = 0;
    ROS_ERROR("[RUSH] Exploration Manager: initial active subregion = %d",
              active_root_grid_id_);
  }

  // RUSH: re-anchoring only for the depth-0 root grid.
  // The depth-0 root (40×40 m) is created at init and subdivides into
  // depth-1 subregions.  When it becomes inactive, active_root_grid_id_
  // must be re-anchored to the drone's actual depth-1 subregion.
  //
  // CRITICAL: do NOT re-anchor depth-1 subregions that have subdivided.
  // Depth-1 subregions remain valid even when inactive because their
  // depth-2 children still belong to them via getSubRegionId.
  // Re-anchoring away from a valid subdivided subregion causes a ping-pong:
  //   re-anchor → complete → switch → re-anchor → …
  if (active_root_grid_id_ >= 0 &&
      active_root_grid_id_ < static_cast<int>(grid_planner_->all_grids_.size()) &&
      !grid_planner_->all_grids_[active_root_grid_id_].active_ &&
      grid_planner_->all_grids_[active_root_grid_id_].depth_ == 0) {
    // Depth-0 root was subdivided away — re-anchor to drone position.
    const int new_sr = grid_planner_->getSubRegionForPoint(pos);
    if (new_sr >= 0 && new_sr != active_root_grid_id_) {
      ROS_ERROR("[RUSH] Root %d subdivided, re-anchoring to subregion %d",
                active_root_grid_id_, new_sr);
      active_root_grid_id_ = new_sr;
      active_root_enter_time_ = ros::Time::now();
      active_root_timed_out_ = false;
      grid_planner_->setActiveSubRegion(new_sr);  // sync grid_planner state
      grid_planner_->computeSubRegionTSP(pos);
    }
  }

  // RUSH: subregion auto-switch — must happen BEFORE grid_ids filtering
  // so the new subregion's grids are used.
  if (active_root_grid_id_ >= 0 &&
      grid_planner_->isSubRegionComplete(active_root_grid_id_)) {
    const int next_sr = grid_planner_->getNextSubRegion(active_root_grid_id_);
    if (next_sr >= 0) {
      ROS_WARN("[RUSH] Subregion %d complete, switching to subregion %d",
               active_root_grid_id_, next_sr);
      active_root_grid_id_ = next_sr;
      active_root_enter_time_ = ros::Time::now();
      active_root_timed_out_ = false;
      root_hold_until_round_ = grid_decision_round_ + 8;
      grid_planner_->setActiveSubRegion(next_sr);  // sync grid_planner state
      grid_planner_->computeSubRegionTSP(pos);
    } else {
      ROS_WARN("[RUSH] Subregion %d complete, no more pending subregions",
               active_root_grid_id_);
    }
  }

  // RUSH soft-boundary mode: do NOT hard-filter grid_ids_ to the active
  // subregion.  Hard filtering forces the drone to exhaust every frontier
  // in the current subregion before even considering adjacent ones —
  // causing long detours when nearer frontiers sit just across a boundary.
  //
  // Instead we keep ALL grids and use the subregion TSP as a *soft*
  // gradient bias in local frontier sorting (w_gradient, below).  The
  // drone naturally prefers closer frontiers (motion cost) while
  // drifting toward the next TSP-ordered subregion when frontiers are
  // equally good.
  //
  // The auto-switch still fires when a subregion is structurally
  // complete, but it only updates active_root_grid_id_ (for logging and
  // TSP re-computation) — it does NOT restrict the frontier set.
  bool root_incomplete_this_round = false;

  const int cur_gid = grid_planner_->getGridContainingPoint(pos);
  if (cur_gid >= 0 && cur_gid < static_cast<int>(grid_planner_->all_grids_.size())) {
    recent_path_grid_ids_.erase(
        std::remove(recent_path_grid_ids_.begin(), recent_path_grid_ids_.end(), cur_gid),
        recent_path_grid_ids_.end());
    recent_path_grid_ids_.insert(recent_path_grid_ids_.begin(), cur_gid);
    if (static_cast<int>(recent_path_grid_ids_.size()) > recent_path_grid_limit_) {
      recent_path_grid_ids_.resize(recent_path_grid_limit_);
    }
  }

  // RUSH: subregion completion replaces root completeness
  root_incomplete_this_round =
      active_root_grid_id_ >= 0 &&
      !grid_planner_->isSubRegionComplete(active_root_grid_id_);
  if (active_root_grid_id_ >= 0) {
    if (root_incomplete_this_round) {
      ++active_root_incomplete_rounds_;
      active_root_complete_rounds_ = 0;
    } else {
      ++active_root_complete_rounds_;
      active_root_incomplete_rounds_ = 0;
    }
  }
  // Component/root locks removed — using simplified TSP-guided selection
  // RUSH: disable the fixed root sweep lock.  Subregions are dynamic and
  // the non-closed TSP naturally guides the drone to the next subregion.
  // Hard locks on artificial boundaries are no longer needed.
  const bool active_root_sweep_locked = false;
  // RUSH + V14: when to restrict fallback stages (4/5/6 & rescue) to the
  // active root/subregion.  In fixed-root mode this is gated by sweep_locked;
  // in RUSH mode it is always true when a subregion is active.
  // RUSH: soft boundaries — fallback/rescue stages should consider
  // frontiers in ALL subregions, not just the "active" one.  Hard
  // filtering here would force the drone to return to the active
  // subregion even when closer frontiers exist elsewhere.
  const bool restrict_fallback_to_active_root =
      active_root_sweep_locked ||
      (!grid_planner_->isRushMode() && active_root_grid_id_ >= 0);
  const bool rush_soft_layout = grid_planner_->isRushMode();
  // NOTE: root_scope_locked is temporarily disabled (always false).
  // The 10+ root-filtering branches in the 6-stage local frontier selection
  // were dead code when root_scope_locked was hardcoded to false before V1.
  // Enabling them (V1.1) exposed cascading filter failures: Stages 0-2 clear
  // grid_frontier_ids_ and fail to repopulate when all frontiers are filtered
  // by root-scope + component + path gates, leading to NO_FRONTIER → FINISH.
  // TODO: re-enable root_scope_locked incrementally after verifying each
  // root-filtering branch independently.
  const bool root_scope_locked = false;
  // active_root_sweep_locked computed above, will be used when
  // root_scope_locked is re-enabled after per-branch verification.
  if (fine_grid_cleanup_hold_rounds_ > 0) {
    --fine_grid_cleanup_hold_rounds_;
  }
  double view_time = (ros::Time::now() - t1).toSec();
  ROS_WARN(
      "Frontier: %d, t: %lf, viewpoint: %d, t: %lf, active_grids: %d",
      (int)ed_->frontiers_.size(), frontier_time, (int)ed_->points_.size(), view_time,
      (int)ed_->grid_ids_.size());
  
  // 4、全局层：网格TSP + 局部层：前沿细化
  // Keep failure traces deterministic.  These values are also written when
  // no executable frontier is found, so never serialize uninitialized data.
  Vector3d next_pos = pos;
  double next_yaw = yaw[0];
  int selected_frontier_id = -1;
  int selected_frontier_grid_id = -1;
  int selected_frontier_root_id = -1;
  std::string selected_source = "unknown";
  // Scoring detail for decision trace log
  double sel_motion_cost_ = -1.0, sel_detour_cost_ = -1.0;
  double sel_pocket_score_ = -1.0, sel_gradient_ = -1.0, sel_backward_penalty_ = -1.0;
  double sel_total_score_ = -1.0;
  int candidate_count_ = 0;

  // [ABLATION] 本轮机制证据（L1/L2），默认 -1 表示不可用
  double sel_straight_dist_ = -1.0;
  double sel_path_stretch_ = -1.0;
  int sel_crosses_wall_ = -1;
  int sel_same_grid_other_component_ = -1;
  double sel_ray_term_ = -1.0;
  double sel_density_term_ = -1.0;
  double sel_structure_score_ = -1.0;
  int sel_pocket_flag_ = -1;
  int sel_tsp_order_ = -1;
  int ccl_candidates_before_ = -1;
  int ccl_filtered_num = 0;
  int alt_changed_by_structure_ = -1;
  int alt_changed_by_guidance_ = -1;
  auto write_decision_trace = [&](int stage, int result_code, double path_len,
                                  double time_lb, int reachable_view_count) {
    if (!enable_decision_trace_log_ || !decision_trace_log_file_.is_open()) return;
    ++decision_trace_log_round_;
    const int grid_count = static_cast<int>(ed_->grid_ids_.size());
    const int grid_frontier_count = static_cast<int>(ed_->grid_frontier_ids_.size());
    const int refined_id_count = static_cast<int>(ed_->refined_ids_.size());
    const int refined_point_count = static_cast<int>(ed_->refined_points_.size());
    const int backup_goal_count = reachable_view_count;
    if (selected_frontier_id >= 0 &&
        selected_frontier_id < static_cast<int>(ed_->points_.size())) {
      selected_frontier_grid_id =
          grid_planner_->getGridContainingPoint(ed_->points_[selected_frontier_id]);
      selected_frontier_root_id =
          grid_planner_->getRootGridId(selected_frontier_grid_id);
    } else {
      selected_frontier_grid_id = -1;
      selected_frontier_root_id = -1;
    }
    const bool root_complete =
        active_root_grid_id_ >= 0 &&
        grid_planner_ != nullptr &&
        grid_planner_->isRootGridComplete(active_root_grid_id_);
    decision_trace_log_file_
        << decision_trace_log_round_ << "," << ros::Time::now().toSec() << ","
        << stage << "," << result_code << ","
        << active_grid_id_ << "," << active_root_grid_id_ << ","
        << active_component_grid_id_ << "," << active_component_local_id_ << ","
        << 0 << "," << root_complete << ","
        << 0 << "," << 0 << ","
        << root_unreachable_rounds_ << ","
        << static_cast<int>(pending_root_grid_ids_.size()) << ","
        << soft_released_root_id_ << ","
        << grid_count << "," << grid_frontier_count << ","
        << refined_id_count << "," << refined_point_count << ","
        << backup_goal_count << "," << selected_source << ","
        << selected_frontier_id << "," << selected_frontier_grid_id << ","
        << selected_frontier_root_id << "," << next_pos.x() << ","
        << next_pos.y() << "," << next_pos.z() << "," << next_yaw << ","
        << path_len << "," << time_lb << "," << backup_goal_count << ","
        << sel_motion_cost_ << "," << sel_detour_cost_ << ","
        << sel_pocket_score_ << "," << sel_gradient_ << "," << sel_backward_penalty_ << ","
        << sel_total_score_ << "," << candidate_count_ << ","
        // [ABLATION] L1/L2 机制证据 + 阶段耗时
        << sel_straight_dist_ << "," << sel_path_stretch_ << ","
        << sel_crosses_wall_ << "," << sel_same_grid_other_component_ << ","
        << ccl_candidates_before_ << "," << ccl_filtered_num << ","
        << sel_ray_term_ << "," << sel_density_term_ << ","
        << sel_structure_score_ << "," << sel_pocket_flag_ << ","
        << sel_gradient_ << "," << has_next_grid_flag_ << ","
        << sel_tsp_order_ << ","
        << alt_changed_by_structure_ << "," << alt_changed_by_guidance_ << ","
        << (ros::Time::now() - plan_begin_time).toSec() * 1000.0 << ","
        << plan_ms_grid_ << "," << plan_ms_reach_ << "," << plan_ms_rank_ << "\n";
    decision_trace_log_file_.flush();
  };

  auto eval_frontier_motion =
      [&](int fid, double& motion_cost, double& path_len, bool strict_gate = true) -> bool {
    if (fid < 0 || fid >= static_cast<int>(ed_->points_.size()) ||
        fid >= static_cast<int>(ed_->yaws_.size())) {
      return false;
    }

    // Hard reachability gate (independent of yaw/motion weighting):
    // 1) path search fallback(1000) => unreachable
    // 2) path too long / too tortuous => not suitable as current local target
    vector<Vector3d> reach_path;
    const double reach_len = ViewNode::searchPath(pos, ed_->points_[fid], reach_path);
    if (!std::isfinite(reach_len) || reach_len >= 999.0) {
      return false;
    }
    const double euclid = std::max((ed_->points_[fid] - pos).norm(), 0.25);
    const double stretch = reach_len / euclid;
    if (strict_gate) {
      if (reach_len > ep_->grid_switch_max_path_length_ ||
          stretch > ep_->grid_switch_max_path_stretch_) {
        return false;
      }
    }

    vector<Vector3d> tmp_path;
    motion_cost = ViewNode::computeCost(
        pos, ed_->points_[fid], yaw[0], ed_->yaws_[fid], vel, yaw[1], tmp_path);
    if (!std::isfinite(motion_cost) || motion_cost >= 900.0) {
      return false;
    }
    path_len = 0.0;
    for (int pi = 1; pi < static_cast<int>(tmp_path.size()); ++pi) {
      path_len += (tmp_path[pi] - tmp_path[pi - 1]).norm();
    }

    // computeCost may reuse a fallback path internally; keep another geometric
    // guard here, but return the true search-path length to the caller.
    const double used_len = (path_len > 1e-3) ? path_len : reach_len;
    const double used_stretch = used_len / euclid;
    if (strict_gate) {
      if (used_len > ep_->grid_switch_max_path_length_ * 1.1 ||
          used_stretch > ep_->grid_switch_max_path_stretch_ + 0.2) {
        return false;
      }
    }
    path_len = reach_len;
    return true;
  };

  auto yaw_delta = [](double from, double to) {
    return std::fabs(std::atan2(std::sin(to - from), std::cos(to - from)));
  };

  auto yaw_time_cost = [&](double from, double to) {
    return yaw_delta(from, to) / std::max(1e-3, ViewNode::yd_);
  };

  auto frontier_small_area_priority = [&](int fid) -> double {
    const Frontier* f = frontier_finder_->getFrontierById(fid);
    if (f == nullptr) return 0.0;

    const double eff_cov =
        std::max(f->coverage_overlap_, f->filtered_coverage_overlap_);
    const double adaptive =
        std::min(std::max(f->adaptive_voxel_score_, 0.0), 1.0);
    const double bounded =
        std::min(std::max(f->bounded_pocket_score_, 0.0), 1.0);
    const double pass_debt =
        std::min(std::max(f->pass_debt_score_, 0.0), 1.0);
    const double back_unknown =
        std::min(std::max(static_cast<double>(f->back_unknown_num_) / 35.0, 0.0), 1.0);

    // FAEP-style independent small-area evidence belongs to the frontier
    // layer.  bounded_pocket_score is intentionally low-weighted because in
    // cluttered maps it can saturate for many frontiers; adaptive/pass-debt
    // and rear unknown count keep the signal selective.
    double priority =
        0.42 * adaptive + 0.28 * pass_debt + 0.18 * bounded + 0.12 * back_unknown;
    if (f->bounded_pocket_flag_ && f->back_unknown_num_ >= 8) priority += 0.10;

    if (eff_cov >= 0.88) {
      priority *= 0.40;
    } else if (eff_cov >= 0.68) {
      priority *= 0.70;
    }
    return std::min(std::max(priority, 0.0), 1.0);
  };

  // Grid CP (RACER coverage path) — single global ordering.
  // IMPORTANT: TSP works on active fine grids; coarse/root grids are only used
  // for lock and switching conditions, not as TSP nodes.
  ed_->grid_tour_ids_.clear();
  ed_->grid_tour_.clear();
  vector<int> tsp_gid_idx;
  tsp_gid_idx.reserve(ed_->grid_ids_.size());
  if (ed_->grid_ids_.size() >= 2) {
    for (int i = 0; i < static_cast<int>(ed_->grid_ids_.size()); ++i) {
      tsp_gid_idx.push_back(i);
    }

    vector<int> tsp_grid_ids;
    vector<Vector3d> tsp_grid_centers;
    tsp_grid_ids.reserve(tsp_gid_idx.size());
    tsp_grid_centers.reserve(tsp_gid_idx.size());
    for (int idx : tsp_gid_idx) {
      if (idx < 0 || idx >= static_cast<int>(ed_->grid_ids_.size())) continue;
      tsp_grid_ids.push_back(ed_->grid_ids_[idx]);
      tsp_grid_centers.push_back(ed_->grid_centers_[idx]);
    }

    vector<int> coarse_tour;
    if (tsp_grid_ids.size() >= 2) {
      findGlobalTourViaGrids(pos, vel, yaw, tsp_grid_ids, tsp_grid_centers, coarse_tour);
      for (int cid : coarse_tour) {
        if (cid < 0 || cid >= static_cast<int>(tsp_gid_idx.size())) continue;
        ed_->grid_tour_ids_.push_back(tsp_gid_idx[cid]);
      }
    } else if (tsp_grid_ids.size() == 1) {
      ed_->grid_tour_ids_.push_back(tsp_gid_idx.front());
    }
  } else if (ed_->grid_ids_.size() == 1) {
    tsp_gid_idx = {0};
    ed_->grid_tour_ids_ = {0};
  }

  // Keep the physical current fine grid and current active fine grid inside the
  // near lookahead, but do not force them to the first position. RACER's global
  // hgrid guidance depends on the TSP order remaining visible to the selector.
  auto keep_grid_in_tour_lookahead = [&](int pin_gid) {
    if (pin_gid < 0 || ed_->grid_ids_.empty()) return;
    for (int i = 0; i < static_cast<int>(ed_->grid_ids_.size()); ++i) {
      if (ed_->grid_ids_[i] != pin_gid) continue;
      auto it = std::find(ed_->grid_tour_ids_.begin(), ed_->grid_tour_ids_.end(), i);
      if (it != ed_->grid_tour_ids_.end()) {
        const int cur_order = static_cast<int>(std::distance(ed_->grid_tour_ids_.begin(), it));
        if (cur_order < std::max(1, ep_->grid_switch_lookahead_)) break;
        ed_->grid_tour_ids_.erase(it);
      }
      const int keep_pos =
          ed_->grid_tour_ids_.empty()
              ? 0
              : std::min(static_cast<int>(ed_->grid_tour_ids_.size()),
                         std::max(1, ep_->grid_switch_lookahead_ - 1));
      ed_->grid_tour_ids_.insert(ed_->grid_tour_ids_.begin() + keep_pos, i);
      break;
    }
  };
  keep_grid_in_tour_lookahead(active_grid_id_);
  keep_grid_in_tour_lookahead(cur_gid);

  ed_->grid_tour_.clear();
  ed_->grid_tour_waypoints_.clear();
  ed_->grid_tour_.push_back(pos);
  ed_->grid_tour_waypoints_.push_back(pos);

  Vector3d last_waypoint = pos;
  for (int gid_idx : ed_->grid_tour_ids_) {
    if (gid_idx < 0 || gid_idx >= (int)ed_->grid_centers_.size()) continue;
    const Vector3d& waypoint = ed_->grid_centers_[gid_idx];
    ed_->grid_tour_waypoints_.push_back(waypoint);
    ed_->grid_tour_.push_back(waypoint);
    last_waypoint = waypoint;
  }

  ed_->grid_frontier_ids_.clear();

  if (ed_->points_.empty()) {
    ROS_WARN("No frontiers from detector, fallback to grid-coverage cleanup logic.");
  }

  int selected_grid_id = -1;
  int selected_grid_order = -1;
  Vector3d next_grid_pos = Vector3d::Zero();
  double next_grid_yaw = yaw[0];
  bool has_next_grid = false;

  ++grid_decision_round_;

  struct GridPickCandidate {
    int order_ = -1;
    int gid_idx_ = -1;
    vector<int> expanded_grid_ids_;
    vector<int> frontier_ids_;
    vector<int> own_frontier_ids_;
    int unknown_voxels_ = 0;
    int frontier_voxels_ = 0;
    int adjacency_hops_ = 99;
    int reachable_frontier_cnt_ = 0;
    int own_reachable_frontier_cnt_ = 0;
    double best_motion_cost_ = std::numeric_limits<double>::infinity();
    double best_path_len_ = std::numeric_limits<double>::infinity();
    double own_best_motion_cost_ = std::numeric_limits<double>::infinity();
    double own_best_path_len_ = std::numeric_limits<double>::infinity();
    double center_dist_ = std::numeric_limits<double>::infinity();
    bool has_reachable_frontier_ = false;
    bool has_own_reachable_frontier_ = false;
    bool is_current_grid_ = false;
    bool is_active_grid_ = false;
    double info_norm_ = 0.0;
    double coverage_deficit_ = 0.0;
    double travel_cost_ = std::numeric_limits<double>::infinity();
    double neighbor_penalty_ = 0.0;
    double score_ = std::numeric_limits<double>::infinity();
    double grid_anchor_path_len_ = std::numeric_limits<double>::infinity();
    bool recent_visited_ = false;
    bool residual_closeout_ = false;
  };

  // CRITICAL: refresh frontier coverage BEFORE grid selection.
  // The coverage gate checks eff_cov which was computed from stale
  // coverage stamps in the previous round. We must:
  // 1) updateCoverage: mark path segments as covered (lateral inflation)
  // 2) updateFrontierTrackingStats: recompute coverage_overlap_ from
  //    the fresh coverage stamps
  // Without both, the gate sees old values and allows switching away
  // from grids with freshly-passed-but-unobserved frontiers.
  frontier_finder_->updateCoverage(pos, yaw);
  frontier_finder_->updateFrontierTrackingStats(pos);

  // Declare local state used by simplified handoff and trajectory planning
  vector<pair<Vector3d, double>> backup_view_goals;
  auto add_backup_goal = [&](const Vector3d& p, double y) {
    for (const auto& g : backup_view_goals)
      if ((g.first - p).norm() < 0.05) return;
    backup_view_goals.emplace_back(p, y);
  };
  int local_frontier_stage = 3;  // simplified always uses stage 3

  // ============================================================
  // SIMPLIFIED GRID-TO-FRONTIER HANDOFF
  // Architecture: Global TSP decides grid order.
  //               Local 3-factor scoring picks best frontier.
  //  - Factor 1: Motion cost (A* path + yaw alignment)
  //  - Factor 2: Small-area priority (FAEP Bottom Ray evidence)
  //  - Factor 3: Gradient guidance (RUSH next-grid direction)
  // ============================================================

  // --- Step 1: Select target grid from TSP order ---
  // Simply pick the first unfinished, frontier-containing grid in TSP order.
  selected_grid_id = -1;
  selected_grid_order = -1;
  if (!ed_->grid_tour_ids_.empty()) {
    for (size_t o = 0; o < ed_->grid_tour_ids_.size(); ++o) {
      int gid_idx = ed_->grid_tour_ids_[o];
      if (gid_idx < 0 || gid_idx >= static_cast<int>(ed_->grid_ids_.size())) continue;
      int gid = ed_->grid_ids_[gid_idx];
      const auto& grid = grid_planner_->getGrid(gid);
      if (grid.status_ == GridCell::FINISHED || grid.status_ == GridCell::PURE_OBSTACLE)
        continue;
      selected_grid_id = gid;
      selected_grid_order = static_cast<int>(o);
      break;
    }
  }

  // Fallback: current physical grid
  if (selected_grid_id < 0 && cur_gid >= 0 &&
      cur_gid < static_cast<int>(grid_planner_->all_grids_.size())) {
    selected_grid_id = cur_gid;
  }

  // Ultimate fallback: closest grid with frontiers
  if (selected_grid_id < 0) {
    double best_d = 1e9;
    for (int gid : ed_->grid_ids_) {
      double d = (grid_planner_->getGrid(gid).center_ - pos).norm();
      if (d < best_d) { best_d = d; selected_grid_id = gid; }
    }
  }

  // Update active grid (replaces old root/component lock tracking)
  if (selected_grid_id >= 0 && selected_grid_id != active_grid_id_) {
    active_grid_id_ = selected_grid_id;
    // Update root tracking for visualization
    active_root_grid_id_ = grid_planner_->getSubRegionId(selected_grid_id);
  }
  // [MODIFIED BY CODEX]: The tracking logger was initialized but never
  // invoked by the simplified handoff, leaving a CSV with only its header.
  frontier_finder_->setGridPlanningLogContext(
      local_frontier_stage, false, root_unreachable_rounds_,
      active_grid_id_, active_root_grid_id_);
  frontier_finder_->writeFrontierTrackingLogOnly(pos);

  // --- Step 2: FUEL-original-style start row for all active frontiers ---
  // [MODIFIED BY CODEX]: All frontiers participate in the ATSP start-cost
  // comparison.  This uses ViewNode's 1 ms bounded A* estimator exactly like
  // FUEL-original; it is not a hard reachability gate.  Only the finally
  // selected target receives the planner-manager's 1 s exact A* search.
  ed_->grid_frontier_ids_.clear();
  vector<int> candidate_fids;
  std::unordered_map<int, double> candidate_start_path_lengths;
  std::unordered_map<int, double> candidate_start_motion_costs;
  candidate_fids.reserve(ed_->points_.size());
  candidate_start_path_lengths.reserve(ed_->points_.size());
  candidate_start_motion_costs.reserve(ed_->points_.size());
  // [ABLATION] 可达性检验（逐前沿有界 A*）耗时
  const ros::Time t_reach_begin = ros::Time::now();
  for (int fid = 0; fid < static_cast<int>(ed_->points_.size()); ++fid) {
    vector<Vector3d> path;
    // Zero velocity keeps the direction-consistency term separate below while
    // retaining the original max(position-time, yaw-time) edge definition.
    const double motion_cost = ViewNode::computeCost(
        pos, ed_->points_[fid], yaw[0], ed_->yaws_[fid],
        Vector3d::Zero(), yaw[1], path);
    double path_len = 0.0;
    for (size_t p = 1; p < path.size(); ++p)
      path_len += (path[p] - path[p - 1]).norm();
    candidate_fids.push_back(fid);
    candidate_start_path_lengths[fid] = path_len > 1e-6
                                             ? path_len
                                             : (ed_->points_[fid] - pos).norm();
    candidate_start_motion_costs[fid] =
        std::isfinite(motion_cost) ? motion_cost : 1000.0;
  }
  ROS_INFO("[LocalStartCost] original_bounded_astar_all=%zu", candidate_fids.size());
  plan_ms_reach_ = (ros::Time::now() - t_reach_begin).toSec() * 1000.0;

  // [ABLATION]: 论文算法 1 第 17 行——同网格但分属不同连通分量的前沿
  // 本轮不参与局部排序。仅在 grid_planner/use_ccl_for_filter=true 时生效；
  // 关闭后即为“去掉连通约束”配置。
  ccl_candidates_before_ = static_cast<int>(candidate_fids.size());
  if (grid_planner_->cclFilterEnabled() && !candidate_fids.empty()) {
    vector<int> allowed_fids;
    allowed_fids.reserve(candidate_fids.size());
    for (int fid : candidate_fids) {
      const int gid = grid_planner_->getGridContainingPoint(ed_->points_[fid]);
      const Frontier* ftr = frontier_finder_->getFrontierById(fid);
      const Vector3d ftr_pos = (ftr != nullptr) ? ftr->average_ : ed_->points_[fid];
      if (grid_planner_->componentAllowsFrontier(gid, ftr_pos, pos)) {
        allowed_fids.push_back(fid);
      } else {
        ++ccl_filtered_num;
        candidate_start_path_lengths.erase(fid);
        candidate_start_motion_costs.erase(fid);
      }
    }
    if (!allowed_fids.empty()) {
      candidate_fids.swap(allowed_fids);
    } else {
      // 安全兜底：全部被筛时保留原候选，避免人为制造 NO_FRONTIER
      ROS_WARN("[CCLFilter] all %d candidates rejected, keep original set",
               static_cast<int>(candidate_fids.size()));
      ccl_filtered_num = 0;
    }
    ROS_INFO("[CCLFilter] candidates=%d, filtered_by_component=%d",
             static_cast<int>(candidate_fids.size()), ccl_filtered_num);
  }

  ed_->grid_frontier_ids_ = candidate_fids;

  // --- Step 3: Get next grid for RUSH-style gradient guidance ---
  Vector3d next_grid_center = pos;
  has_next_grid = false;
  if (!ed_->grid_tour_ids_.empty() && selected_grid_order >= 0 &&
      selected_grid_order + 1 < static_cast<int>(ed_->grid_tour_ids_.size())) {
    int next_gid_idx = ed_->grid_tour_ids_[selected_grid_order + 1];
    if (next_gid_idx >= 0 && next_gid_idx < static_cast<int>(ed_->grid_ids_.size())) {
      next_grid_center = ed_->grid_centers_[next_gid_idx];
      has_next_grid = true;
    }
  }
  has_next_grid_flag_ = has_next_grid ? 1 : 0;

  // --- Step 4: Local-frontier start costs ---
  // [MODIFIED BY CODEX]: Four-term start cost only:
  // C_s = motion/yaw time + velocity-direction consistency
  //       - pocket reward - grid-direction reward.
  // None of these start-only corrections is added to frontier-pair ATSP edges.
  const double w_pocket = ep_->local_pocket_weight_;

  struct SimpleCand {
    double score_; int fid_;
    double motion_cost_, path_length_, detour_cost_, pocket_score_, gradient_, backward_penalty_;
    // [ABLATION]: 反事实打分——去掉结构项 / 去掉方向项后的同一候选代价，
    // 用于统计“机制是否改变了首目标”（L2 证据）。
    double score_no_structure_, score_no_guidance_;
  };

  // [ABLATION] 阶段计时与反事实统计（供 decision_trace 使用）
  ros::Time t_rank_begin = ros::Time::now();
  vector<SimpleCand> scored;
  scored.reserve(candidate_fids.size());

  for (int fid : candidate_fids) {
    if (fid < 0 || fid >= static_cast<int>(ed_->points_.size())) continue;

    const auto path_it = candidate_start_path_lengths.find(fid);
    if (path_it == candidate_start_path_lengths.end()) continue;
    const double path_len = path_it->second;
    const auto motion_it = candidate_start_motion_costs.find(fid);
    if (motion_it == candidate_start_motion_costs.end()) continue;
    const double motion_cost = motion_it->second;

    const Frontier* ftr = frontier_finder_->getFrontierById(fid);
    const Vector3d& frontier_center = ftr != nullptr ? ftr->average_ : ed_->points_[fid];
    double pocket_score = 0.0;
    bool is_pocket = false;
    if (ftr != nullptr) {
      is_pocket = ftr->bounded_pocket_flag_;
      if (is_pocket) pocket_score = ftr->bounded_pocket_score_;
    }

    // The planner minimizes score, so a frontier aligned with the next global
    // grid reduces the score.
    // [ABLATION]: exploration/use_grid_guidance=false 时该项恒为 0
    double gradient = 0.0;
    if (has_next_grid && use_grid_guidance_) {
      double dv = std::max(0.5, (frontier_center - next_grid_center).norm());
      gradient = std::min(2.0, 1.0 / (dv * dv));
    }

    // Direction-consistency cost from the current translational velocity to
    // the candidate displacement.  At near-hover the flight direction is not
    // observable, so the term is disabled instead of evaluating an unstable
    // normalized dot product.
    double direction_cost = 0.0;
    const Vector3d displacement = ed_->points_[fid] - pos;
    if (vel.norm() > 0.1 && displacement.norm() > 1e-3) {
      const double cosine = std::max(
          -1.0, std::min(1.0, vel.dot(displacement) /
                                  (vel.norm() * displacement.norm())));
      direction_cost = ViewNode::w_dir_ * std::acos(cosine);
    }

    const double score = motion_cost + direction_cost
                         - w_pocket * pocket_score - gradient;
    // [ABLATION] 反事实代价：把结构奖励/方向奖励各自加回去
    const double score_no_structure = score + w_pocket * pocket_score;
    const double score_no_guidance = score + gradient;
    scored.push_back({score, fid, motion_cost, path_len, 0.0,
                      pocket_score, gradient, direction_cost,
                      score_no_structure, score_no_guidance});
  }

  const double rank_ms = (ros::Time::now() - t_rank_begin).toSec() * 1000.0;
  plan_ms_rank_ = rank_ms;

  // [ABLATION] 三种口径下的首目标，用于判断机制是否改变决策
  int cf_best_fid = -1, cf_best_no_structure = -1, cf_best_no_guidance = -1;
  double cf_best = std::numeric_limits<double>::infinity();
  double cf_best_ns = std::numeric_limits<double>::infinity();
  double cf_best_ng = std::numeric_limits<double>::infinity();
  for (const auto& c : scored) {
    if (c.score_ < cf_best) { cf_best = c.score_; cf_best_fid = c.fid_; }
    if (c.score_no_structure_ < cf_best_ns) { cf_best_ns = c.score_no_structure_; cf_best_no_structure = c.fid_; }
    if (c.score_no_guidance_ < cf_best_ng) { cf_best_ng = c.score_no_guidance_; cf_best_no_guidance = c.fid_; }
  }

  // Sort by score ascending
  std::sort(scored.begin(), scored.end(),
            [](const SimpleCand& a, const SimpleCand& b) { return a.score_ < b.score_; });

  // [MODIFIED BY CODEX]: Restore one aligned record per global-TSP grid.
  // This makes a later pull-back diagnosable as grid guidance or local ATSP.
  if (enable_grid_select_log_ && grid_select_log_file_.is_open()) {
    int next_grid_id = -1;
    if (has_next_grid && selected_grid_order + 1 < static_cast<int>(ed_->grid_tour_ids_.size())) {
      const int idx = ed_->grid_tour_ids_[selected_grid_order + 1];
      if (idx >= 0 && idx < static_cast<int>(ed_->grid_ids_.size())) next_grid_id = ed_->grid_ids_[idx];
    }
    for (int order = 0; order < static_cast<int>(ed_->grid_tour_ids_.size()); ++order) {
      const int grid_idx = ed_->grid_tour_ids_[order];
      if (grid_idx < 0 || grid_idx >= static_cast<int>(ed_->grid_ids_.size())) continue;
      const int gid = ed_->grid_ids_[grid_idx];
      const GridCell& grid = grid_planner_->getGrid(gid);
      int reachable_count = 0;
      double best_motion = std::numeric_limits<double>::infinity();
      double best_path = std::numeric_limits<double>::infinity();
      double best_score = std::numeric_limits<double>::infinity();
      for (const auto& cand : scored) {
        if (grid_planner_->getGridContainingPoint(ed_->points_[cand.fid_]) != gid) continue;
        ++reachable_count;
        if (cand.score_ < best_score) {
          best_score = cand.score_;
          best_motion = cand.motion_cost_;
          best_path = cand.path_length_;
        }
      }
      const bool selected = gid == selected_grid_id;
      grid_select_log_file_
          << grid_decision_round_ << "," << ros::Time::now().toSec() << ","
          << order << "," << grid_idx << "," << gid << ",1,"
          << grid.contained_frontier_ids_.size() << "," << reachable_count << ","
          << best_motion << "," << best_path << ","
          << (grid.center_ - pos).norm() << "," << best_score << ","
          << static_cast<int>(grid.status_) << "," << grid.unknown_num_ << ","
          << grid.unknown_frontier_num_ << "," << grid.coverage_ratio_ << ","
          << grid.covered_by_path_ << "," << selected_grid_order << "," << selected << ","
          << grid.unknown_num_ << "," << grid.contained_frontier_ids_.size() << ",-1,"
          << best_motion << ",0,0," << reachable_count << "," << best_motion << ","
          << best_path << "," << (gid == cur_gid) << "," << grid.active_ << ","
          << local_frontier_stage << ",0," << root_unreachable_rounds_ << ",-1,"
          << selected_grid_order << "," << selected_grid_id << ",-1,"
          << order << "," << gid << "," << best_score << ","
          << "tsp_next_" << next_grid_id << "\n";
    }
    grid_select_log_file_.flush();
  }

  // --- Step 5: Original-FUEL-style local ATSP solved by LKH ---
  // [MODIFIED BY CODEX]: The four-term C_s score fixes the first frontier.
  // LKH only orders the remaining local candidates; it never overrides the
  // frontier-layer decision by selecting another start node.
  ed_->local_tsp_ids_.clear();
  ed_->refined_ids_.clear();
  ed_->refined_exit_id_ = -1;
  const int local_num = std::min(static_cast<int>(scored.size()),
                                 std::max(1, local_tsp_candidate_num_));
  std::unordered_map<int, double> start_path_lengths;
  std::unordered_map<int, int> dijkstra_orders;
  std::unordered_map<int, std::string> dijkstra_reasons;
  if (local_num > 0) {
    vector<SimpleCand> local_nodes(scored.begin(), scored.begin() + local_num);
    struct LocalEdge { double cost_ = 0.0, length_ = 0.0; bool valid_ = false; };
    vector<vector<LocalEdge>> edges(local_num, vector<LocalEdge>(local_num));
    std::string lkh_sequence;
    auto log_edge = [&](const char* type, int from_idx, int to_idx,
                        double raw, double final, double plen, double euclid,
                        bool reachable, bool local_ok, const char* reason,
                        int order, int selected_first) {
      if (!enable_local_tsp_cost_log_ || !local_tsp_cost_log_file_.is_open()) return;
      const int from_id = from_idx >= 0 ? local_nodes[from_idx].fid_ : -1;
      const int to_id = to_idx >= 0 ? local_nodes[to_idx].fid_ : -1;
      const int from_gid = from_idx >= 0 ? grid_planner_->getGridContainingPoint(ed_->points_[from_id]) : cur_gid;
      const int to_gid = to_idx >= 0 ? grid_planner_->getGridContainingPoint(ed_->points_[to_id]) : -1;
      const double stretch = plen > 0.0 ? plen / std::max(0.25, euclid) : 1.0;
      const auto path_it = start_path_lengths.find(to_id);
      const auto order_it = dijkstra_orders.find(to_id);
      const auto reason_it = dijkstra_reasons.find(to_id);
      const double start_path = path_it == start_path_lengths.end() ? -1.0 : path_it->second;
      const int dijkstra_order = order_it == dijkstra_orders.end() ? -1 : order_it->second;
      const int in_dijkstra = dijkstra_order >= 0 ? 1 : 0;
      const std::string dijkstra_reason =
          reason_it == dijkstra_reasons.end() ? "pending" : reason_it->second;
      local_tsp_cost_log_file_ << grid_decision_round_ << "," << ros::Time::now().toSec()
          << "," << type << "," << from_id << "," << to_id << ","
          << from_gid << "," << to_gid << "," << raw << "," << final << ","
          << plen << "," << euclid << "," << stretch << ","
          << std::max(0.0, plen - euclid) << ",0,"
          << (reachable ? 1 : 0) << "," << (local_ok ? 1 : 0) << "," << reason
          << "," << (reachable ? 1 : 0) << ",0," << cur_gid << ","
          << active_root_grid_id_ << "," << order << "," << selected_first
          << ",0,0,0,0,0,0,0,"
          << lkh_sequence << "," << start_path << "," << dijkstra_order << ","
          << in_dijkstra << "," << dijkstra_reason << "\n";
    };
    for (int i = 0; i < local_num; ++i) {
      start_path_lengths[local_nodes[i].fid_] = local_nodes[i].path_length_;
      log_edge("start", -1, i, local_nodes[i].motion_cost_, local_nodes[i].score_,
               local_nodes[i].path_length_, (ed_->points_[local_nodes[i].fid_] - pos).norm(),
               true, true, "ok", i, i == 0 ? 1 : 0);
      for (int j = 0; j < local_num; ++j) {
        if (i == j) continue;
        const int fid_i = local_nodes[i].fid_, fid_j = local_nodes[j].fid_;
        const double euclid = (ed_->points_[fid_j] - ed_->points_[fid_i]).norm();
        // [MODIFIED BY CODEX]: Match FUEL-original's incremental frontier
        // graph for inter-frontier ATSP edges.
        vector<Vector3d> edge_path;
        double raw = 1000.0;
        const bool cache_ok = frontier_finder_->getFrontierPairCost(
            fid_i, fid_j, raw, edge_path);
        double plen = 0.0;
        for (size_t p = 1; p < edge_path.size(); ++p)
          plen += (edge_path[p] - edge_path[p - 1]).norm();
        if (!cache_ok || !std::isfinite(raw)) {
          log_edge("pair", i, j, raw, 0.0, plen, euclid, false, false,
                   "cache_invalid", -1, 0);
          continue;
        }
        edges[i][j].valid_ = true;
        edges[i][j].length_ = plen;
        edges[i][j].cost_ = std::max(1e-3, raw);
        log_edge("pair", i, j, raw, edges[i][j].cost_, plen, euclid,
                 raw < 900.0, true, "frontier_cost_cache", -1, 0);
      }
    }
    if (enable_local_tsp_cost_log_ && local_tsp_cost_log_file_.is_open()) local_tsp_cost_log_file_.flush();

    // Matrix node 0 is the predicted UAV state; nodes 1..N are the sorted
    // local candidates. This is the same ATSP closure used by FUEL-original.
    Eigen::MatrixXd local_cost = Eigen::MatrixXd::Constant(
        local_num + 1, local_num + 1, std::max(1.0, local_tsp_block_cost_));
    local_cost.diagonal().setZero();
    for (int i = 0; i < local_num; ++i) {
      local_cost(i + 1, 0) = 0.0;
      for (int j = 0; j < local_num; ++j) {
        if (i != j && edges[i][j].valid_)
          local_cost(i + 1, j + 1) = edges[i][j].cost_;
      }
    }
    // Only the lowest-C_s candidate is reachable from node 0 at normal cost.
    // All other start arcs are blocked, so LKH cannot steal the first target.
    local_cost(0, 1) = std::max(1e-3, local_nodes.front().score_);

    // LKH rejects ATSP instances smaller than three nodes.  For exactly one
    // local frontier the order is already determined: current state -> that
    // frontier.  Do not send the 2-node instance to the in-process LKH
    // library, because its error handler terminates exploration_node.
    const bool use_lkh = local_num >= 2;
    vector<int> order;
    if (!use_lkh) order.push_back(0);
    if ((!use_lkh || solveLocalTourViaLKH(local_cost, order)) &&
        !order.empty() && order.front() == 0) {
      for (int local_idx : order) {
        if (local_idx >= 0 && local_idx < local_num)
          ed_->local_tsp_ids_.push_back(local_nodes[local_idx].fid_);
      }
      for (size_t i = 0; i < ed_->local_tsp_ids_.size(); ++i) {
        if (i > 0) lkh_sequence += ";";
        lkh_sequence += std::to_string(ed_->local_tsp_ids_[i]);
      }
      // [MODIFIED BY CODEX]: Match FUEL Sect. V-B.  R_rf is an Euclidean
      // viewpoint radius, not an A* path-length radius.  Frontiers inside the
      // continuous prefix expose multiple viewpoints.  The first successor
      // outside the radius is retained as a fixed exit anchor so the last
      // refined layer pays the cost of continuing along the local ATSP tour.
      const double refine_radius = ep_->refined_radius_ > 0.0 ? ep_->refined_radius_ : 5.0;
      for (size_t i = 0; i < ed_->local_tsp_ids_.size() &&
                         i < static_cast<size_t>(std::max(1, local_tsp_candidate_num_)); ++i) {
        const int fid = ed_->local_tsp_ids_[i];
        if (fid < 0 || fid >= static_cast<int>(ed_->points_.size())) continue;
        const double start_euclidean = (ed_->points_[fid] - pos).norm();
        if (!std::isfinite(start_euclidean) || start_euclidean > refine_radius) {
          ed_->refined_exit_id_ = fid;
          dijkstra_orders[fid] = static_cast<int>(ed_->refined_ids_.size());
          dijkstra_reasons[fid] = ed_->refined_ids_.empty()
                                      ? "exit_anchor_no_in_radius_frontier"
                                      : "exit_anchor_after_euclidean_radius";
          for (size_t j = i + 1; j < ed_->local_tsp_ids_.size(); ++j)
            dijkstra_reasons[ed_->local_tsp_ids_[j]] = "after_exit_anchor";
          break;
        }
        ed_->refined_ids_.push_back(fid);
        dijkstra_orders[fid] = static_cast<int>(i);
        dijkstra_reasons[fid] = "within_start_euclidean_radius";
      }
      selected_source = use_lkh ? "local_lkh_atsp" : "local_single_frontier";
      if (enable_local_tsp_cost_log_ && local_tsp_cost_log_file_.is_open()) {
        for (int k = 0; k < static_cast<int>(order.size()); ++k) {
          const int from = k == 0 ? -1 : order[k - 1];
          const int to = order[k];
          const double raw = from < 0 ? local_nodes[to].motion_cost_ : edges[from][to].cost_;
          const double plen = from < 0 ? local_nodes[to].path_length_ : edges[from][to].length_;
          const double euclid = from < 0 ? (ed_->points_[local_nodes[to].fid_] - pos).norm()
                                         : (ed_->points_[local_nodes[to].fid_] -
                                            ed_->points_[local_nodes[from].fid_]).norm();
          log_edge("solution", from, to, raw, raw, plen, euclid, true, true,
                   use_lkh ? "lkh" : "single_candidate_no_lkh", k,
                   k == 0 ? 1 : 0);
        }
        local_tsp_cost_log_file_.flush();
      }
    } else {
      ROS_WARN("[LocalLKH] invalid local tour, fall back to fixed lowest-C_s frontier");
      ed_->local_tsp_ids_.push_back(local_nodes.front().fid_);
      ed_->refined_ids_.push_back(local_nodes.front().fid_);
      dijkstra_orders[local_nodes.front().fid_] = 0;
      dijkstra_reasons[local_nodes.front().fid_] = "lkh_fallback";
      selected_source = "local_lkh_fallback";
    }
  }

  // No executable chain must never be turned into a fake straight-line route.
  if (ed_->refined_ids_.empty() && ed_->refined_exit_id_ < 0 && !scored.empty()) {
    ed_->local_tsp_ids_.push_back(scored.front().fid_);
    ed_->refined_ids_.push_back(scored.front().fid_);
    start_path_lengths[scored.front().fid_] = scored.front().path_length_;
    dijkstra_orders[scored.front().fid_] = 0;
    dijkstra_reasons[scored.front().fid_] = "score_fallback";
    selected_source = "local_score_fallback";
  }

  candidate_count_ = static_cast<int>(scored.size());
  const int first_graph_frontier = !ed_->refined_ids_.empty()
                                       ? ed_->refined_ids_.front()
                                       : ed_->refined_exit_id_;
  if (first_graph_frontier >= 0) {
    const auto it = std::find_if(scored.begin(), scored.end(), [&](const SimpleCand& c) {
      return c.fid_ == first_graph_frontier;
    });
    if (it != scored.end()) {
      sel_motion_cost_ = it->motion_cost_;
      sel_detour_cost_ = it->detour_cost_;
      sel_pocket_score_ = it->pocket_score_;
      sel_gradient_ = it->gradient_;
      sel_backward_penalty_ = it->backward_penalty_;
      sel_total_score_ = it->score_;
    }
  }

  // Clean up component tracking (no longer using component locks)
  ed_->frontier_component_active_.clear();

  // --- Viewpoint refinement for the short-range LKH prefix only ---
  ed_->unrefined_points_.clear();
  for (int fid : ed_->refined_ids_)
    if (fid >= 0 && fid < static_cast<int>(ed_->points_.size()))
      ed_->unrefined_points_.push_back(ed_->points_[fid]);
  if (ed_->refined_exit_id_ >= 0 &&
      ed_->refined_exit_id_ < static_cast<int>(ed_->points_.size()))
    ed_->unrefined_points_.push_back(ed_->points_[ed_->refined_exit_id_]);

  ed_->n_points_.clear();
  vector<vector<double>> n_yaws;

  if (!ed_->refined_ids_.empty())
    frontier_finder_->setLocalViewRefinementLogContext(
        ed_->local_tsp_ids_, start_path_lengths, dijkstra_orders, dijkstra_reasons);
  if (!ed_->refined_ids_.empty())
    frontier_finder_->getViewpointsInfo(pos, yaw[0], ed_->refined_ids_,
                                        ep_->top_view_num_, ep_->max_decay_,
                                        ed_->n_points_, n_yaws);

  // [MODIFIED BY CODEX]: Equation (6) in FUEL terminates the local graph at
  // x_{Nrf+1,1}.  Append exactly that representative viewpoint, rather than
  // optimizing all viewpoints of the exit frontier.  If no frontier lies
  // inside R_rf, this singleton makes the graph start->exit and avoids an
  // empty refinement chain.
  if (ed_->refined_exit_id_ >= 0 &&
      ed_->refined_exit_id_ < static_cast<int>(ed_->points_.size()) &&
      ed_->refined_exit_id_ < static_cast<int>(ed_->yaws_.size())) {
    ed_->n_points_.push_back({ed_->points_[ed_->refined_exit_id_]});
    n_yaws.push_back({ed_->yaws_[ed_->refined_exit_id_]});
  }

  ed_->refined_points_.clear();
  ed_->refined_views_.clear();
  vector<double> refined_yaws;
  if (!ed_->n_points_.empty()) {
    bool has_any = false;
    for (auto& v : ed_->n_points_) { if (!v.empty()) { has_any = true; break; } }
    if (has_any)
      refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws,
                      ed_->refined_points_, refined_yaws);
  }

  if (!ed_->refined_points_.empty()) {
    next_pos = ed_->refined_points_[0];
    next_yaw = refined_yaws[0];
    selected_source = ed_->refined_ids_.empty() && ed_->refined_exit_id_ >= 0
                          ? "exit_anchor_direct"
                          : "refined_points";
    selected_frontier_id = first_graph_frontier;
    // [MODIFIED BY CODEX]: Dijkstra's later viewpoints are a pose look-ahead,
    // not a flight queue.  Only its first node can be published to the FSM.
    add_backup_goal(ed_->refined_points_[0], refined_yaws[0]);
    // [DIAG/FIX]: 细化视点偶尔会落在已知占据体素上（地图在两轮之间更新所致），
    // 此时 A* 必然 NO_PATH。按“细化视点 → 该前沿代表视点 → 前沿中心”的顺序兜底，
    // 后两者都是前沿检测保证的自由体素。
    if (selected_frontier_id >= 0 &&
        selected_frontier_id < static_cast<int>(ed_->points_.size())) {
      add_backup_goal(ed_->points_[selected_frontier_id],
                      ed_->yaws_[selected_frontier_id]);
      const Frontier* sel_ftr = frontier_finder_->getFrontierById(selected_frontier_id);
      if (sel_ftr != nullptr) {
        add_backup_goal(sel_ftr->average_, ed_->yaws_[selected_frontier_id]);
      }
    }
  } else {
    int best = -1; double bc = 1e9;
    for (int fid : ed_->grid_frontier_ids_) {
      if (fid < 0 || fid >= static_cast<int>(ed_->points_.size())) continue;
      vector<Vector3d> tmp_path;
      double mc = ViewNode::computeCost(pos, ed_->points_[fid], yaw[0],
                                        ed_->yaws_[fid], vel, yaw[1], tmp_path);
      if (std::isfinite(mc) && mc < bc) { bc = mc; best = fid; }
    }
    if (best >= 0) {
      next_pos = ed_->points_[best];
      next_yaw = ed_->yaws_[best];
      selected_source = "fallback_best";
      selected_frontier_id = best;
      add_backup_goal(next_pos, next_yaw);
    } else if (!ed_->grid_frontier_ids_.empty()) {
      int fid0 = ed_->grid_frontier_ids_[0];
      if (fid0 >= 0 && fid0 < static_cast<int>(ed_->points_.size()) &&
          fid0 < static_cast<int>(ed_->yaws_.size())) {
        next_pos = ed_->points_[fid0];
        next_yaw = ed_->yaws_[fid0];
        selected_source = "grid_frontier";
        selected_frontier_id = fid0;
        add_backup_goal(next_pos, next_yaw);
      }
    }
  }

  // [ABLATION] 本轮机制证据：选中目标的几何/结构/方向信息（L1），
  // 以及“去掉结构项 / 去掉方向项是否会改变首目标”（L2）。
  sel_tsp_order_ = selected_grid_order;
  if (selected_frontier_id >= 0 &&
      selected_frontier_id < static_cast<int>(ed_->points_.size())) {
    const Vector3d tgt = ed_->points_[selected_frontier_id];
    sel_straight_dist_ = (tgt - pos).norm();
    const double euclid = std::max(sel_straight_dist_, 0.25);
    const auto pl_it = candidate_start_path_lengths.find(selected_frontier_id);
    const double plen = (pl_it != candidate_start_path_lengths.end())
                            ? pl_it->second
                            : sel_straight_dist_;
    sel_path_stretch_ = plen / euclid;

    // 跨墙判定：沿当前位置→目标直线等距采样，命中已知/膨胀占据即置 1
    const int steps = std::max(2, static_cast<int>(std::ceil(sel_straight_dist_ / 0.1)));
    int occ_hits = 0;
    for (int s = 1; s < steps; ++s) {
      const Vector3d p = pos + (tgt - pos) * (static_cast<double>(s) / steps);
      Eigen::Vector3i idx;
      sdf_map_->posToIndex(p, idx);
      if (!sdf_map_->isInMap(idx)) continue;
      if (sdf_map_->getInflateOccupancy(idx) == 1) { ++occ_hits; break; }
    }
    sel_crosses_wall_ = occ_hits > 0 ? 1 : 0;

    const Frontier* sf = frontier_finder_->getFrontierById(selected_frontier_id);
    if (sf != nullptr) {
      sel_ray_term_ = sf->back_leak_ratio_;
      sel_density_term_ = sf->back_obstacle_ratio_;
      sel_structure_score_ = sf->bounded_pocket_score_;
      sel_pocket_flag_ = sf->bounded_pocket_flag_ ? 1 : 0;
      sel_pocket_score_ = sel_structure_score_;
      const int sel_gid = grid_planner_->getGridContainingPoint(sf->average_);
      sel_same_grid_other_component_ =
          grid_planner_->componentAllowsFrontier(sel_gid, sf->average_, pos) ? 0 : 1;
      if (has_next_grid && use_grid_guidance_) {
        const double dv = std::max(0.5, (sf->average_ - next_grid_center).norm());
        sel_gradient_ = std::min(2.0, 1.0 / (dv * dv));
      } else {
        sel_gradient_ = 0.0;
      }
    }
  }
  alt_changed_by_structure_ =
      (cf_best_no_structure >= 0 && selected_frontier_id >= 0 &&
       cf_best_no_structure != selected_frontier_id) ? 1 : 0;
  alt_changed_by_guidance_ =
      (cf_best_no_guidance >= 0 && selected_frontier_id >= 0 &&
       cf_best_no_guidance != selected_frontier_id) ? 1 : 0;

  for (size_t i = 0; i < ed_->refined_points_.size(); ++i)
    ed_->refined_views_.push_back(
        ed_->refined_points_[i] + 2.0 * Vector3d(cos(refined_yaws[i]), sin(refined_yaws[i]), 0));
  ed_->refined_views1_.clear(); ed_->refined_views2_.clear();
  for (size_t i = 0; i < ed_->refined_points_.size(); ++i) {
    vector<Vector3d> v1, v2;
    frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], refined_yaws[i]);
    frontier_finder_->percep_utils_->getFOV(v1, v2);
    ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
    ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
  }

  std::cout << "Next view: " << next_pos.transpose() << ", " << next_yaw << std::endl;

  // 规划到下一个观测点的轨迹
  t1 = ros::Time::now();

  // 生成 x,y,z 轨迹
  bool found_reachable_view = false;
  for (const auto& candidate : backup_view_goals) {
    planner_manager_->path_finder_->reset();
    const int astar_res = planner_manager_->path_finder_->search(pos, candidate.first);
    if (astar_res != Astar::REACH_END) {
      // [ABLATION][DIAG] 记录 A* 失败原因，便于区分“地图不通”与“目标不可达”
      Eigen::Vector3i si, ti;
      sdf_map_->posToIndex(pos, si);
      sdf_map_->posToIndex(candidate.first, ti);
      const bool t_in = sdf_map_->isInMap(ti);
      ROS_WARN("[LocalTarget] astar_fail code=%d start_occ=%d start_inf=%d "
               "target_in_map=%d target_occ=%d target_inf=%d",
               astar_res, sdf_map_->getOccupancy(si),
               sdf_map_->getInflateOccupancy(si), static_cast<int>(t_in),
               t_in ? sdf_map_->getOccupancy(ti) : -9,
               t_in ? sdf_map_->getInflateOccupancy(ti) : -9);
      continue;
    }
    ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();
    shortenPath(ed_->path_next_goal_);
    if (ed_->path_next_goal_.empty()) continue;
    if (ed_->path_next_goal_.size() < 2) {
      ed_->path_next_goal_.clear();
      ed_->path_next_goal_.push_back(pos);
      ed_->path_next_goal_.push_back(candidate.first);
    }
    next_pos = candidate.first;
    next_yaw = candidate.second;
    found_reachable_view = true;
    break;
  }
  if (!found_reachable_view) {
    ROS_WARN("[LocalTarget] no reachable viewpoint among %zu candidates",
             backup_view_goals.size());
    write_decision_trace(local_frontier_stage, NO_FRONTIER, -1.0, -1.0,
                         static_cast<int>(backup_view_goals.size()));
    return NO_FRONTIER;
  }

  // 计算偏航角的时间下界
  double diff = fabs(next_yaw - yaw[0]);
  double time_lb = min(diff, 2 * M_PI - diff) / ViewNode::yd_;

  // 根据距离选择不同的轨迹规划策略
  const double radius_far = 5.0;
  const double radius_close = 1.5;
  const double len = Astar::pathLength(ed_->path_next_goal_);
  if (len < radius_close) {
    // 目标很近，使用基于航点的优化
    planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
    ed_->next_goal_ = next_pos;

  } else if (len > radius_far) {
    // 目标很远，选择几何路径上的中间目标
    std::cout << "Far goal." << std::endl;
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = { ed_->path_next_goal_.front() };
    for (int i = 1; i < ed_->path_next_goal_.size() && len2 < radius_far; ++i) {
      auto cur_pt = ed_->path_next_goal_[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      truncated_path.push_back(cur_pt);
    }
    ed_->next_goal_ = truncated_path.back();
    planner_manager_->planExploreTraj(truncated_path, vel, acc, time_lb);
  } else {
    // 目标中等距离，使用动力学路径规划
    std::cout << "Mid goal" << std::endl;
    ed_->next_goal_ = next_pos;

    if (!planner_manager_->kinodynamicReplan(
            pos, vel, acc, ed_->next_goal_, Vector3d(0, 0, 0), time_lb)) {
      // The geometric A* above has already established that this viewpoint is
      // reachable.  A Kino search can still reject it transiently when the
      // current start state is close to an inflated voxel.  Do not turn that
      // into a command gap: use FUEL's existing waypoint/B-spline branch on
      // the validated geometric path and retry Kino on a later replan.
      ROS_WARN("[LocalTarget] Kino failed, use geometric A* trajectory fallback");
      planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
    }
  }

  // 检查轨迹时间是否满足偏航角时间下界
  if (planner_manager_->local_data_.position_traj_.getTimeSum() < time_lb - 0.1)
    ROS_ERROR("Lower bound not satified!");

  // 规划偏航角轨迹
  planner_manager_->planYawExplore(yaw, next_yaw, true, ep_->relax_time_);

  // 计算规划时间
  double traj_plan_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  double yaw_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Traj: %lf, yaw: %lf", traj_plan_time, yaw_time);
  double total = (ros::Time::now() - t2).toSec();
  ROS_WARN("Total time: %lf", total);
  ROS_WARN_COND(total > 1.0, "Total time too long: %lf", total);
  if (selected_frontier_id >= 0 &&
      selected_frontier_id < static_cast<int>(ed_->points_.size())) {
    const int selected_gid =
        grid_planner_->getGridContainingPoint(ed_->points_[selected_frontier_id]);
    if (selected_frontier_id == last_selected_frontier_id_) {
      ++selected_frontier_hold_rounds_;
    } else {
      selected_frontier_hold_rounds_ = 1;
    }
    last_selected_frontier_id_ = selected_frontier_id;
    last_selected_frontier_grid_id_ = selected_gid;
  }
  write_decision_trace(local_frontier_stage, SUCCEED, len, time_lb,
                       static_cast<int>(backup_view_goals.size()));

  return SUCCEED;
} 

bool FastExplorationManager::pathStaysInGrid(
    int grid_id, const vector<Vector3d>& path) const {
  if (grid_id < 0 || grid_id >= static_cast<int>(grid_planner_->all_grids_.size()) ||
      path.empty()) {
    return false;
  }

  const auto& grid = grid_planner_->getGrid(grid_id);
  const double sample_step = std::max(0.05, local_tsp_path_sample_step_);
  // Tight epsilon (2.5 cm) causes massive false negatives: A* path
  // discretisation routinely places waypoints a few cm outside the grid
  // bounds, especially for diagonal paths near boundaries.  This makes
  // pathStaysInGrid() reject frontiers that ARE in the same connected
  // component — the grid appears "done" prematurely, the lock releases,
  // and the drone jumps to far-away frontiers (pull-push oscillation).
  //
  // A genuine wall-separated path exits the grid by multiple metres.
  // Use 0.5 m tolerance to absorb discretisation noise while still
  // detecting real obstacle detours.
  const double eps = std::max(0.5, sample_step);
  auto inside_xy = [&](const Vector3d& p) {
    return p[0] >= grid.vmin_[0] - eps && p[0] <= grid.vmax_[0] + eps &&
           p[1] >= grid.vmin_[1] - eps && p[1] <= grid.vmax_[1] + eps;
  };

  // Track whether the path exits and then RE-ENTERS the grid.
  // A single exit near the boundary is discretisation noise;
  // exit + re-enter signals a true obstacle detour.
  bool exited = false;
  if (!inside_xy(path.front())) return false;
  for (int i = 1; i < static_cast<int>(path.size()); ++i) {
    const Vector3d delta = path[i] - path[i - 1];
    const int samples =
        std::max(1, static_cast<int>(std::ceil(delta.norm() / sample_step)));
    for (int k = 1; k <= samples; ++k) {
      const Vector3d p =
          path[i - 1] + delta * (static_cast<double>(k) / static_cast<double>(samples));
      if (!inside_xy(p)) {
        exited = true;
      } else if (exited) {
        // Exited and then re-entered → different connected component.
        return false;
      }
    }
  }
  return true;
}






/**
 * @brief 缩短路径，只保留关键的中间点
 * @param path 输入输出参数，需要缩短的路径，修改后返回缩短后的路径
 * 
 * 该函数通过以下步骤缩短路径：
 * 1. 检查路径是否为空，为空则返回错误
 * 2. 设置距离阈值，只保留距离足够远的点
 * 3. 遍历路径，判断是否需要保留当前点
 * 4. 确保路径中至少有三个点，以保证路径的平滑性
 */
void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }
  // 缩短路径，只保留关键的中间点
  const double dist_thresh = 3.0; // 距离阈值，超过此值的点会被保留
  vector<Vector3d> short_tour = { path.front() }; // 初始化新路径，包含原路径的第一个点
  
  // 遍历原路径的中间点（从第二个到倒数第二个）
  for (int i = 1; i < path.size() - 1; ++i) {
    // 如果当前点与新路径最后一个点的距离大于阈值，则保留该点
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // 否则，检查从新路径最后一个点到下一个点的直线是否会与障碍物或未知区域碰撞
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      // 使用射线投射检查路径是否碰撞
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        // 如果遇到障碍物或未知区域，则需要保留当前点
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }
  // 将原路径的最后一个点加入新路径（如果与新路径最后一个点的距离大于阈值）
  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());

  // 确保新路径至少有三个点，以保证路径的平滑性
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  // 将原路径替换为缩短后的路径
  path = short_tour;
}

void FastExplorationManager::findGlobalTourViaGrids(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
    const vector<int>& grid_ids, const vector<Vector3d>& grid_centers,
    vector<int>& tour_indices) {
  if (grid_ids.size() < 2) {
    tour_indices = {0};
    return;
  }

  auto t1 = ros::Time::now();

  // Get grid-based cost matrix
  Eigen::MatrixXd cost_mat;
  grid_planner_->getGridCostMatrix(cur_pos, cur_vel, cur_yaw, grid_ids, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Write TSP problem file
  ofstream prob_file(ep_->tsp_dir_ + "/single.tsp");
  string prob_spec = "NAME : grid\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
      "\nEDGE_WEIGHT_TYPE : "
      "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
  prob_file << prob_spec;

  const int scale = 100;
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = cost_mat(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }
  prob_file << "EOF";
  prob_file.close();

  // Call LKH TSP solver
  solveTSPLKH((ep_->tsp_dir_ + "/single.par").c_str());

  // Read tour. Matrix rows are: 0 virtual depot, 1 current state, 2.. grids.
  ifstream res_file(ep_->tsp_dir_ + "/single.txt");
  string res;
  while (getline(res_file, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }

  vector<int> rows;
  tour_indices.clear();
  while (getline(res_file, res)) {
    int id = stoi(res);
    if (id == -1) break;
    rows.push_back(id - 1);
  }
  res_file.close();

  int start_pos = -1;
  for (int i = 0; i < (int)rows.size(); ++i) {
    if (rows[i] == 1) {
      start_pos = i;
      break;
    }
  }
  if (start_pos >= 0) {
    for (int off = 1; off < (int)rows.size(); ++off) {
      int row = rows[(start_pos + off) % rows.size()];
      if (row == 0) break;
      if (row >= 2 && row < dimension)
        tour_indices.push_back(row - 2);
    }
  }
  if (tour_indices.empty() && !grid_ids.empty()) {
    tour_indices.push_back(0);
  }

  // [MODIFIED BY CODEX]: Do not repair the coverage-task order with a graph
  // that only contains already-observed free-space links.  At startup that
  // graph is empty and would erase unknown grids from the global direction.
  // Executability is enforced later by the local frontier A* graph.

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("[GridTSP] mat_time=%.3f, tsp_time=%.3f, tour_len=%zu",
           mat_time, tsp_time, tour_indices.size());

  // Log
  {
    std::ostringstream ss;
    ss << "[GridTSP_Tour] tour=[";
    for (size_t i = 0; i < tour_indices.size(); ++i) {
      if (i > 0) ss << "->";
      ss << tour_indices[i];
    }
    ss << "]";
    ROS_WARN_STREAM(ss.str());
  }
}

bool FastExplorationManager::solveLocalTourViaLKH(
    const Eigen::MatrixXd& cost_mat, vector<int>& tour_indices) {
  tour_indices.clear();
  const int dimension = cost_mat.rows();
  // LKH's ReadProblem() calls exit(EXIT_FAILURE) for DIMENSION < 3.  Never
  // delegate that input validation to the library because it runs in this ROS
  // process rather than in a child process.
  if (dimension < 3 || cost_mat.cols() != dimension) {
    ROS_WARN_STREAM("[LocalLKH] skip invalid local ATSP dimension=" << dimension);
    return false;
  }

  // [MODIFIED BY CODEX]: Use a separate local problem/result pair so the
  // global grid LKH output remains intact for visualization and logging.
  const string tsp_file = ep_->tsp_dir_ + "/local_frontier.tsp";
  const string par_file = ep_->tsp_dir_ + "/local_frontier.par";
  const string result_file = ep_->tsp_dir_ + "/local_frontier.txt";
  // A failed solver run must not be allowed to leave an earlier tour that is
  // then parsed as the current local order.
  std::remove(result_file.c_str());
  ofstream prob_file(tsp_file);
  if (!prob_file.is_open()) {
    ROS_ERROR_STREAM("[LocalLKH] cannot write " << tsp_file);
    return false;
  }
  prob_file << "NAME : local_frontier\nTYPE : ATSP\nDIMENSION : " << dimension
            << "\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX"
            << "\nEDGE_WEIGHT_SECTION\n";
  const int scale = 100;
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      const double c = std::max(0.0, std::min(cost_mat(i, j), local_tsp_block_cost_));
      prob_file << static_cast<int>(std::round(c * scale)) << " ";
    }
    prob_file << "\n";
  }
  prob_file << "EOF";
  prob_file.close();

  if (solveTSPLKH(par_file.c_str()) != 0) {
    ROS_ERROR_STREAM("[LocalLKH] solver failed for " << tsp_file);
    return false;
  }

  ifstream res_file(result_file);
  if (!res_file.is_open()) {
    ROS_ERROR_STREAM("[LocalLKH] cannot read " << result_file);
    return false;
  }
  string line;
  int result_dimension = -1;
  bool have_tour_section = false;
  while (getline(res_file, line)) {
    if (line.rfind("DIMENSION", 0) == 0) {
      const size_t colon = line.find(':');
      if (colon != string::npos) {
        try {
          result_dimension = std::stoi(line.substr(colon + 1));
        } catch (const std::exception&) {
          ROS_ERROR_STREAM("[LocalLKH] malformed DIMENSION in " << result_file);
          return false;
        }
      }
    }
    if (line == "TOUR_SECTION") {
      have_tour_section = true;
      break;
    }
  }
  if (!have_tour_section || result_dimension != dimension) {
    ROS_ERROR_STREAM("[LocalLKH] stale or malformed tour: expected dimension "
                     << dimension << ", got " << result_dimension);
    return false;
  }
  vector<int> rows;
  while (getline(res_file, line)) {
    int id = 0;
    try {
      id = std::stoi(line);
    } catch (const std::exception&) {
      ROS_ERROR_STREAM("[LocalLKH] malformed TOUR_SECTION entry in " << result_file);
      return false;
    }
    if (id == -1) break;
    rows.push_back(id - 1);
  }
  res_file.close();
  if (rows.empty()) return false;

  int current_pos = -1;
  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    if (rows[i] == 0) {
      current_pos = i;
      break;
    }
  }
  if (current_pos < 0) return false;
  for (int offset = 1; offset < static_cast<int>(rows.size()); ++offset) {
    const int node = rows[(current_pos + offset) % rows.size()];
    if (node <= 0 || node >= dimension) continue;
    tour_indices.push_back(node - 1);
  }
  if (tour_indices.size() != static_cast<size_t>(dimension - 1)) {
    ROS_ERROR_STREAM("[LocalLKH] incomplete tour, expected " << dimension - 1
                     << ", got " << tour_indices.size());
    return false;
  }
  return true;
}

// ============================================================
// Dual-threshold grid switching (Step 2)
// ============================================================

// ============================================================
// Single-grid timeout protection (Step 3)
// ============================================================

// ============================================================
// Batch pending cleanup (Step 5)
// ============================================================

void FastExplorationManager::refineLocalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
    const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_yaws,
    vector<Vector3d>& refined_pts, vector<double>& refined_yaws) {
  refined_pts.clear();
  refined_yaws.clear();
  if (n_points.empty()) return;

  // Layered graph: current state -> refinable frontier viewpoints -> optional
  // fixed exit viewpoint -> virtual terminal.  The exit singleton carries the
  // continuation cost from FUEL Eq. (6); the virtual terminal only closes the
  // graph search and contributes zero physical motion cost.
  GraphSearch<ViewNode> g_search;
  vector<ViewNode::Ptr> last_group, cur_group;
  auto start = make_shared<ViewNode>(cur_pos, cur_yaw[0]);
  start->vel_ = cur_vel;
  start->yaw_dot_ = cur_yaw[1];
  g_search.addNode(start);
  last_group.push_back(start);

  for (int i = 0; i < static_cast<int>(n_points.size()) &&
                  i < static_cast<int>(n_yaws.size()); ++i) {
    const int view_num = std::min(static_cast<int>(n_points[i].size()),
                                  static_cast<int>(n_yaws[i].size()));
    for (int j = 0; j < view_num; ++j) {
      auto view = make_shared<ViewNode>(n_points[i][j], n_yaws[i][j]);
      view->yaw_dot_ = 0.0;
      g_search.addNode(view);
      for (const auto& prev : last_group) g_search.addEdge(prev->id_, view->id_);
      cur_group.push_back(view);
    }
    // A missing group means the requested frontier no longer has a valid
    // view; stop here rather than adding a disconnected graph layer.
    if (cur_group.empty()) break;
    last_group.swap(cur_group);
    cur_group.clear();
  }

  if (last_group.size() == 1 && last_group.front() == start) return;
  auto terminal = make_shared<ViewNode>();
  terminal->terminal_ = true;
  g_search.addNode(terminal);
  for (const auto& last : last_group) g_search.addEdge(last->id_, terminal->id_);

  vector<ViewNode::Ptr> node_path;
  g_search.DijkstraSearch(start->id_, terminal->id_, node_path);
  if (node_path.empty()) return;
  for (const auto& node : node_path) {
    if (node == start || node == terminal) continue;
    refined_pts.push_back(node->pos_);
    refined_yaws.push_back(node->yaw_);
  }
  if (refined_pts.empty()) return;

  ed_->refined_tour_.clear();
  ed_->refined_tour_.push_back(cur_pos);
  // [MODIFIED BY CODEX]: refined_tour_ is visualization-only.  Display the
  // abstract Euclidean chain directly instead of running another A* for every
  // selected viewpoint; executable motion is generated and checked later.
  for (const auto& pt : refined_pts) {
    ed_->refined_tour_.push_back(pt);
  }
}

}  // namespace fast_planner
