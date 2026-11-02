#include <active_perception/grid_planner.h>
#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <queue>
#include <stack>
#include <cmath>
#include <limits>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace fast_planner {

GridPlanner::GridPlanner(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh) {
  edt_ = edt;
  sdf_map_ = edt_->sdf_map_;

  nh.param("grid_planner/grid_size", grid_size_, 20.0);
  nh.param("grid_planner/max_depth", max_depth_, 2);
  nh.param("grid_planner/max_level", max_depth_, max_depth_);
  nh.param("grid_planner/min_grid_size", min_grid_size_, 10.0);
  nh.param("grid_planner/min_unknown", min_unknown_, 500);
  nh.param("grid_planner/min_free", min_free_, 3000);
  nh.param("grid_planner/w_info", w_info_, 0.3);
  nh.param("grid_planner/w_frontier", w_frontier_, 0.2);
  nh.param("grid_planner/w_region", w_region_, 10.0);
  nh.param("grid_planner/consistent_cost", consistent_cost_, -5.0);
  nh.param("grid_planner/consistent_cost2", consistent_cost2_, 8.0);
  nh.param("grid_planner/astar_connect_max_dist", astar_connect_max_dist_, 12.0);
  nh.param("grid_planner/unreachable_edge_penalty", unreachable_edge_penalty_, 80.0);
  nh.param("grid_planner/boundary_touch_eps", boundary_touch_eps_, 0.15);
  nh.param("grid_planner/reachable_neighbor_only", reachable_neighbor_only_, false);
  nh.param("grid_planner/use_legacy_connectivity", use_legacy_connectivity_, true);
  nh.param("grid_planner/enable_dynamic_subdivision", enable_dynamic_subdivision_, true);
  nh.param("grid_planner/use_rush_subregions", use_rush_subregions_, false);
  nh.param("grid_planner/subdivide_only_current_grid", subdivide_only_current_grid_, true);
  nh.param("grid_planner/max_subdivide_per_update", max_subdivide_per_update_, 8);
  nh.param("grid_planner/subdivide_frontier_grids", subdivide_frontier_grids_, true);
  nh.param("grid_planner/max_frontier_subdivide_per_update",
           max_frontier_subdivide_per_update_, 2);
  nh.param("grid_planner/frontier_subdivide_max_dist",
           frontier_subdivide_max_dist_, 18.0);
  nh.param("grid_planner/fixed_root_x", fixed_root_x_, 2);
  nh.param("grid_planner/fixed_root_y", fixed_root_y_, 2);
  nh.param("grid_planner/enforce_root_2x2", enforce_root_2x2_, true);
  nh.param("grid_planner/only_adjacent_edge_for_tsp", only_adjacent_edge_for_tsp_, true);
  nh.param("grid_planner/non_adjacent_penalty", non_adjacent_penalty_, 200.0);
  nh.param("grid_planner/relevant_frontier_only", relevant_frontier_only_, true);
  nh.param("grid_planner/frontier_fallback_max_dist", frontier_fallback_max_dist_, 6.0);
  nh.param("grid_planner/hard_block_non_adjacent_edge", hard_block_non_adjacent_edge_, true);
  nh.param("grid_planner/hard_block_cost", hard_block_cost_, 5000.0);
  nh.param("grid_planner/coverage_radius", coverage_radius_, 4.5);
  nh.param("grid_planner/coverage_height", coverage_height_, 1.2);
  nh.param("grid_planner/coverage_complete_ratio", coverage_complete_ratio_, 0.85);
  nh.param("grid_planner/root_complete_coverage_ratio",
           root_complete_coverage_ratio_, 0.95);
  nh.param("grid_planner/root_complete_unknown_ratio",
           root_complete_unknown_ratio_, 0.05);
  nh.param("grid_planner/root_tail_gap_ratio",
           root_tail_gap_ratio_, 0.01);
  nh.param("grid_planner/root_tail_gap_free_voxels",
           root_tail_gap_free_voxels_, 120);
  // [MODIFIED BY CODEX]: global TSP retirement is based on a per-leaf
  // unknown-voxel baseline, not on transient frontier existence.
  nh.param("grid_planner/tsp_retire_unknown_ratio",
           tsp_retire_unknown_ratio_, 0.10);
  nh.param("grid_planner/coverage_voxel_stride", coverage_voxel_stride_, 0.15);

  // CCL parameters (adapted from FALCON)
  nh.param("grid_planner/use_ccl", use_ccl_, true);
  nh.param("grid_planner/ccl_step", ccl_step_, 2);
  nh.param("grid_planner/ccl_clearance_thresh", ccl_clearance_thresh_, 0.5);
  nh.param("grid_planner/ccl_min_component_voxels", ccl_min_component_voxels_, 10);
  nh.param("grid_planner/grid_state_sample_res", grid_state_sample_res_, 0.2);

  path_finder_.reset(new Astar);
  path_finder_->init(nh, edt_);
  // [MODIFIED BY CODEX]: The global-grid connectivity checker is a custom
  // layer and must not inherit FUEL-original's 1 ms frontier-ATSP estimator.
  // Preserve the grid layer's existing search budget and coarse resolution.
  path_finder_->max_search_time_ = 0.5;
  path_finder_->setResolution(0.4);
  voxel_res_ = sdf_map_->getResolution();

  nh.param("sdf_map/box_min_x", map_min_[0], -20.0);
  nh.param("sdf_map/box_min_y", map_min_[1], -20.0);
  nh.param("sdf_map/box_min_z", map_min_[2],  0.1);
  nh.param("sdf_map/box_max_x", map_max_[0],  20.0);
  nh.param("sdf_map/box_max_y", map_max_[1],  20.0);
  nh.param("sdf_map/box_max_z", map_max_[2],  1.2);

  initRootGrids();

  grid_pub_      = nh.advertise<visualization_msgs::Marker>("/grid_planner/grid_cells", 10);
  grid_tour_pub_ = nh.advertise<visualization_msgs::Marker>("/grid_planner/grid_tour", 10);
  grid_text_pub_ = nh.advertise<visualization_msgs::Marker>("/grid_planner/grid_labels", 10);
  grid_order_pub_ = nh.advertise<visualization_msgs::Marker>("/grid_planner/grid_tour_order", 10);

  ROS_WARN("[GridPlanner] %dx%d root grids, ids 0-%d",
           root_x_, root_y_, next_id_ - 1);
}

GridPlanner::~GridPlanner() {}

// ==================== Init: 4 root grids (2x2) ====================
void GridPlanner::initRootGrids() {
  Vector3d size = map_max_ - map_min_;

  // RUSH-style: single root grid covering the entire map.
  // The root subdivides dynamically along the longest side when it
  // contains frontiers.  Depth-1 children become the "subregions"
  // that the non-closed TSP orders.  This replaces the fixed 2×2
  // root layout which created artificial boundaries that cut through
  // continuous frontier regions.
  if (use_rush_subregions_) {
    root_x_ = 1;
    root_y_ = 1;
  } else if (enforce_root_2x2_) {
    root_x_ = std::max(1, fixed_root_x_);
    root_y_ = std::max(1, fixed_root_y_);
  } else {
    root_x_ = ceil(size[0] / grid_size_);
    root_y_ = ceil(size[1] / grid_size_);
  }
  for (int i = 0; i < 2; ++i)
    root_res_[i] = size[i] / double(i == 0 ? root_x_ : root_y_);
  root_res_[2] = size[2];

  root_grids_.resize(root_y_);
  for (int y = 0; y < root_y_; ++y) {
    root_grids_[y].resize(root_x_);
    for (int x = 0; x < root_x_; ++x) {
      GridCell g;
      g.id_       = next_id_++;
      g.parent_id_ = -1;
      g.depth_    = 0;
      g.active_   = true;
      g.is_relevant_ = false;
      g.region_id_ = -1;

      Vector3d lb(x * root_res_[0] + map_min_[0],
                  y * root_res_[1] + map_min_[1],
                  map_min_[2]);
      g.vmin_ = lb;
      g.vmax_ = Vector3d(lb[0] + root_res_[0], lb[1] + root_res_[1], map_max_[2]);
      g.center_ = Vector3d(lb[0] + 0.5 * root_res_[0],
                           lb[1] + 0.5 * root_res_[1],
                           0.5 * (map_min_[2] + map_max_[2]));

      Vector3d rt(g.vmax_[0], g.vmax_[1], lb[2]);
      Vector3d lt(lb[0], g.vmax_[1], lb[2]);
      Vector3d rb(g.vmax_[0], lb[1], lb[2]);
      g.vertices_ = {lb, rb, rt, lt};
      for (int j = 0; j < 4; ++j)
        g.normals_.push_back((g.vertices_[(j+1)%4] - g.vertices_[j]).normalized());

      g.unknown_num_ = (root_res_[0] * root_res_[1] * root_res_[2]) /
                       (voxel_res_ * voxel_res_ * voxel_res_);

      root_grids_[y][x] = g.id_;
      all_grids_.push_back(g);

      if (use_rush_subregions_ && g.depth_ == 0) {
        single_root_id_ = g.id_;
      }
    }
  }

  // RUSH: initialize subregion tracking for depth-0 root
  if (use_rush_subregions_ && single_root_id_ >= 0) {
    if (static_cast<int>(subregions_.size()) <= single_root_id_)
      subregions_.resize(single_root_id_ + 1);
    subregions_[single_root_id_].grid_id_ = single_root_id_;
    subregions_[single_root_id_].state_ = SubRegionState::ACTIVE;
    subregions_[single_root_id_].depth_ = 0;
    active_subregion_id_ = single_root_id_;
    printf("\\n==============================================\\n");
    printf("  RUSH MODE ACTIVE\\n");
    printf("  Single root grid %d covering %.1fx%.1f m\\n",
           single_root_id_, size[0], size[1]);
    printf("  Dynamic subregions replace fixed 2x2 roots\\n");
    printf("==============================================\\n\\n");
    fflush(stdout);
    ROS_ERROR("==============================================");
    ROS_ERROR("  RUSH MODE ACTIVE");
    ROS_ERROR("  Single root grid %d covering %.1f×%.1f m",
             single_root_id_, size[0], size[1]);
    ROS_ERROR("==============================================");
  }
}

// ==================== Spatial lookup ====================
int GridPlanner::findGridAt(const Vector3d& pos) {
  if (pos[0] < map_min_[0] || pos[1] < map_min_[1] ||
      pos[0] > map_max_[0] || pos[1] > map_max_[1])
    return -1;

  int best_id = -1;
  int best_depth = -1;
  for (const auto& cell : all_grids_) {
    if (!cell.active_) continue;
    if (pos[0] < cell.vmin_[0] || pos[0] > cell.vmax_[0] ||
        pos[1] < cell.vmin_[1] || pos[1] > cell.vmax_[1])
      continue;
    if (cell.depth_ > best_depth) {
      best_depth = cell.depth_;
      best_id = cell.id_;
    }
  }

  return best_id;
}

int GridPlanner::getRootGridId(int grid_id) const {
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size())) return -1;
  // RUSH: return subregion (depth-1) instead of full-map root (depth-0).
  // This makes all existing root-based candidate filtering, coverage
  // checks, and TSP ordering operate at the subregion level without
  // changing a single call site in the exploration manager.
  if (use_rush_subregions_) return getSubRegionId(grid_id);
  int cur = grid_id;
  int guard = 0;
  while (cur >= 0 && cur < static_cast<int>(all_grids_.size()) &&
         all_grids_[cur].parent_id_ >= 0 && guard++ < 16) {
    cur = all_grids_[cur].parent_id_;
  }
  return cur;
}

bool GridPlanner::isGridComplete(int grid_id) const {
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size())) return false;
  const auto& g = all_grids_[grid_id];
  if (!g.active_) return true;
  if (g.status_ == GridCell::PURE_OBSTACLE || g.status_ == GridCell::FINISHED) return true;

  // Exploration switching is frontier-driven. Unknown voxels without a
  // free/unknown boundary can be behind walls, so they must not force detours.
  if (!g.contained_frontier_ids_.empty() || g.unknown_frontier_num_ > 0) return false;
  return true;
}

bool GridPlanner::isRootGridComplete(int root_id) const {
  if (root_id < 0 || root_id >= static_cast<int>(all_grids_.size())) return false;
  bool has_active_child = false;
  int total_free = 0;
  int total_covered_free = 0;
  int total_unknown = 0;
  int total_sampled = 0;
  for (const auto& g : all_grids_) {
    if (!g.active_) continue;
    if (getRootGridId(g.id_) != root_id) continue;
    has_active_child = true;
    if (!g.contained_frontier_ids_.empty() || g.unknown_frontier_num_ > 0) {
      return false;
    }
    total_unknown += std::max(0, g.unknown_num_);
    total_sampled += std::max(0, g.free_num_ + g.unknown_num_ + g.occupied_num_);
    if (g.free_num_ > 0) {
      total_free += std::max(0, g.free_num_);
      total_covered_free += std::max(0, std::min(g.free_num_, g.covered_free_num_));
    }
  }
  if (!has_active_child) return false;
  if (total_sampled <= 0) return false;  // V11: unsampled root is NOT complete

  const double unknown_ratio = static_cast<double>(total_unknown) /
                               static_cast<double>(total_sampled);
  const double coverage = total_free > 1e-6
                              ? static_cast<double>(total_covered_free) /
                                    static_cast<double>(total_free)
                              : 1.0;
  return unknown_ratio <=
             std::max(0.0, std::min(1.0, root_complete_unknown_ratio_)) &&
         coverage >=
             std::max(0.0, std::min(1.0, root_complete_coverage_ratio_));
}

// ==================== RUSH: Subregion Management ====================

int GridPlanner::getSubRegionId(int grid_id) const {
  if (!use_rush_subregions_) return getRootGridId(grid_id);
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size())) return -1;
  // Walk up to depth 1 (subregion).  If the grid is at depth 0 or 1, return itself.
  // If depth > 1, return its depth-1 ancestor.
  int cur = grid_id;
  int guard = 0;
  while (cur >= 0 && cur < static_cast<int>(all_grids_.size()) &&
         all_grids_[cur].depth_ > 1 && guard++ < 16) {
    cur = all_grids_[cur].parent_id_;
  }
  return cur;
}

int GridPlanner::getSubRegionForPoint(const Vector3d& pos) {
  int gid = findGridAt(pos);
  if (gid < 0) return (active_subregion_id_ >= 0) ? active_subregion_id_ : single_root_id_;
  return getSubRegionId(gid);
}

bool GridPlanner::isSubRegionComplete(int sr_id) const {
  if (sr_id < 0 || sr_id >= static_cast<int>(all_grids_.size())) return true;

  // Check ALL active grids whose ancestor is this subregion.
  // This covers BOTH cases:
  //   a) subregion is active and holds frontiers directly (depth-1 active)
  //   b) subregion has subdivided (inactive) — depth-2 children are
  //      still assigned to this subregion via getSubRegionId, so we
  //      must check them for remaining frontiers.
  bool any_child = false;
  for (const auto& g : all_grids_) {
    if (!g.active_) continue;
    if (getSubRegionId(g.id_) != sr_id) continue;
    any_child = true;
    if (!g.contained_frontier_ids_.empty() || g.unknown_frontier_num_ > 0) return false;
  }
  return any_child;  // no active children with frontiers → complete
}

int GridPlanner::getSubRegionFrontierCount(int sr_id) const {
  if (sr_id < 0 || sr_id >= static_cast<int>(subregions_.size())) return 0;
  return subregions_[sr_id].frontier_count_;
}

double GridPlanner::getSubRegionCoverage(int sr_id) const {
  if (sr_id < 0 || sr_id >= static_cast<int>(all_grids_.size())) return 1.0;
  double total_free = 0.0, total_covered = 0.0;
  for (const auto& g : all_grids_) {
    if (!g.active_) continue;
    if (getSubRegionId(g.id_) != sr_id) continue;
    total_free += static_cast<double>(std::max(0, g.free_num_));
    total_covered += static_cast<double>(std::max(0, std::min(g.free_num_, g.covered_free_num_)));
  }
  return total_free > 1e-6 ? total_covered / total_free : 0.0;
}

void GridPlanner::updateSubRegionStates(const Vector3d& drone_pos) {
  if (!use_rush_subregions_) return;

  // Ensure subregions_ is large enough for all grids
  if (subregions_.size() < all_grids_.size())
    subregions_.resize(all_grids_.size());

  // Identify which grid is the "subregion" — depth-0 or depth-1 active grids
  // Count frontiers per subregion
  std::unordered_map<int, int> sr_frontier_count;
  std::unordered_set<int> all_sr_ids;  // ALL subregions, even those without frontiers
  for (const auto& g : all_grids_) {
    if (!g.active_) continue;
    const int sr_id = getSubRegionId(g.id_);
    if (sr_id < 0) continue;
    all_sr_ids.insert(sr_id);
    sr_frontier_count[sr_id] += static_cast<int>(g.contained_frontier_ids_.size());
  }

  // Determine active subregion from drone position
  const int drone_sr = getSubRegionForPoint(drone_pos);

  // Update active_subregion_id_ from drone position.
  // In RUSH soft-boundary mode the drone freely crosses subregion
  // boundaries — the active subregion simply reflects where the drone
  // physically is, for visualization and TSP starting-point purposes.
  // The exploration manager's auto-switch still handles subregion
  // completion tracking independently via active_root_grid_id_.
  if (drone_sr >= 0 && drone_sr < static_cast<int>(all_grids_.size())) {
    active_subregion_id_ = drone_sr;

    // Ensure the drone's subregion has a valid grid_id_ in subregions_,
    // otherwise computeSubRegionTSP will skip it (grid_id_ == -1 check).
    if (drone_sr < static_cast<int>(subregions_.size())) {
      subregions_[drone_sr].grid_id_ = drone_sr;
      subregions_[drone_sr].depth_ = all_grids_[drone_sr].depth_;
    }
  }

  // Update states for ALL subregions (with and without frontiers)
  for (int sr_id : all_sr_ids) {
    if (sr_id < 0 || sr_id >= static_cast<int>(subregions_.size())) continue;
    auto& info = subregions_[sr_id];
    info.grid_id_ = sr_id;
    info.frontier_count_ = sr_frontier_count[sr_id];  // 0 if not in map
    info.coverage_ratio_ = getSubRegionCoverage(sr_id);
    info.depth_ = (sr_id < static_cast<int>(all_grids_.size())) ? all_grids_[sr_id].depth_ : -1;

    if (drone_sr == sr_id) {
      info.state_ = SubRegionState::ACTIVE;
    } else if (isSubRegionComplete(sr_id)) {
      info.state_ = SubRegionState::COMPLETED;
    } else {
      info.state_ = SubRegionState::PENDING;
    }
  }

  // Clean up stale subregions: any subregion with grid_id_ set but NOT
  // in all_sr_ids has no active descendant grids → mark COMPLETED.
  // This handles the depth-0 root after subdivision (its children are
  // depth-1 subregions that map to themselves, not to the root).
  for (auto& info : subregions_) {
    if (info.grid_id_ < 0) continue;
    if (all_sr_ids.count(info.grid_id_) == 0) {
      info.state_ = SubRegionState::COMPLETED;
    }
  }

  // Diagnostic: log subregion states once per call
  {
    int n_active = 0, n_pending = 0, n_complete = 0;
    for (const auto& info : subregions_) {
      if (info.grid_id_ < 0) continue;
      switch (info.state_) {
        case SubRegionState::ACTIVE:    ++n_active;   break;
        case SubRegionState::PENDING:   ++n_pending;  break;
        case SubRegionState::COMPLETED: ++n_complete; break;
      }
    }
    ROS_WARN("[RUSH] SubRegions: active=%d (id=%d), pending=%d, complete=%d | drone_sr=%d",
             n_active, active_subregion_id_, n_pending, n_complete, drone_sr);
  }
}

void GridPlanner::computeSubRegionTSP(const Vector3d& cur_pos) {
  if (!use_rush_subregions_) {
    subregion_tour_.clear();
    return;
  }

  // Collect pending subregions (depth-1 or depth-0 if no split yet).
  // IMPORTANT: subregions may be inactive (subdivided into depth-2 children).
  // Their children still belong to them via getSubRegionId, so we must NOT
  // skip inactive subregions here — isSubRegionComplete already checks
  // children frontiers for inactive subregions, and the state_ reflects that.
  struct SR {
    int id;
    Vector3d center;
    double cost_from_cur = 0.0;
  };
  vector<SR> pending;
  for (const auto& info : subregions_) {
    if (info.grid_id_ < 0) continue;
    if (info.state_ == SubRegionState::COMPLETED) continue;
    const int gid = info.grid_id_;
    if (gid >= static_cast<int>(all_grids_.size())) continue;
    // Use geometric center from all_grids_ (still valid even when inactive).
    const auto& g = all_grids_[gid];
    SR sr;
    sr.id = gid;
    sr.center = g.center_;
    // [MODIFIED BY CODEX]: A subregion tour is global guidance, not an
    // executable path.  Using A* here made the greedy ordering launch a new
    // search for every remaining subregion at every step.  The selected local
    // frontier still passes the current-state A* reachability gate.
    sr.cost_from_cur = (cur_pos - sr.center).norm();
    pending.push_back(sr);
  }

  subregion_tour_.clear();
  if (pending.empty()) return;

  // Simple greedy TSP: start from closest to drone, then nearest-neighbor
  // (for ≤6 subregions this is sufficient; for more, use LKH or full TSP)
  std::sort(pending.begin(), pending.end(),
            [](const SR& a, const SR& b) { return a.cost_from_cur < b.cost_from_cur; });

  // Nearest-neighbor tour starting from active subregion
  if (active_subregion_id_ >= 0) {
    // Find active in pending list
    auto it = std::find_if(pending.begin(), pending.end(),
                           [&](const SR& s) { return s.id == active_subregion_id_; });
    if (it != pending.end()) {
      subregion_tour_.push_back(it->id);
      it->id = -1;  // mark as visited
    }
  }

  // Greedy nearest-neighbor for remaining
  while (true) {
    int best_id = -1;
    double best_cost = std::numeric_limits<double>::max();
    const Vector3d& ref = subregion_tour_.empty()
        ? cur_pos
        : all_grids_[subregion_tour_.back()].center_;

    for (auto& sr : pending) {
      if (sr.id < 0) continue;
      // [MODIFIED BY CODEX]: Keep the abstract subregion nearest-neighbour
      // tour Euclidean; collision-free executability is decided downstream.
      const double cost = (ref - sr.center).norm();
      if (cost < best_cost) {
        best_cost = cost;
        best_id = sr.id;
      }
    }
    if (best_id < 0) break;
    subregion_tour_.push_back(best_id);
    // Mark visited
    for (auto& sr : pending) {
      if (sr.id == best_id) { sr.id = -1; break; }
    }
  }

  if (!subregion_tour_.empty()) {
    std::stringstream ss;
    ss << "[RUSH] Subregion tour:";
    for (int id : subregion_tour_) ss << " " << id;
    ROS_WARN_STREAM(ss.str());
  }
}

int GridPlanner::getNextSubRegion(int current_sr_id) const {
  if (subregion_tour_.empty()) return -1;
  for (size_t i = 0; i < subregion_tour_.size(); ++i) {
    if (subregion_tour_[i] == current_sr_id && i + 1 < subregion_tour_.size())
      return subregion_tour_[i + 1];
  }
  // current_sr_id not found or is last in tour.
  // If current is not the first element, return first as fallback
  // (this handles the case where current is complete and was removed from tour).
  if (!subregion_tour_.empty() && subregion_tour_[0] != current_sr_id)
    return subregion_tour_[0];
  return -1;  // current is first and only element → no next
}

int GridPlanner::getLocalComponentId(int grid_id, const Vector3d& pos,
                                     bool allow_unknown) const {
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size())) return -1;
  const auto& g = all_grids_[grid_id];
  if (!g.active_) return -1;
  if (pos[0] < g.vmin_[0] || pos[0] > g.vmax_[0] ||
      pos[1] < g.vmin_[1] || pos[1] > g.vmax_[1]) {
    return -1;
  }

  const double res = std::max(0.35, grid_state_sample_res_);
  const double x0 = g.vmin_[0];
  const double y0 = g.vmin_[1];
  const int nx = std::max(1, static_cast<int>(std::ceil((g.vmax_[0] - g.vmin_[0]) / res)));
  const int ny = std::max(1, static_cast<int>(std::ceil((g.vmax_[1] - g.vmin_[1]) / res)));
  if (nx <= 0 || ny <= 0) return -1;

  auto sample_pos = [&](int ix, int iy) {
    Vector3d p;
    p[0] = std::min(g.vmax_[0] - 1e-3, x0 + (ix + 0.5) * res);
    p[1] = std::min(g.vmax_[1] - 1e-3, y0 + (iy + 0.5) * res);
    p[2] = std::min(std::max(pos[2], g.vmin_[2] + 0.05), g.vmax_[2] - 0.05);
    return p;
  };

  auto traversable_sample = [&](int ix, int iy) {
    const Vector3d p = sample_pos(ix, iy);
    if (!sdf_map_->isInBox(p)) return false;
    Eigen::Vector3i idx;
    sdf_map_->posToIndex(p, idx);
    if (!sdf_map_->isInMap(idx)) return false;
    if (sdf_map_->getInflateOccupancy(idx) == 1) return false;
    const auto occ = sdf_map_->getOccupancy(idx);
    if (allow_unknown) {
      return occ == SDFMap::FREE || occ == SDFMap::UNKNOWN;
    }
    return occ == SDFMap::FREE;
  };

  const auto pos_to_cell = [&](const Vector3d& p) {
    const double xx = std::min(std::max(p[0], g.vmin_[0]), g.vmax_[0] - 1e-6);
    const double yy = std::min(std::max(p[1], g.vmin_[1]), g.vmax_[1] - 1e-6);
    const int ix = std::min(nx - 1, std::max(0, static_cast<int>((xx - x0) / res)));
    const int iy = std::min(ny - 1, std::max(0, static_cast<int>((yy - y0) / res)));
    return std::pair<int, int>(ix, iy);
  };

  const auto target = pos_to_cell(pos);
  const int start_x = target.first;
  const int start_y = target.second;
  if (!traversable_sample(start_x, start_y)) return -1;

  const long long cache_key =
      (static_cast<long long>(grid_id) << 32) ^
      (static_cast<long long>(start_x) << 16) ^
      static_cast<long long>(start_y) ^
      (allow_unknown ? (1LL << 48) : 0LL);
  const auto cache_it = local_component_cache_.find(cache_key);
  if (cache_it != local_component_cache_.end()) return cache_it->second;

  std::vector<int> labels(nx * ny, -1);
  int comp_id = 0;
  std::queue<std::pair<int, int>> q;
  const int start_idx = start_y * nx + start_x;
  labels[start_idx] = comp_id;
  q.push({start_x, start_y});

  auto push_if_free = [&](int nx_, int ny_, int cid) {
    if (nx_ < 0 || ny_ < 0 || nx_ >= nx || ny_ >= ny) return;
    const int flat = ny_ * nx + nx_;
    if (labels[flat] >= 0) return;
    if (!traversable_sample(nx_, ny_)) return;
    labels[flat] = cid;
    q.push({nx_, ny_});
  };

  const auto bfs_fill = [&](int sx, int sy, int cid) {
    while (!q.empty()) {
      const auto cur = q.front();
      q.pop();
      const int cx = cur.first;
      const int cy = cur.second;
      push_if_free(cx - 1, cy, cid);
      push_if_free(cx + 1, cy, cid);
      push_if_free(cx, cy - 1, cid);
      push_if_free(cx, cy + 1, cid);
    }
  };

  bfs_fill(start_x, start_y, comp_id);

  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const int flat = iy * nx + ix;
      if (labels[flat] >= 0) continue;
      if (!traversable_sample(ix, iy)) continue;
      ++comp_id;
      labels[flat] = comp_id;
      q.push({ix, iy});
      bfs_fill(ix, iy, comp_id);
    }
  }

  const int result = labels[start_idx];
  local_component_cache_[cache_key] = result;
  return result;
}

bool GridPlanner::isVoxelCovered(const Eigen::Vector3i& idx) const {
  if (!sdf_map_->isInMap(idx)) return false;
  return covered_voxel_addrs_.count(sdf_map_->toAddress(idx)) > 0;
}

void GridPlanner::updatePathCoverage(const Vector3d& cur_pos) {
  if (!std::isfinite(cur_pos[0]) || !std::isfinite(cur_pos[1]) ||
      !std::isfinite(cur_pos[2]) || !sdf_map_->isInBox(cur_pos)) {
    return;
  }
  const int step_xy = std::max(1, static_cast<int>(std::ceil(coverage_radius_ / voxel_res_)));
  const int step_z = std::max(1, static_cast<int>(std::ceil(coverage_height_ / voxel_res_)));
  const int stride = std::max(1, static_cast<int>(std::round(coverage_voxel_stride_ / voxel_res_)));
  const double r2 = coverage_radius_ * coverage_radius_;

  auto mark_disk = [&](const Vector3d& p) {
    if (!sdf_map_->isInBox(p)) return;
    Eigen::Vector3i cidx;
    sdf_map_->posToIndex(p, cidx);
    for (int dx = -step_xy; dx <= step_xy; dx += stride) {
      for (int dy = -step_xy; dy <= step_xy; dy += stride) {
        const double xy2 = double(dx * dx + dy * dy) * voxel_res_ * voxel_res_;
        if (xy2 > r2) continue;
        for (int dz = -step_z; dz <= step_z; dz += stride) {
          Eigen::Vector3i idx = cidx + Eigen::Vector3i(dx, dy, dz);
          if (!sdf_map_->isInBox(idx)) continue;
          if (sdf_map_->getInflateOccupancy(idx) == 1) continue;
          if (sdf_map_->getOccupancy(idx) != SDFMap::FREE) continue;
          covered_voxel_addrs_.insert(sdf_map_->toAddress(idx));
        }
      }
    }
  };

  auto segment_clear = [&](const Vector3d& a, const Vector3d& b) {
    const double seg_len = (b - a).norm();
    if (seg_len < 1e-6) return true;
    const double sample = std::max(0.08, 0.5 * voxel_res_);
    const int n = std::max(1, static_cast<int>(std::ceil(seg_len / sample)));
    for (int i = 1; i <= n; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(n);
      const Vector3d p = a + t * (b - a);
      if (!sdf_map_->isInBox(p)) return false;
      Eigen::Vector3i idx;
      sdf_map_->posToIndex(p, idx);
      if (!sdf_map_->isInMap(idx)) return false;
      if (sdf_map_->getInflateOccupancy(idx) == 1 ||
          sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        return false;
      }
    }
    return true;
  };

  if (!std::isfinite(last_coverage_pos_[0])) {
    mark_disk(cur_pos);
    last_coverage_pos_ = cur_pos;
    return;
  }

  const Vector3d delta = cur_pos - last_coverage_pos_;
  const double dist = delta.norm();
  const double step = std::max(0.2, 0.5 * coverage_voxel_stride_);
  if (dist < step) {
    mark_disk(cur_pos);
    last_coverage_pos_ = cur_pos;
    return;
  }

  const int n = std::max(1, static_cast<int>(std::ceil(dist / step)));
  Vector3d last_safe = last_coverage_pos_;
  for (int i = 1; i <= n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(n);
    const Vector3d p = last_coverage_pos_ + t * delta;
    if (!segment_clear(last_safe, p)) break;
    last_safe = p;
    mark_disk(p);
  }
  last_coverage_pos_ = last_safe;
}

// ==================== Subdivision ====================
void GridPlanner::subdivideGrid(int grid_id) {
  if (grid_id < 0 || grid_id >= (int)all_grids_.size()) return;
  auto& parent_ref = all_grids_[grid_id];
  if (parent_ref.depth_ >= max_depth_) return;

  // IMPORTANT:
  // all_grids_.push_back(child) may reallocate and invalidate references.
  // Cache all parent fields needed by child generation before push_back loop.
  const int parent_depth = parent_ref.depth_;
  const Vector3d parent_vmin = parent_ref.vmin_;
  const Vector3d parent_vmax = parent_ref.vmax_;
  const double sub_w = (parent_vmax[0] - parent_vmin[0]) * 0.5;
  const double sub_h = (parent_vmax[1] - parent_vmin[1]) * 0.5;
  const double min_side = std::max(0.1, min_grid_size_);
  if (std::min(sub_w, sub_h) < min_side - 1e-6) {
    return;
  }

  parent_ref.active_ = false;
  const int first_child_id = next_id_;

  for (int sy = 0; sy < 2; ++sy) {
    for (int sx = 0; sx < 2; ++sx) {
      GridCell child;
      child.id_        = next_id_++;
      child.parent_id_  = grid_id;
      child.depth_     = parent_depth + 1;
      child.active_    = true;
      child.is_relevant_ = false;
      child.region_id_  = -1;

      child.vmin_ = Vector3d(parent_vmin[0] + sx * sub_w,
                              parent_vmin[1] + sy * sub_h,
                              parent_vmin[2]);
      child.vmax_ = Vector3d(child.vmin_[0] + sub_w,
                              child.vmin_[1] + sub_h,
                              parent_vmax[2]);
      child.center_ = Vector3d(child.vmin_[0] + 0.5 * sub_w,
                                child.vmin_[1] + 0.5 * sub_h,
                                0.5 * (child.vmin_[2] + child.vmax_[2]));

      Vector3d lb = child.vmin_;
      Vector3d rt(child.vmax_[0], child.vmax_[1], lb[2]);
      Vector3d lt(lb[0], child.vmax_[1], lb[2]);
      Vector3d rb(child.vmax_[0], lb[1], lb[2]);
      child.vertices_ = {lb, rb, rt, lt};
      for (int j = 0; j < 4; ++j)
        child.normals_.push_back((child.vertices_[(j+1)%4] - child.vertices_[j]).normalized());

      child.unknown_num_ = (sub_w * sub_h * (child.vmax_[2] - child.vmin_[2])) /
                           (voxel_res_ * voxel_res_ * voxel_res_);

      all_grids_.push_back(child);
    }
  }

  ROS_WARN("[GridPlanner] subdivide %d -> children %d-%d (depth %d)",
           grid_id, first_child_id, next_id_ - 1, parent_depth + 1);
}

// ==================== Main update ====================
void GridPlanner::updateGrids() {
  updateGrids(Vector3d(std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN()));
}

void GridPlanner::updateGrids(const Vector3d& cur_pos) {
  pair_reachability_cache_.clear();
  local_component_cache_.clear();
  vis_astar_path_cache_.clear();  // invalidate A* visualization cache on map change
  updatePathCoverage(cur_pos);

  Vector3d upd_min, upd_max;
  sdf_map_->getUpdatedBox(upd_min, upd_max);

  auto overlap = [](const Vector3d& a1, const Vector3d& a2,
                     const Vector3d& b1, const Vector3d& b2) {
    for (int m = 0; m < 2; ++m) {
      if (std::max(a1[m], b1[m]) > std::min(a2[m], b2[m]) + 1e-3) return false;
    }
    return true;
  };

  const bool has_valid_pos = std::isfinite(cur_pos[0]) && std::isfinite(cur_pos[1]);
  const int cur_gid = has_valid_pos ? findGridAt(cur_pos) : -1;

  // Single-pass: update all FOV grids, track which need subdivision
  vector<int> to_subdivide;
  for (auto& g : all_grids_) {
    if (!g.active_) continue;
    if (!overlap(g.vmin_, g.vmax_, upd_min, upd_max)) continue;
    updateGridCell(g.id_);

    if (enable_dynamic_subdivision_ && g.active_ && g.depth_ < max_depth_ && g.need_subdivide_)
      to_subdivide.push_back(g.id_);
  }

  // Ensure current grid is freshly evaluated even if it is outside update box.
  if (enable_dynamic_subdivision_ && cur_gid >= 0 && cur_gid < static_cast<int>(all_grids_.size())) {
    auto& cg = all_grids_[cur_gid];
    if (cg.active_ && cg.depth_ < max_depth_) {
      updateGridCell(cur_gid);
    }
  }

  // User-required mode: prioritize current grid, but also refine a small
  // number of frontier-bearing grids. A strict current-only split leaves the
  // global layer making decisions on overly coarse cells for several cycles.
  if (enable_dynamic_subdivision_ && subdivide_only_current_grid_) {
    to_subdivide.clear();
    if (cur_gid >= 0 && cur_gid < static_cast<int>(all_grids_.size())) {
      auto& cg = all_grids_[cur_gid];
      if (cg.active_ && cg.depth_ < max_depth_) {
        // Split current grid directly if it is large/root, or by need_subdivide evidence.
        if (cg.depth_ == 0 || cg.need_subdivide_) {
          to_subdivide.push_back(cur_gid);
        }
      }
    }

    if (subdivide_frontier_grids_) {
      vector<std::pair<double, int>> frontier_candidates;
      frontier_candidates.reserve(all_grids_.size());
      for (const auto& g : all_grids_) {
        if (!g.active_ || g.depth_ >= max_depth_) continue;
        if (g.id_ == cur_gid) continue;
        if (g.contained_frontier_ids_.empty() && g.unknown_frontier_num_ <= 0) continue;
        if (!(g.depth_ == 0 || g.need_subdivide_)) continue;
        double dist = 0.0;
        if (has_valid_pos) {
          dist = (g.center_ - cur_pos).norm();
          if (frontier_subdivide_max_dist_ > 0.0 &&
              dist > frontier_subdivide_max_dist_) {
            continue;
          }
        }
        const double frontier_bonus =
            -0.25 * static_cast<double>(
                std::max<int>(g.unknown_frontier_num_,
                              static_cast<int>(g.contained_frontier_ids_.size())));
        frontier_candidates.emplace_back(dist + frontier_bonus, g.id_);
      }
      std::sort(frontier_candidates.begin(), frontier_candidates.end());
      const int extra_limit = std::max(0, max_frontier_subdivide_per_update_);
      int added = 0;
      for (const auto& cand : frontier_candidates) {
        if (added >= extra_limit) break;
        if (std::find(to_subdivide.begin(), to_subdivide.end(), cand.second) !=
            to_subdivide.end()) {
          continue;
        }
        to_subdivide.push_back(cand.second);
        ++added;
      }
    }
  }

  int subdivide_limit = max_subdivide_per_update_;
  if (subdivide_only_current_grid_) {
    subdivide_limit =
        std::max(1, max_subdivide_per_update_) +
        std::max(0, max_frontier_subdivide_per_update_);
  }
  if (subdivide_limit > 0 &&
      static_cast<int>(to_subdivide.size()) > subdivide_limit) {
    to_subdivide.resize(subdivide_limit);
  }

  for (int gid : to_subdivide)
    subdivideGrid(gid);

  for (auto& g : all_grids_) {
    if (g.active_) g.is_relevant_ = isRelevant(g);
  }

  // CCL: update drone component for all active grids
  if (use_ccl_) updateDroneComponent(cur_pos);

  // [MODIFIED BY CODEX]: inputFrontiersDetailed() runs immediately after
  // updateGrids() and clusters regions after mapping the current frontiers.
  // Do not perform the same reachability clustering once here with stale
  // frontier ownership and then repeat it again in the same planning round.

  // NOTE: updateSubRegionStates is deliberately NOT called here.
  // At this point frontiers have not yet been mapped (inputFrontiersDetailed
  // runs later in planExploreMotion), so sr_frontier_count would be empty
  // and active_subregion_id_ would be left stale.  The exploration manager
  // calls updateSubRegionStates at the correct point (after frontier mapping).
}

void GridPlanner::updateGridCell(int grid_id) {
  auto& g = all_grids_[grid_id];
  if (!g.active_) return;

  double res = std::max(voxel_res_, grid_state_sample_res_);
  Vector3d unknown_center = Vector3d::Zero();
  g.unknown_num_ = 0;
  g.free_num_ = 0;
  g.occupied_num_ = 0;
  g.covered_free_num_ = 0;

  auto inside = [&](const Vector3d& v) {
    for (int m = 0; m < 4; ++m)
      if ((v - g.vertices_[m]).dot(g.normals_[m]) <= 0.0) return false;
    return true;
  };

  for (double x = g.vmin_[0]; x <= g.vmax_[0]; x += res)
    for (double y = g.vmin_[1]; y <= g.vmax_[1]; y += res)
      for (double z = g.vmin_[2]; z <= g.vmax_[2]; z += res) {
        Vector3d pos(x, y, z);
        if (!inside(pos)) continue;
        int st = sdf_map_->getOccupancy(pos);
        Eigen::Vector3i idx;
        sdf_map_->posToIndex(pos, idx);
        if (st == SDFMap::FREE) {
          g.free_num_++;
          if (isVoxelCovered(idx)) g.covered_free_num_++;
        } else if (st == SDFMap::OCCUPIED ||
                   sdf_map_->getInflateOccupancy(pos) == 1) {
          g.occupied_num_++;
        } else if (st == SDFMap::UNKNOWN) {
          unknown_center =
              (unknown_center * g.unknown_num_ + pos) / (g.unknown_num_ + 1);
          g.unknown_num_++;
        }
      }

  if (g.unknown_num_ > 0) {
    g.center_ = unknown_center;
  } else {
    // No unknown observed in this update: keep geometric center stable.
    g.center_ = 0.5 * (g.vmin_ + g.vmax_);
  }

  // [MODIFIED BY CODEX]: latch the unknown workload only after a real map
  // scan.  Newly created/split grids may already be partly observed, so the
  // geometric/theoretical voxel count is not a valid baseline.
  const int sampled_voxels = g.free_num_ + g.unknown_num_ + g.occupied_num_;
  if (!g.unknown_baseline_valid_ && sampled_voxels > 0) {
    g.initial_unknown_num_ = std::max(1, g.unknown_num_);
    g.unknown_baseline_valid_ = true;
  }
  if (g.unknown_baseline_valid_) {
    const double unknown_ratio = static_cast<double>(g.unknown_num_) /
                                 static_cast<double>(std::max(1, g.initial_unknown_num_));
    g.tsp_retired_ = unknown_ratio <= std::max(0.0, std::min(1.0, tsp_retire_unknown_ratio_));
  }

  g.is_relevant_ = isRelevant(g);
  g.coverage_ratio_ =
      g.free_num_ > 0 ? double(g.covered_free_num_) / double(g.free_num_) : 0.0;
  g.covered_by_path_ = g.free_num_ > 0 && g.coverage_ratio_ >= coverage_complete_ratio_;
  const bool residual_frontier_exists =
      !g.contained_frontier_ids_.empty() || g.unknown_frontier_num_ > 0;
  g.need_subdivide_ =
      (g.depth_ < max_depth_ &&
       (g.free_num_ > min_free_ ||
        (g.covered_by_path_ && residual_frontier_exists)));

  if (g.free_num_ == 0 && g.unknown_num_ == 0 && g.occupied_num_ > 0) {
    g.status_ = GridCell::PURE_OBSTACLE;
  } else if (g.unknown_num_ == 0 && g.unknown_frontier_num_ == 0 && g.covered_by_path_) {
    g.status_ = GridCell::FINISHED;
  } else if (isRelevant(g)) {
    g.status_ = GridCell::EFFECTIVE_UNKNOWN;
  } else {
    g.status_ = GridCell::INACTIVE;
  }

  // CCL: detect obstacle-split components (FALCON-style connectivity decomposition)
  if (use_ccl_ && g.status_ == GridCell::EFFECTIVE_UNKNOWN && g.free_num_ > 0) {
    runCCLInGrid(grid_id);
  } else if (use_ccl_) {
    // Clear stale CCL data when grid leaves EFFECTIVE_UNKNOWN state
    g.components_.clear();
    g.drone_component_id_ = -1;
    g.is_split_ = false;
  }
}

bool GridPlanner::isRelevant(const GridCell& g) {
  const bool pure_obstacle =
      (g.free_num_ == 0 && g.unknown_num_ == 0 && g.occupied_num_ > 0);
  const bool finished =
      (g.unknown_num_ == 0 && g.unknown_frontier_num_ == 0 && g.covered_by_path_);
  if (pure_obstacle || finished) {
    return false;
  }
  if (relevant_frontier_only_) {
    return !g.contained_frontier_ids_.empty();
  }
  return g.unknown_num_ >= min_unknown_ || !g.contained_frontier_ids_.empty();
}

// ==================== Active grids ====================
void GridPlanner::getActiveGrids(vector<int>& grid_ids) {
  grid_ids.clear();
  for (auto& g : all_grids_)
    if (g.active_ && g.is_relevant_)
      grid_ids.push_back(g.id_);
}

void GridPlanner::getActiveGridCenters(vector<Vector3d>& centers, vector<int>& grid_ids) {
  getActiveGrids(grid_ids);
  centers.resize(grid_ids.size());
  for (size_t i = 0; i < grid_ids.size(); ++i)
    centers[i] = all_grids_[grid_ids[i]].center_;
}

void GridPlanner::getGlobalTspGridCenters(vector<Vector3d>& centers,
                                          vector<int>& grid_ids) {
  // [MODIFIED BY CODEX]: A global grid represents unknown workload, not an
  // immediately executable viewpoint.  Keep active unexplored leaves even
  // before a frontier/free A* anchor exists, otherwise an all-unknown map
  // would yield an empty TSP at startup.
  grid_ids.clear();
  centers.clear();
  for (const auto& g : all_grids_) {
    if (!g.active_ || g.status_ == GridCell::PURE_OBSTACLE || g.tsp_retired_) continue;
    grid_ids.push_back(g.id_);
    centers.push_back(g.center_);
  }
}

// ==================== Frontier mapping ====================
void GridPlanner::inputFrontiers(const vector<Vector3d>& avgs) {
  inputFrontiersDetailed(avgs, {});
}

void GridPlanner::inputFrontiersDetailed(const vector<Vector3d>& avgs,
                                         const vector<Vector3d>& views) {
  for (auto& g : all_grids_) {
    g.contained_frontier_ids_.clear();
    g.unknown_frontier_num_ = 0;
  }

  auto add_frontier_to_grid = [&](int fid, int gid) {
    if (gid < 0 || gid >= static_cast<int>(all_grids_.size())) return;
    if (!all_grids_[gid].active_) return;
    all_grids_[gid].contained_frontier_ids_[fid] = 1;

    // CCL: also assign frontier to its connected component
    if (use_ccl_ && !all_grids_[gid].components_.empty()) {
      int comp = findComponentForPoint(gid, avgs[fid]);
      if (comp >= 0 && comp < static_cast<int>(all_grids_[gid].components_.size())) {
        all_grids_[gid].components_[comp].frontier_ids_.push_back(fid);
      }
    }
  };

  for (int i = 0; i < static_cast<int>(avgs.size()); ++i) {
    std::unordered_set<int> member_grids;

    int gid_avg = findGridAt(avgs[i]);
    if (gid_avg >= 0) member_grids.insert(gid_avg);

    if (i < static_cast<int>(views.size())) {
      int gid_view = findGridAt(views[i]);
      if (gid_view >= 0) member_grids.insert(gid_view);

      Vector3d mid = 0.5 * (avgs[i] + views[i]);
      int gid_mid = findGridAt(mid);
      if (gid_mid >= 0) member_grids.insert(gid_mid);
    }

    if (member_grids.empty()) {
      continue;
    }

    int primary_gid = -1;
    double best_score = std::numeric_limits<double>::infinity();
    for (int gid : member_grids) {
      if (gid < 0 || gid >= static_cast<int>(all_grids_.size())) continue;
      const auto& g = all_grids_[gid];
      if (!g.active_) continue;
      const double score = (g.center_ - avgs[i]).squaredNorm();
      if (score < best_score) {
        best_score = score;
        primary_gid = gid;
      }
    }
    if (primary_gid >= 0) {
      // Single-owner mapping:
      // A frontier belongs to one primary grid only.  Multi-grid duplication
      // makes the grid TSP and residual statistics unstable, because the same
      // frontier can keep inflating several cells on both sides of a wall.
      //
      // The local planner still has access to the frontier's viewpoint set,
      // so we do not need to duplicate ownership to preserve executability.
      add_frontier_to_grid(i, primary_gid);
    }
  }

  for (auto& g : all_grids_) {
    if (!g.active_) {
      g.status_ = GridCell::INACTIVE;
      continue;
    }
    g.unknown_frontier_num_ = static_cast<int>(g.contained_frontier_ids_.size());
    if (g.free_num_ == 0 && g.unknown_num_ == 0 && g.occupied_num_ > 0) {
      g.status_ = GridCell::PURE_OBSTACLE;
    } else if (g.unknown_num_ == 0 && g.unknown_frontier_num_ == 0 && g.covered_by_path_) {
      g.status_ = GridCell::FINISHED;
    } else if (isRelevant(g)) {
      g.status_ = GridCell::EFFECTIVE_UNKNOWN;
    } else {
      g.status_ = GridCell::INACTIVE;
    }
    g.is_relevant_ = isRelevant(g);
  }
  clusterRegions();
}

void GridPlanner::getFrontiersInGrids(const vector<int>& grid_ids, vector<int>& ftr_ids) {
  ftr_ids.clear();
  unordered_map<int, bool> seen;
  for (int gid : grid_ids) {
    if (gid < 0 || gid >= static_cast<int>(all_grids_.size())) continue;
    if (!all_grids_[gid].active_) continue;
    for (auto& kv : all_grids_[gid].contained_frontier_ids_)
      if (!seen[kv.first]) { ftr_ids.push_back(kv.first); seen[kv.first] = true; }
  }
}

bool GridPlanner::areGridsAdjacentLegacy(int id1, int id2) const {
  if (id1 == id2) return false;
  if (id1 < 0 || id2 < 0 ||
      id1 >= static_cast<int>(all_grids_.size()) ||
      id2 >= static_cast<int>(all_grids_.size())) {
    return false;
  }
  const auto& g1 = all_grids_[id1];
  const auto& g2 = all_grids_[id2];
  if (!g1.active_ || !g2.active_) return false;

  // Legacy mode keeps dynamic split strategy, but adjacency is still geometric
  // edge-sharing to avoid cross-root dense false adjacency.
  return shareBoundary(g1, g2);
}

bool GridPlanner::shareBoundary(const GridCell& g1, const GridCell& g2) const {
  const double eps = boundary_touch_eps_;
  const double ox = std::min(g1.vmax_[0], g2.vmax_[0]) - std::max(g1.vmin_[0], g2.vmin_[0]);
  const double oy = std::min(g1.vmax_[1], g2.vmax_[1]) - std::max(g1.vmin_[1], g2.vmin_[1]);

  const bool touch_x = std::abs(g1.vmax_[0] - g2.vmin_[0]) <= eps ||
                       std::abs(g2.vmax_[0] - g1.vmin_[0]) <= eps;
  const bool touch_y = std::abs(g1.vmax_[1] - g2.vmin_[1]) <= eps ||
                       std::abs(g2.vmax_[1] - g1.vmin_[1]) <= eps;

  const bool share_x_edge = touch_x && oy > eps;
  const bool share_y_edge = touch_y && ox > eps;
  return share_x_edge || share_y_edge;
}

// ==================== Adjacency (boundary-sharing neighbour) ====================
bool GridPlanner::areGridsAdjacent(int id1, int id2) {
  if (use_legacy_connectivity_) {
    return areGridsAdjacentLegacy(id1, id2);
  }

  if (id1 == id2) return false;
  if (id1 < 0 || id2 < 0 ||
      id1 >= static_cast<int>(all_grids_.size()) ||
      id2 >= static_cast<int>(all_grids_.size())) {
    return false;
  }
  const auto& g1 = all_grids_[id1];
  const auto& g2 = all_grids_[id2];
  if (!g1.active_ || !g2.active_) return false;
  return shareBoundary(g1, g2);
}

long long GridPlanner::makePairKey(int id1, int id2) const {
  if (id1 > id2) std::swap(id1, id2);
  return (static_cast<long long>(id1) << 32) |
         static_cast<unsigned int>(id2);
}

bool GridPlanner::findGridAnchor(const GridCell& g, Vector3d& anchor) {
  auto is_free = [&](const Vector3d& p) {
    if (!sdf_map_->isInBox(p)) return false;
    if (sdf_map_->getInflateOccupancy(p) == 1) return false;
    return sdf_map_->getOccupancy(p) == SDFMap::FREE;
  };

  Vector3d center = g.center_;
  center[2] = std::min(std::max(center[2], map_min_[2] + 0.3), map_max_[2] - 0.05);
  if (is_free(center)) {
    anchor = center;
    return true;
  }

  const Vector3d half = 0.5 * (g.vmax_ - g.vmin_);
  constexpr double kPi = 3.14159265358979323846;
  for (double ratio : {0.25, 0.45, 0.65}) {
    for (int k = 0; k < 16; ++k) {
      const double a = 2.0 * kPi * double(k) / 16.0;
      Vector3d p = center;
      p[0] += ratio * half[0] * std::cos(a);
      p[1] += ratio * half[1] * std::sin(a);
      if (is_free(p)) {
        anchor = p;
        return true;
      }
    }
  }

  const int samples = 5;
  for (int ix = 0; ix < samples; ++ix) {
    for (int iy = 0; iy < samples; ++iy) {
      Vector3d p;
      p[0] = g.vmin_[0] + (double(ix) + 0.5) / double(samples) * (g.vmax_[0] - g.vmin_[0]);
      p[1] = g.vmin_[1] + (double(iy) + 0.5) / double(samples) * (g.vmax_[1] - g.vmin_[1]);
      p[2] = center[2];
      if (is_free(p)) {
        anchor = p;
        return true;
      }
    }
  }
  return false;
}

bool GridPlanner::findGridAnchorById(int grid_id, Vector3d& anchor) {
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size())) return false;
  return findGridAnchor(all_grids_[grid_id], anchor);
}

bool GridPlanner::isGridPairReachableInternal(int id1, int id2, double& path_len) {
  if (id1 == id2) {
    path_len = 0.0;
    return true;
  }

  const long long key = makePairKey(id1, id2);
  auto it = pair_reachability_cache_.find(key);
  if (it != pair_reachability_cache_.end()) {
    path_len = it->second;
    return path_len >= 0.0;
  }

  if (id1 < 0 || id2 < 0 ||
      id1 >= static_cast<int>(all_grids_.size()) ||
      id2 >= static_cast<int>(all_grids_.size())) {
    pair_reachability_cache_[key] = -1.0;
    path_len = -1.0;
    return false;
  }
  const auto& g1 = all_grids_[id1];
  const auto& g2 = all_grids_[id2];
  if (!g1.active_ || !g2.active_) {
    pair_reachability_cache_[key] = -1.0;
    path_len = -1.0;
    return false;
  }

  Vector3d a1, a2;
  if (!findGridAnchor(g1, a1) || !findGridAnchor(g2, a2)) {
    pair_reachability_cache_[key] = -1.0;
    path_len = -1.0;
    return false;
  }

  path_finder_->reset();
  if (path_finder_->search(a1, a2) == Astar::REACH_END) {
    path_len = path_finder_->pathLength(path_finder_->getPath());
    pair_reachability_cache_[key] = path_len;
    return true;
  }

  pair_reachability_cache_[key] = -1.0;
  path_len = -1.0;
  return false;
}

bool GridPlanner::isGridReachable(int id1, int id2, double* path_len) {
  double pl = -1.0;
  const bool ok = isGridPairReachableInternal(id1, id2, pl);
  if (path_len != nullptr) *path_len = pl;
  return ok;
}

void GridPlanner::getNeighborGridIds(int grid_id, int hop, vector<int>& grid_ids,
                                     bool include_self) {
  grid_ids.clear();
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size()) ||
      !all_grids_[grid_id].active_) {
    return;
  }

  hop = std::max(0, hop);
  std::queue<std::pair<int, int>> q;
  std::unordered_set<int> visited;
  visited.insert(grid_id);
  q.push({grid_id, 0});

  while (!q.empty()) {
    const auto cur = q.front();
    q.pop();
    const int gid = cur.first;
    const int dep = cur.second;

    if (include_self || gid != grid_id) {
      grid_ids.push_back(gid);
    }
    if (dep >= hop) continue;

    for (const auto& g : all_grids_) {
      if (!g.active_) continue;
      const int ng = g.id_;
      if (visited.count(ng)) continue;
      if (!areGridsAdjacent(gid, ng)) continue;
      // Always verify path-reachability: geometrically adjacent grids
      // separated by walls must not be treated as neighbors.
      double dummy_len = -1.0;
      if (!isGridPairReachableInternal(gid, ng, dummy_len)) continue;
      visited.insert(ng);
      q.push({ng, dep + 1});
    }
  }
}

// ==================== Region clustering ====================
void GridPlanner::clusterRegions() {
  for (auto& g : all_grids_) g.region_id_ = -1;
  regions_.clear();

  vector<int> active;
  getActiveGrids(active);
  int N = active.size();
  if (N == 0) return;

  vector<int> parent(N);
  for (int i = 0; i < N; ++i) parent[i] = i;
  auto find = [&](int x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  auto unite = [&](int a, int b) {
    int ra = find(a), rb = find(b);
    if (ra != rb) parent[ra] = rb;
  };

  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      if (areGridsAdjacent(active[i], active[j])) {
        // Always verify path-reachability between geometrically adjacent grids.
        // Two grids sharing a boundary but separated by a wall must not be
        // clustered into the same region, otherwise TSP routing can jump across.
        double pl = -1.0;
        if (!isGridPairReachableInternal(active[i], active[j], pl)) continue;
        unite(i, j);
      }

  unordered_map<int, vector<int>> comps;
  for (int i = 0; i < N; ++i) comps[find(i)].push_back(i);

  int rid = 0;
  for (auto& kv : comps) {
    RegionInfo r;
    r.id_ = rid;
    r.center_.setZero();
    r.total_unknown_ = 0;
    double tw = 0;
    for (int idx : kv.second) {
      int gid = active[idx];
      auto& g = all_grids_[gid];
      g.region_id_ = rid;
      r.grid_ids_.push_back(gid);
      double w = std::max(g.unknown_num_, 1);
      r.center_ += g.center_ * w;
      tw += w;
      r.total_unknown_ += g.unknown_num_;
    }
    if (tw > 0) r.center_ /= tw;
    regions_.push_back(r);
    rid++;
  }
  ROS_WARN("[GridPlanner] clusterRegions: %d grids -> %d regions", N, rid);
}

// ==================== TSP cost matrix ====================
void GridPlanner::getGridCostMatrix(const Vector3d& cur_pos, const Vector3d& cur_vel,
                                    const Vector3d& cur_yaw,
                                    const vector<int>& grid_ids,
                                    Eigen::MatrixXd& mat) {
  (void)cur_vel;
  (void)cur_yaw;

  int M = grid_ids.size();
  int dim = 2 + M;
  mat = Eigen::MatrixXd::Zero(dim, dim);

  // P0: info weights
  double max_unk = 1.0, max_ftr = 1.0;
  for (int j = 0; j < M; ++j) {
    auto& g = all_grids_[grid_ids[j]];
    if (g.unknown_num_ > max_unk) max_unk = g.unknown_num_;
    int nf = g.contained_frontier_ids_.size();
    if (nf > max_ftr) max_ftr = nf;
  }
  vector<double> iw(M, 1.0);
  vector<char> adj_to_finished(M, 0);
  auto is_finished_like = [](const GridCell& g) {
    return g.status_ == GridCell::FINISHED ||
           (g.covered_by_path_ && g.contained_frontier_ids_.empty() &&
            g.unknown_frontier_num_ == 0);
  };
  for (int j = 0; j < M; ++j) {
    for (const auto& ag : all_grids_) {
      if (!ag.active_) continue;
      if (!is_finished_like(ag)) continue;
      if (areGridsAdjacent(grid_ids[j], ag.id_)) {
        adj_to_finished[j] = 1;
        break;
      }
    }
  }
  for (int j = 0; j < M; ++j) {
    auto& g = all_grids_[grid_ids[j]];
    double nu = max_unk > 0 ? g.unknown_num_ / max_unk : 0.0;
    double nf = max_ftr > 0 ? (double)g.contained_frontier_ids_.size() / max_ftr : 0.0;
    // Unknown volume must participate in the score with a nontrivial weight,
    // otherwise frontier count alone dominates and wall-behind unknown space
    // is ignored.
    iw[j] = 1.0 / (1.0 + w_frontier_ * nf + 0.35 * w_info_ * nu);
    iw[j] = std::max(0.12, iw[j]);
  }

  // ATSP depot structure
  mat(0, 1) = -1000;  mat(1, 0) = 1000;
  for (int j = 0; j < M; ++j) {
    mat(0, 2 + j) = 1000;  mat(2 + j, 0) = 0;
  }

  // Drone -> grid
  for (int j = 0; j < M; ++j) {
    auto& g = all_grids_[grid_ids[j]];
    double d = (cur_pos - g.center_).norm();
    // [MODIFIED BY CODEX]: The grid ATSP supplies a global direction only.
    // Use Euclidean distance here; running current-to-grid A* for every node
    // duplicated the downstream current-to-frontier safety search.
    double c = 1.35 * d + consistent_cost2_;

    // Third weight: if this target grid is adjacent to explored/covered area,
    // reduce entry cost to encourage contiguous outward expansion.
    if (adj_to_finished[j]) c += consistent_cost_;

    mat(1, 2 + j) = c * iw[j];
    mat(2 + j, 1) = 1000;   // block return to drone
  }

  // Grid <-> grid
  for (int i = 0; i < M; ++i) {
    for (int j = i + 1; j < M; ++j) {
      auto& g1 = all_grids_[grid_ids[i]];
      auto& g2 = all_grids_[grid_ids[j]];
      double d = (g1.center_ - g2.center_).norm();
      // [MODIFIED BY CODEX]: Pairwise grid edges are abstract ordering costs.
      // Region clustering has already established wall-aware connectivity for
      // relevant adjacent cells; do not launch up to O(M^2) fresh A* searches
      // while constructing every global ATSP matrix.
      double c = 1.35 * d + consistent_cost2_;
      const bool adjacent = areGridsAdjacent(g1.id_, g2.id_);

      const bool same_region =
          (g1.region_id_ >= 0 && g1.region_id_ == g2.region_id_);
      const bool reachable_edge = same_region;

      // Same parent (siblings) -> adhesion
      if (reachable_edge && g1.parent_id_ >= 0 && g1.parent_id_ == g2.parent_id_)
        c += consistent_cost_;

      // Same region -> strong adhesion
      if (reachable_edge && same_region)
        c -= w_region_;

      const double block_cost = std::max(hard_block_cost_, non_adjacent_penalty_);

      // Enforce neighborhood connection on global TSP graph:
      // non-adjacent edges are allowed but heavily penalized.
      if (only_adjacent_edge_for_tsp_ && !adjacent) {
        if (hard_block_non_adjacent_edge_) {
          c = block_cost + d;
        } else {
          c += non_adjacent_penalty_;
        }
      }

      // Region clustering performs the wall-aware adjacent-grid reachability
      // check once.  Reuse its result: adjacent cells assigned to different
      // valid regions are blocked without repeating A* in this matrix.
      if (adjacent && g1.region_id_ >= 0 && g2.region_id_ >= 0 && !same_region) {
        c = std::max(c, block_cost * 0.85 + d);
      }

      double avg = 0.5 * (iw[i] + iw[j]);
      mat(2 + i, 2 + j) = c * avg;
      mat(2 + j, 2 + i) = c * avg;
    }
  }
  for (int i = 0; i < dim; ++i) mat(i, i) = 1000;
}

// ==================== Visualization ====================
void GridPlanner::publishGridMarkers() {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.w = 1.0;

  // ── Root grid borders (depth=0) — white, thin ──
  {
    mk.ns = "root_grid_borders";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.scale.x = 0.08;
    mk.id = 0;
    mk.points.clear();
    mk.colors.clear();
    for (auto& g : all_grids_) {
      if (g.depth_ != 0) continue;
      std_msgs::ColorRGBA c;
      c.r = 1.0; c.g = 1.0; c.b = 1.0; c.a = 0.45;
      for (int i = 0; i < 4; ++i) {
        geometry_msgs::Point p1, p2;
        p1.x = g.vertices_[i][0];     p1.y = g.vertices_[i][1];     p1.z = 0.5;
        p2.x = g.vertices_[(i+1)%4][0]; p2.y = g.vertices_[(i+1)%4][1]; p2.z = 0.5;
        mk.points.push_back(p1); mk.points.push_back(p2);
        mk.colors.push_back(c); mk.colors.push_back(c);
      }
    }
    grid_pub_.publish(mk);
  }

  // ── Sub-grid borders (depth>0) — yellow, thick ──
  {
    mk.ns = "sub_grid_borders";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.scale.x = 0.12;
    mk.id = 0;
    mk.points.clear();
    mk.colors.clear();
    for (auto& g : all_grids_) {
      if (!g.active_ || g.depth_ == 0) continue;
      std_msgs::ColorRGBA c;
      c.r = 1.0; c.g = 0.85; c.b = 0.15; c.a = 0.75;
      for (int i = 0; i < 4; ++i) {
        geometry_msgs::Point p1, p2;
        p1.x = g.vertices_[i][0];     p1.y = g.vertices_[i][1];     p1.z = 0.5;
        p2.x = g.vertices_[(i+1)%4][0]; p2.y = g.vertices_[(i+1)%4][1]; p2.z = 0.5;
        mk.points.push_back(p1); mk.points.push_back(p2);
        mk.colors.push_back(c); mk.colors.push_back(c);
      }
    }
    grid_pub_.publish(mk);
  }
}

void GridPlanner::publishGlobalTour(const vector<Vector3d>& tour_path,
                                    const vector<Vector3d>& tour_waypoints,
                                    const vector<int>& tour_grid_indices,
                                    const vector<int>& all_grid_ids) {
  // Selected frontier highlight — yellow sphere at the chosen viewpoint
  if (vis_selected_ftr_.norm() > 0.01) {
    visualization_msgs::Marker sel_sphere;
    sel_sphere.header.frame_id = "world";
    sel_sphere.header.stamp = ros::Time::now();
    sel_sphere.ns = "selected_frontier";
    sel_sphere.type = visualization_msgs::Marker::SPHERE;
    sel_sphere.action = visualization_msgs::Marker::ADD;
    sel_sphere.pose.orientation.w = 1.0;
    sel_sphere.pose.position.x = vis_selected_ftr_[0];
    sel_sphere.pose.position.y = vis_selected_ftr_[1];
    sel_sphere.pose.position.z = 0.8;
    sel_sphere.scale.x = 0.4; sel_sphere.scale.y = 0.4; sel_sphere.scale.z = 0.4;
    sel_sphere.id = 0;
    sel_sphere.color.r = 1.0; sel_sphere.color.g = 0.9; sel_sphere.color.b = 0.1; sel_sphere.color.a = 1.0;
    grid_tour_pub_.publish(sel_sphere);
  }
}

// ==================== CCL (Connected Component Labeling) ====================
// Adapted from FALCON's getCCLCenters2D()
// Source: /home/djg/FALCON/src/FALCON/falcon_planner/exploration_preprocessing/src/hierarchical_grid.cpp

void GridPlanner::runCCLInGrid(int grid_id) {
  GridCell& g = all_grids_[grid_id];
  g.components_.clear();
  g.is_split_ = false;
  g.drone_component_id_ = -1;

  if (g.free_num_ == 0) return;

  // 1. Compute voxel bounding box for this grid cell
  Vector3i bbox_min_idx, bbox_max_idx;
  sdf_map_->posToIndex(g.vmin_, bbox_min_idx);
  sdf_map_->posToIndex(g.vmax_, bbox_max_idx);

  int size_x = bbox_max_idx.x() - bbox_min_idx.x();
  int size_y = bbox_max_idx.y() - bbox_min_idx.y();
  int size_z = bbox_max_idx.z() - bbox_min_idx.z();
  if (size_x <= 0 || size_y <= 0) return;

  const int sz = std::max(1, size_z);
  int fixed_z = bbox_min_idx.z() + sz / 2;  // scan at mid-height (2D CCL)

  int step = std::max(1, ccl_step_);
  std::vector<int> labels(size_x * size_y, 0);  // 0=unlabeled, >0=component_id
  int component_id = 1;

  // 2. Free-voxel test lambda (FALCON-style: FREE + ESDF > clearance threshold)
  auto isFreeInflated = [&](int ix, int iy, int iz) -> bool {
    Eigen::Vector3i idx(ix, iy, iz);
    if (!sdf_map_->isInMap(idx)) return false;
    if (sdf_map_->getOccupancy(idx) != SDFMap::FREE) return false;
    return sdf_map_->getDistance(idx) > ccl_clearance_thresh_;
  };

  // 3. Single-pass BFS flood-fill (FALCON CCL core)
  for (int y = 0; y < size_y; y += step) {
    for (int x = 0; x < size_x; x += step) {
      int index = y * size_x + x;
      int wx = x + bbox_min_idx.x();
      int wy = y + bbox_min_idx.y();

      if (labels[index] != 0) continue;
      if (!isFreeInflated(wx, wy, fixed_z)) continue;

      // New component found — BFS flood-fill
      std::stack<int> stack;
      stack.push(index);
      labels[index] = component_id;

      int num_voxels = 0;
      double X_avg = 0.0, Y_avg = 0.0;

      while (!stack.empty()) {
        int curr = stack.top(); stack.pop();
        num_voxels++;

        int cy = curr / size_x;
        int cx = curr - cy * size_x;
        X_avg += cx;
        Y_avg += cy;

        // 4-neighbor expansion (step-aware)
        const int dx[4] = {step, -step, 0, 0};
        const int dy[4] = {0, 0, step, -step};
        for (int d = 0; d < 4; ++d) {
          int nx = cx + dx[d];
          int ny = cy + dy[d];
          if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y) continue;
          int nidx = ny * size_x + nx;
          if (labels[nidx] != 0) continue;
          if (!isFreeInflated(nx + bbox_min_idx.x(), ny + bbox_min_idx.y(), fixed_z))
            continue;
          labels[nidx] = component_id;
          stack.push(nidx);
        }
      }

      // Filter noise: too-small components are discarded
      if (num_voxels < ccl_min_component_voxels_) {
        component_id++;
        continue;
      }

      // Create component with centroid
      GridComponent comp;
      comp.comp_id_ = component_id - 1;  // 0-indexed
      comp.free_voxels_ = num_voxels;
      sdf_map_->indexToPos(
          Eigen::Vector3i(static_cast<int>(X_avg / num_voxels) + bbox_min_idx.x(),
                          static_cast<int>(Y_avg / num_voxels) + bbox_min_idx.y(),
                          fixed_z),
          comp.center_);
      g.components_.push_back(comp);
      component_id++;
    }
  }

  g.is_split_ = (g.components_.size() > 1);

  // 4. If CCL found components, update grid center to dominant component
  if (!g.components_.empty()) {
    // Sort by size descending
    std::sort(g.components_.begin(), g.components_.end(),
              [](const GridComponent& a, const GridComponent& b) {
                return a.free_voxels_ > b.free_voxels_;
              });
    // Use largest component center as grid's representative center
    g.center_ = g.components_[0].center_;
  }
}

void GridPlanner::updateDroneComponent(const Vector3d& drone_pos) {
  int gid = findGridAt(drone_pos);
  if (gid < 0 || gid >= static_cast<int>(all_grids_.size())) return;

  // Reset all drone component flags
  for (auto& g : all_grids_) {
    g.drone_component_id_ = -1;
    for (auto& comp : g.components_)
      comp.is_drone_here_ = false;
  }

  GridCell& g = all_grids_[gid];
  if (g.components_.empty() || !g.active_) return;

  // Find which component contains the drone position
  double best_d = std::numeric_limits<double>::max();
  int best_comp = -1;
  for (size_t i = 0; i < g.components_.size(); ++i) {
    double d = (g.components_[i].center_ - drone_pos).norm();
    if (d < best_d) { best_d = d; best_comp = static_cast<int>(i); }
  }
  if (best_comp >= 0) {
    g.drone_component_id_ = best_comp;
    g.components_[best_comp].is_drone_here_ = true;
  }
}

int GridPlanner::findComponentForPoint(int grid_id, const Vector3d& pos) const {
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size())) return -1;
  const GridCell& g = all_grids_[grid_id];
  if (g.components_.empty()) return -1;

  // Find the closest component center
  double best_d = std::numeric_limits<double>::max();
  int best_comp = -1;
  for (size_t i = 0; i < g.components_.size(); ++i) {
    double d = (g.components_[i].center_ - pos).norm();
    if (d < best_d) { best_d = d; best_comp = static_cast<int>(i); }
  }
  return best_comp;
}

int GridPlanner::getDroneComponent(int grid_id) const {
  if (grid_id < 0 || grid_id >= static_cast<int>(all_grids_.size())) return -1;
  return all_grids_[grid_id].drone_component_id_;
}

void GridPlanner::detectIsolatedComponents(const Vector3d& drone_pos) {
  for (auto& g : all_grids_) {
    for (auto& comp : g.components_) {
      // Check reachability from drone via A*
      vector<Vector3d> path;
      double cost = ViewNode::computeCost(
          drone_pos, comp.center_, 0.0, 0.0, Vector3d::Zero(), 0.0, path);
      comp.is_isolated_ = !std::isfinite(cost) || cost >= 900.0;
    }
  }
}

bool GridPlanner::componentsShareConnectivity(int gid1, int cid1, int gid2, int cid2) {
  // Same cell: only same component is connected
  if (gid1 == gid2) return cid1 == cid2;

  // Different cells: check A* reachability between component centers
  if (gid1 < 0 || gid2 < 0) return false;
  if (gid1 >= static_cast<int>(all_grids_.size()) ||
      gid2 >= static_cast<int>(all_grids_.size())) return false;
  if (cid1 < 0 || cid2 < 0) return false;
  if (cid1 >= static_cast<int>(all_grids_[gid1].components_.size()) ||
      cid2 >= static_cast<int>(all_grids_[gid2].components_.size())) return false;

  const Vector3d& c1 = all_grids_[gid1].components_[cid1].center_;
  const Vector3d& c2 = all_grids_[gid2].components_[cid2].center_;

  double path_len = -1.0;
  return isGridReachable(gid1, gid2, &path_len) && path_len > 0;
}

}  // namespace fast_planner
