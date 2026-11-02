#ifndef _GRID_PLANNER_H_
#define _GRID_PLANNER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <visualization_msgs/Marker.h>

using Eigen::Vector3d;
using Eigen::Vector3i;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::unordered_map;
using std::unordered_set;

namespace fast_planner {

class EDTEnvironment;
class SDFMap;
class Astar;

// Forward declaration for CCL
struct GridComponent;

struct GridCell {
  enum Status {
    INACTIVE = 0,
    PURE_OBSTACLE = 1,
    FINISHED = 2,
    EFFECTIVE_UNKNOWN = 3
  };

  int id_ = -1;              // unique sequential ID = index into all_grids_
  int parent_id_ = -1;       // parent coarse grid ID, -1 for root
  int depth_ = 0;            // RACER hgrid: 0=coarse, each deeper level halves X/Y
  bool active_ = false;      // currently alive (not subdivided)
  bool is_relevant_ = false; // has unknown/frontiers after FOV scan
  bool need_subdivide_ = false; // set by updateGridCell when free > min_free_
  int unknown_num_ = 0;
  // [MODIFIED BY CODEX]: Global grid TSP lifecycle is deliberately
  // independent from frontier/CCL completion.  A grid remains a global task
  // until its unknown voxels drop below a fraction of the first valid scan.
  int initial_unknown_num_ = 0;
  bool unknown_baseline_valid_ = false;
  bool tsp_retired_ = false;
  int free_num_ = 0;
  int occupied_num_ = 0;
  int unknown_frontier_num_ = 0;
  int covered_free_num_ = 0;
  double coverage_ratio_ = 0.0;
  bool covered_by_path_ = false;
  Status status_ = INACTIVE;
  Vector3d center_;
  Vector3d vmin_, vmax_;
  vector<Vector3d> vertices_;  // 4 corners XY
  vector<Vector3d> normals_;   // 4 edge normals
  unordered_map<int, int> contained_frontier_ids_;
  int region_id_ = -1;

  // CCL: per-cell connectivity decomposition (FALCON-style)
  vector<GridComponent> components_;    // connected FREE components
  int drone_component_id_ = -1;         // which component the drone is in
  bool is_split_ = false;               // cell has >1 disconnected FREE component
};

struct RegionInfo {
  int id_;
  vector<int> grid_ids_;
  Vector3d center_;
  int total_unknown_ = 0;
};

// CCL (Connected Component Labeling) — lightweight per-cell connectivity decomposition.
// Detects when a grid cell is split by obstacles into disconnected sub-regions.
// Adapted from FALCON (https://github.com/HKUST-Aerial-Robotics/FALCON).
struct GridComponent {
  int comp_id_ = -1;           // component index within the cell (0, 1, 2...)
  Vector3d center_;            // centroid of FREE voxels in this component
  int free_voxels_ = 0;        // number of FREE voxels
  vector<int> frontier_ids_;   // frontier IDs belonging to this component
  bool is_drone_here_ = false; // drone's current position lies in this component
  bool is_isolated_ = false;   // component not reachable from drone position
};

// RUSH-style subregion state for dynamic spatial decomposition.
// Subregions replace the fixed 2×2 root-grid concept: a single root
// (depth 0) covers the full map and recursively splits along the
// longest side.  Depth-1 children are the "subregions" that the
// non-closed TSP orders for exploration.
enum class SubRegionState { ACTIVE = 0, PENDING = 1, COMPLETED = 2 };

struct SubRegionInfo {
  int grid_id_ = -1;
  SubRegionState state_ = SubRegionState::PENDING;
  double coverage_ratio_ = 0.0;
  int frontier_count_ = 0;
  int depth_ = 0;
};

class GridPlanner {
public:
  GridPlanner(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
  ~GridPlanner();

  void updateGrids();
  void updateGrids(const Vector3d& cur_pos);
  void updatePathCoverage(const Vector3d& cur_pos);
  void getActiveGrids(vector<int>& grid_ids);
  void getActiveGridCenters(vector<Vector3d>& centers, vector<int>& grid_ids);
  // [MODIFIED BY CODEX]: Returns global coverage tasks.  Unlike
  // getActiveGrids(), this does not require a current frontier or a current
  // A* route to the unknown centroid.
  void getGlobalTspGridCenters(vector<Vector3d>& centers, vector<int>& grid_ids);
  void inputFrontiers(const vector<Vector3d>& frontier_avgs);
  void inputFrontiersDetailed(const vector<Vector3d>& frontier_avgs,
                              const vector<Vector3d>& frontier_views);
  void getGridCostMatrix(const Vector3d& cur_pos, const Vector3d& cur_vel,
                         const Vector3d& cur_yaw, const vector<int>& grid_ids,
                         Eigen::MatrixXd& mat);
  void getFrontiersInGrids(const vector<int>& grid_ids, vector<int>& ftr_ids);
  void getNeighborGridIds(int grid_id, int hop, vector<int>& grid_ids,
                          bool include_self = true);
  bool isGridReachable(int id1, int id2, double* path_len = nullptr);
  int getGridContainingPoint(const Vector3d& pos) { return findGridAt(pos); }
  int getRootGridId(int grid_id) const;
  // RUSH: get the subregion (depth-1) containing a grid or point.
  // Falls back to depth-0 (root) when no depth-1 exists yet.
  int getSubRegionId(int grid_id) const;
  int getSubRegionForPoint(const Vector3d& pos);
  bool isSubRegionComplete(int sr_id) const;
  double getSubRegionCoverage(int sr_id) const;
  int getSubRegionFrontierCount(int sr_id) const;
  // RUSH: subregion TSP traversal order.
  void computeSubRegionTSP(const Vector3d& cur_pos);
  const vector<int>& getSubRegionTour() const { return subregion_tour_; }
  int getNextSubRegion(int current_sr_id) const;
  int getActiveSubRegion() const { return active_subregion_id_; }
  void setActiveSubRegion(int sr_id) { active_subregion_id_ = sr_id; }
  void updateSubRegionStates(const Vector3d& drone_pos);
  bool isRushMode() const { return use_rush_subregions_; }
  bool isGridComplete(int grid_id) const;
  bool isRootGridComplete(int root_id) const;
  // Local component id for a point inside a grid.
  // allow_unknown=false: strict execution connectivity, only FREE cells.
  // allow_unknown=true : exploration connectivity, UNKNOWN cells can bridge
  //                      free-space components as long as they are not
  //                      inflated occupied.
  int getLocalComponentId(int grid_id, const Vector3d& pos,
                          bool allow_unknown = false) const;

  // --- CCL (Connected Component Labeling) — FALCON-style connectivity ---
  // Run CCL flood-fill on a single grid cell to detect obstacle-split components.
  void runCCLInGrid(int grid_id);
  // [ABLATION]: 三维连通分量（slice_dim=3d），不下采样高度，26/6 邻域沿 z 扩展。
  void runCCLInGrid3D(int grid_id);
  // Update which component the drone is in for all active grids.
  void updateDroneComponent(const Vector3d& drone_pos);
  // Find which component a 3D point belongs to in a given grid.
  int findComponentForPoint(int grid_id, const Vector3d& pos) const;
  // Get the component the drone is currently in.
  int getDroneComponent(int grid_id) const;
  // Detect isolated components (not reachable from drone position).
  void detectIsolatedComponents(const Vector3d& drone_pos);
  // Check if two grids/components are in the same connected region.
  bool componentsShareConnectivity(int gid1, int cid1, int gid2, int cid2);

  // [ABLATION]: 论文算法 1 第 17 行的“同网格不同连通分量前沿不参与本轮竞争”。
  // 参考分量取“无人机所在侧”的分量：无人机在该网格内时用 drone_component_id_，
  // 否则取分量中心距无人机最近者（近似无人机进入该网格后所处的一侧）。
  // 返回 true 表示该前沿被允许参与本轮局部排序。
  bool componentAllowsFrontier(int grid_id, const Vector3d& ftr_pos,
                               const Vector3d& drone_pos) const;
  bool cclFilterEnabled() const { return use_ccl_for_filter_; }
  bool cclCenterEnabled() const { return use_ccl_for_center_; }

  bool findGridAnchorById(int grid_id, Vector3d& anchor);
  void clusterRegions();
  void publishGridMarkers();
  void publishGlobalTour(const vector<Vector3d>& tour_path,
                         const vector<Vector3d>& tour_waypoints,
                         const vector<int>& tour_grid_indices = {},
                         const vector<int>& all_grid_ids = {});
  void setActiveGridIdForVis(int id) { vis_active_grid_id_ = id; }
  void setDroneGridIdForVis(int id) { cur_vis_gid_ = id; }
  void setSelectedFrontierForVis(const Vector3d& pos) { vis_selected_ftr_ = pos; }

  GridCell& getGrid(int id) { return all_grids_[id]; }
  vector<GridCell> all_grids_;    // all_grids_[id] - sequential IDs
  vector<RegionInfo> regions_;
  double grid_size_ = 20.0;      // coarsest level size
  int max_depth_ = 2;            // 0=coarse, 1=fine, 2=finer
  double min_grid_size_ = 10.0;   // do not split below this XY side length
  Vector3d map_min_, map_max_;
  double voxel_res_ = 0.1;
  double getSampleRes() const { return std::max(voxel_res_, grid_state_sample_res_); }

private:
  void initRootGrids();
  void subdivideGrid(int grid_id);
  int findGridAt(const Vector3d& pos);  // finest active grid containing pos
  void updateGridCell(int grid_id);
  bool isRelevant(const GridCell& g);
  bool areGridsAdjacentLegacy(int id1, int id2) const;
  bool shareBoundary(const GridCell& g1, const GridCell& g2) const;
  bool areGridsAdjacent(int id1, int id2);
  bool findGridAnchor(const GridCell& g, Vector3d& anchor);
  bool isGridPairReachableInternal(int id1, int id2, double& path_len);
  long long makePairKey(int id1, int id2) const;
  bool isVoxelCovered(const Eigen::Vector3i& idx) const;

  shared_ptr<EDTEnvironment> edt_;
  shared_ptr<SDFMap> sdf_map_;
  unique_ptr<Astar> path_finder_;

  int next_id_ = 0;
  int root_x_ = 0, root_y_ = 0;  // root grid dimensions
  vector<vector<int>> root_grids_;  // root_grids_[y][x] = grid ID
  Vector3d root_res_;              // coarse grid resolution

  // RUSH-style subregion management
  vector<SubRegionInfo> subregions_;    // indexed by grid ID (may be sparse)
  vector<int> subregion_tour_;          // subregion IDs in TSP visit order
  int active_subregion_id_ = -1;
  int single_root_id_ = -1;             // depth-0 root that covers entire map
  bool use_rush_subregions_ = false;    // default to RACER-style fixed-root hgrid

  ros::Publisher grid_pub_, grid_tour_pub_, grid_text_pub_, grid_order_pub_;

  // RACER-style visualization state
  int vis_active_grid_id_ = -1;
  int cur_vis_gid_ = -1;             // drone's current grid for visualization
  Vector3d vis_selected_ftr_ = Vector3d::Zero();  // selected frontier for visualization
  // A* path cache for grid tour: pair key -> path segments
  std::unordered_map<long long, std::vector<Eigen::Vector3d>> vis_astar_path_cache_;
  std::vector<Eigen::Vector3d> last_vis_waypoints_;

  int min_unknown_ = 500;
  int min_free_ = 3000;
  double w_info_ = 0.3, w_frontier_ = 0.2, w_region_ = 10.0;
  double consistent_cost_ = -5.0, consistent_cost2_ = 8.0;
  double astar_connect_max_dist_ = 12.0;
  double unreachable_edge_penalty_ = 80.0;
  double boundary_touch_eps_ = 0.15;
  bool reachable_neighbor_only_ = false;
  bool use_legacy_connectivity_ = true;
  bool enable_dynamic_subdivision_ = true;
  bool subdivide_only_current_grid_ = true;
  int max_subdivide_per_update_ = 8;
  bool subdivide_frontier_grids_ = true;
  int max_frontier_subdivide_per_update_ = 2;
  double frontier_subdivide_max_dist_ = 18.0;
  int fixed_root_x_ = 2;
  int fixed_root_y_ = 2;
  bool enforce_root_2x2_ = true;
  bool only_adjacent_edge_for_tsp_ = true;
  double non_adjacent_penalty_ = 200.0;
  bool relevant_frontier_only_ = true;
  double frontier_fallback_max_dist_ = 6.0;
  bool hard_block_non_adjacent_edge_ = true;
  double hard_block_cost_ = 5000.0;
  double coverage_radius_ = 4.5;
  double coverage_height_ = 1.2;
  double coverage_complete_ratio_ = 0.85;
  double root_complete_coverage_ratio_ = 0.95;
  double root_complete_unknown_ratio_ = 0.05;
  double root_tail_gap_ratio_ = 0.01;
  int root_tail_gap_free_voxels_ = 120;
  // [MODIFIED BY CODEX]: retire only from the *global TSP* when 90% of the
  // initially observed unknown volume is cleared.  Residual frontiers remain
  // available to the local frontier planner.
  double tsp_retire_unknown_ratio_ = 0.10;
  double coverage_voxel_stride_ = 0.15;
  double grid_state_sample_res_ = 0.2;

  // CCL parameters (adapted from FALCON)
  int ccl_step_ = 2;                  // sub-sampling step for CCL scanning
  double ccl_clearance_thresh_ = 0.5; // ESDF threshold: FREE voxel must have >0.5m clearance
  int ccl_min_component_voxels_ = 10; // filter out noise: components smaller than this
  // [ABLATION]: 原 use_ccl_ 一个开关同时控制“分量筛选”和“网格代表位置修正”，
  // 无法做单点消融，现拆成两个开关；保留 use_ccl_ 作为总开关（false 时两者全关）。
  bool use_ccl_ = true;               // master switch (kept for backward compatibility)
  bool use_ccl_for_filter_ = true;    // 分量筛选：同网格异分量前沿不参与本轮竞争
  bool use_ccl_for_center_ = true;    // 分量修正：网格代表位置取最大分量质心
  // [ABLATION]: 切片维数。"2d"=中间高度水平切片（默认）；"3d"=包围盒内三维连通，不下采样高度
  std::string slice_dim_str_ = "2d";
  bool use_slice_3d_ = false;

  unordered_set<int> covered_voxel_addrs_;
  Vector3d last_coverage_pos_ = Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  mutable std::unordered_map<long long, int> local_component_cache_;

  // Cache pairwise reachability in one planning cycle.
  // key = (min_id << 32) | max_id, value >= 0 path_len, < 0 unreachable.
  unordered_map<long long, double> pair_reachability_cache_;
};

}  // namespace fast_planner
#endif
