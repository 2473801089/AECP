#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

namespace fast_planner {
// Static data
double ViewNode::vm_;
double ViewNode::am_;
double ViewNode::yd_;
double ViewNode::ydd_;
double ViewNode::w_dir_;
shared_ptr<Astar> ViewNode::astar_;
shared_ptr<RayCaster> ViewNode::caster_;
shared_ptr<SDFMap> ViewNode::map_;

// Graph node for viewpoints planning
ViewNode::ViewNode(const Vector3d& p, const double& y) {
  pos_ = p;
  yaw_ = y;
  parent_ = nullptr;
  vel_.setZero();  // vel is zero by default, should be set explicitly
}

double ViewNode::costTo(const ViewNode::Ptr& node) {
  if (node->terminal_) return 0.0;
  // [MODIFIED BY CODEX]: The layered viewpoint graph refines information and
  // yaw choices; it is not a second collision-free path planner.  Candidate
  // executability and the final B-spline are checked separately, so using A*
  // on every graph edge only duplicates work.
  const Vector3d delta = node->pos_ - pos_;
  const double distance = delta.norm();
  double pos_cost = distance / std::max(vm_, 1e-3);
  if (vel_.norm() > 1e-3 && distance > 1e-6) {
    const double dot = std::max(
        -1.0, std::min(1.0, vel_.normalized().dot(delta / distance)));
    pos_cost += w_dir_ * std::acos(dot);
  }
  const double yaw_diff = std::fabs(std::atan2(
      std::sin(node->yaw_ - yaw_), std::cos(node->yaw_ - yaw_)));
  const double yaw_cost = yaw_diff / std::max(yd_, 1e-3);
  return std::max(pos_cost, yaw_cost);
}

double ViewNode::searchPath(const Vector3d& p1, const Vector3d& p2, vector<Vector3d>& path) {
  // Try connect two points with straight line
  bool safe = true;
  Vector3i idx;
  caster_->input(p1, p2);
  while (caster_->nextId(idx)) {
    if (map_->getInflateOccupancy(idx) == 1 || map_->getOccupancy(idx) == SDFMap::UNKNOWN ||
        !map_->isInBox(idx)) {
      safe = false;
      break;
    }
  }
  if (safe) {
    path = { p1, p2 };
    return (p1 - p2).norm();
  }
  // Search a path using decreasing resolution
  vector<double> res = { 0.4 };
  for (int k = 0; k < res.size(); ++k) {
    astar_->reset();
    astar_->setResolution(res[k]);
    if (astar_->search(p1, p2) == Astar::REACH_END) {
      path = astar_->getPath();
      return astar_->pathLength(path);
    }
  }
  // Use Astar early termination cost as an estimate
  path = { p1, p2 };
  return 1000;
}

double ViewNode::computeCost(const Vector3d& p1, const Vector3d& p2, const double& y1, const double& y2,
                             const Vector3d& v1, const double& yd1, vector<Vector3d>& path) {
  // Cost of position change
  double pos_cost = ViewNode::searchPath(p1, p2, path) / vm_;

  // Consider velocity change
  Vector3d dir = p2 - p1;
  const double dir_norm = dir.norm();
  if (dir_norm > 1e-6) {
    dir /= dir_norm;
  }
  if (v1.norm() > 1e-3 && dir_norm > 1e-6) {
    Vector3d vdir = v1.normalized();
    const double dot = std::max(-1.0, std::min(1.0, vdir.dot(dir)));
    const double diff = acos(dot);
    pos_cost += w_dir_ * diff;
    // double vc = v1.dot(dir);
    // pos_cost += w_dir_ * pow(vm_ - fabs(vc), 2) / (2 * vm_ * am_);
    // if (vc < 0)
    //   pos_cost += w_dir_ * 2 * fabs(vc) / am_;
  }

  // Cost of yaw change
  double diff = fabs(y2 - y1);
  diff = min(diff, 2 * M_PI - diff);
  double yaw_cost = diff / yd_;
  return max(pos_cost, yaw_cost);

  // // Consider yaw rate change
  // if (fabs(yd1) > 1e-3)
  // {
  //   double diff1 = y2 - y1;
  //   while (diff1 < -M_PI)
  //     diff1 += 2 * M_PI;
  //   while (diff1 > M_PI)
  //     diff1 -= 2 * M_PI;
  //   double diff2 = diff1 > 0 ? diff1 - 2 * M_PI : 2 * M_PI + diff1;
  // }
  // else
  // {
  // }
}
}
