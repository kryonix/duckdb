//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/detail/rooted_dynamic_forest.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

struct RootedDynamicForestPathValue {
	uint64_t flags = 0;

	void Merge(const RootedDynamicForestPathValue &other);
	bool operator==(const RootedDynamicForestPathValue &other) const;
	bool operator!=(const RootedDynamicForestPathValue &other) const;
};

class RootedDynamicForest;

//! A stable, non-owning sidecar node used by RootedDynamicForest.
class RootedDynamicForestNode {
public:
	RootedDynamicForestNode();

	RootedDynamicForestNode(const RootedDynamicForestNode &) = delete;
	RootedDynamicForestNode &operator=(const RootedDynamicForestNode &) = delete;

private:
	friend class RootedDynamicForest;

	//! An auxiliary parent when it links back through a child slot, and otherwise a represented path-parent.
	optional_ptr<RootedDynamicForestNode> auxiliary_parent;
	optional_ptr<RootedDynamicForestNode> auxiliary_left;
	optional_ptr<RootedDynamicForestNode> auxiliary_right;
	optional_ptr<RootedDynamicForestNode> represented_parent;
	RootedDynamicForestPathValue node_value;
	RootedDynamicForestPathValue edge_to_parent;
	RootedDynamicForestPathValue auxiliary_value;
	RootedDynamicForestPathValue auxiliary_node_value;
};

//! A fixed-root link/cut forest. Represented edges point from a child toward its parent.
class RootedDynamicForest {
public:
	bool Link(RootedDynamicForestNode &child, RootedDynamicForestNode &parent,
	          const RootedDynamicForestPathValue &edge_value = {});
	bool CutFromParent(RootedDynamicForestNode &child);

	//! Exposes the represented root-to-node path and returns node.
	RootedDynamicForestNode &Access(RootedDynamicForestNode &node);
	RootedDynamicForestNode &FindRoot(RootedDynamicForestNode &node);
	bool Connected(RootedDynamicForestNode &left, RootedDynamicForestNode &right);
	bool IsAncestor(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant);
	optional_ptr<RootedDynamicForestNode> LowestCommonAncestor(RootedDynamicForestNode &left,
	                                                           RootedDynamicForestNode &right);

	void SetNodeValue(RootedDynamicForestNode &node, const RootedDynamicForestPathValue &value);
	bool SetEdgeValue(RootedDynamicForestNode &node, const RootedDynamicForestPathValue &value);
	RootedDynamicForestPathValue GetRootPathValue(RootedDynamicForestNode &node);
	bool GetPathValue(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
	                  RootedDynamicForestPathValue &result);
	//! Returns the first matching node on the inclusive path, starting at ancestor.
	optional_ptr<RootedDynamicForestNode> FindFirstNodeOnPath(RootedDynamicForestNode &ancestor,
	                                                          RootedDynamicForestNode &descendant,
	                                                          const RootedDynamicForestPathValue &node_mask);
	//! Returns the last matching node on the inclusive path, ending at descendant.
	optional_ptr<RootedDynamicForestNode> FindLastNodeOnPath(RootedDynamicForestNode &ancestor,
	                                                         RootedDynamicForestNode &descendant,
	                                                         const RootedDynamicForestPathValue &node_mask);
	//! Sets result to the last matching node and returns whether the path exists.
	bool FindLastNodeOnPath(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
	                        const RootedDynamicForestPathValue &node_mask,
	                        optional_ptr<RootedDynamicForestNode> &result);
	//! Also sets path_child to the first node after ancestor, when one exists.
	bool FindLastNodeOnPath(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
	                        const RootedDynamicForestPathValue &node_mask,
	                        optional_ptr<RootedDynamicForestNode> &result,
	                        optional_ptr<RootedDynamicForestNode> &path_child);

	optional_ptr<RootedDynamicForestNode> GetRepresentedParent(RootedDynamicForestNode &node) const;
	RootedDynamicForestPathValue GetEdgeValue(RootedDynamicForestNode &node) const;

private:
	bool IsAuxiliaryRoot(const RootedDynamicForestNode &node) const;
	void Update(RootedDynamicForestNode &node);
	void Rotate(RootedDynamicForestNode &node);
	void Splay(RootedDynamicForestNode &node);
	optional_ptr<RootedDynamicForestNode> Expose(RootedDynamicForestNode &node);

#ifdef DEBUG
	void VerifyLocal(const RootedDynamicForestNode &node) const;
#endif
};

} // namespace duckdb
