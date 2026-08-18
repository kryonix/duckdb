#include "duckdb/planner/detail/rooted_dynamic_forest.hpp"

#include "duckdb/common/assert.hpp"

namespace duckdb {

class RootedDynamicForestRestoreGuard {
public:
	RootedDynamicForestRestoreGuard(RootedDynamicForest &forest_p, RootedDynamicForestNode &child_p,
	                                optional_ptr<RootedDynamicForestNode> parent_p,
	                                const RootedDynamicForestPathValue &edge_value_p)
	    : forest(forest_p), child(child_p), parent(parent_p), edge_value(edge_value_p) {
	}

	~RootedDynamicForestRestoreGuard() noexcept {
		if (!parent) {
			return;
		}
		const bool linked = forest.Link(child, *parent, edge_value);
		D_ASSERT(linked);
	}

private:
	RootedDynamicForest &forest;
	RootedDynamicForestNode &child;
	optional_ptr<RootedDynamicForestNode> parent;
	RootedDynamicForestPathValue edge_value;
};

void RootedDynamicForestPathValue::Merge(const RootedDynamicForestPathValue &other) {
	flags |= other.flags;
	if (other.minimum_cardinality.IsValid() &&
	    (!minimum_cardinality.IsValid() || other.minimum_cardinality.GetIndex() < minimum_cardinality.GetIndex())) {
		minimum_cardinality = other.minimum_cardinality;
	}
}

bool RootedDynamicForestPathValue::operator==(const RootedDynamicForestPathValue &other) const {
	return flags == other.flags && minimum_cardinality == other.minimum_cardinality;
}

bool RootedDynamicForestPathValue::operator!=(const RootedDynamicForestPathValue &other) const {
	return !(*this == other);
}

RootedDynamicForestNode::RootedDynamicForestNode() {
}

bool RootedDynamicForest::IsAuxiliaryRoot(const RootedDynamicForestNode &node) const {
	if (!node.auxiliary_parent) {
		return true;
	}
	auto &parent = *node.auxiliary_parent;
	return parent.auxiliary_left.get() != &node && parent.auxiliary_right.get() != &node;
}

void RootedDynamicForest::Update(RootedDynamicForestNode &node) {
	RootedDynamicForestPathValue result;
	RootedDynamicForestPathValue node_result;
	RootedDynamicForestPathValue edge_result;
	if (node.auxiliary_left) {
		result.Merge(node.auxiliary_left->auxiliary_value);
		node_result.Merge(node.auxiliary_left->auxiliary_node_value);
		edge_result.Merge(node.auxiliary_left->auxiliary_edge_value);
	}
	result.Merge(node.node_value);
	node_result.Merge(node.node_value);
	result.Merge(node.edge_to_parent);
	edge_result.Merge(node.edge_to_parent);
	if (node.auxiliary_right) {
		result.Merge(node.auxiliary_right->auxiliary_value);
		node_result.Merge(node.auxiliary_right->auxiliary_node_value);
		edge_result.Merge(node.auxiliary_right->auxiliary_edge_value);
	}
	node.auxiliary_value = result;
	node.auxiliary_node_value = node_result;
	node.auxiliary_edge_value = edge_result;
#ifdef DEBUG
	VerifyLocal(node);
#endif
}

void RootedDynamicForest::Rotate(RootedDynamicForestNode &node) {
	auto parent = node.auxiliary_parent;
	D_ASSERT(parent && !IsAuxiliaryRoot(node));
	auto grandparent = parent->auxiliary_parent;
	const bool parent_is_auxiliary_root = IsAuxiliaryRoot(*parent);

	if (parent->auxiliary_left.get() == &node) {
		parent->auxiliary_left = node.auxiliary_right;
		if (node.auxiliary_right) {
			node.auxiliary_right->auxiliary_parent = parent;
		}
		node.auxiliary_right = parent;
	} else {
		D_ASSERT(parent->auxiliary_right.get() == &node);
		parent->auxiliary_right = node.auxiliary_left;
		if (node.auxiliary_left) {
			node.auxiliary_left->auxiliary_parent = parent;
		}
		node.auxiliary_left = parent;
	}

	parent->auxiliary_parent = node;
	node.auxiliary_parent = grandparent;
	if (!parent_is_auxiliary_root) {
		D_ASSERT(grandparent);
		if (grandparent->auxiliary_left.get() == parent.get()) {
			grandparent->auxiliary_left = node;
		} else {
			D_ASSERT(grandparent->auxiliary_right.get() == parent.get());
			grandparent->auxiliary_right = node;
		}
	}
	Update(*parent);
	Update(node);
}

void RootedDynamicForest::Splay(RootedDynamicForestNode &node) {
	while (!IsAuxiliaryRoot(node)) {
		auto parent = node.auxiliary_parent;
		if (!IsAuxiliaryRoot(*parent)) {
			auto grandparent = parent->auxiliary_parent;
			const bool node_is_left = parent->auxiliary_left.get() == &node;
			const bool parent_is_left = grandparent->auxiliary_left.get() == parent.get();
			if (node_is_left == parent_is_left) {
				Rotate(*parent);
			} else {
				Rotate(node);
			}
		}
		Rotate(node);
	}
}

optional_ptr<RootedDynamicForestNode> RootedDynamicForest::Expose(RootedDynamicForestNode &node) {
	optional_ptr<RootedDynamicForestNode> previous;
	auto current = optional_ptr<RootedDynamicForestNode>(node);
	optional_ptr<RootedDynamicForestNode> last;
	while (current) {
		Splay(*current);
		current->auxiliary_right = previous;
		if (previous) {
			previous->auxiliary_parent = current;
		}
		Update(*current);
		last = current;
		previous = current;
		current = current->auxiliary_parent;
	}
	Splay(node);
	return last;
}

RootedDynamicForestNode &RootedDynamicForest::Access(RootedDynamicForestNode &node) {
	Expose(node);
	return node;
}

RootedDynamicForestNode &RootedDynamicForest::FindRoot(RootedDynamicForestNode &node) {
	Expose(node);
	auto root = optional_ptr<RootedDynamicForestNode>(node);
	while (root->auxiliary_left) {
		root = root->auxiliary_left;
	}
	Splay(*root);
	return *root;
}

bool RootedDynamicForest::Connected(RootedDynamicForestNode &left, RootedDynamicForestNode &right) {
	if (&left == &right) {
		return true;
	}
	return &FindRoot(left) == &FindRoot(right);
}

optional_ptr<RootedDynamicForestNode> RootedDynamicForest::LowestCommonAncestor(RootedDynamicForestNode &left,
                                                                                RootedDynamicForestNode &right) {
	if (!Connected(left, right)) {
		return nullptr;
	}
	// The second expose stops at the first node it encounters on the already exposed root-to-left path.
	Expose(left);
	return Expose(right);
}

bool RootedDynamicForest::IsAncestor(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant) {
	Expose(ancestor);
	return Expose(descendant).get() == &ancestor;
}

bool RootedDynamicForest::Link(RootedDynamicForestNode &child, RootedDynamicForestNode &parent,
                               const RootedDynamicForestPathValue &edge_value) {
	if (child.represented_parent || Connected(child, parent)) {
		return false;
	}
	Expose(child);
	D_ASSERT(!child.auxiliary_left);
	child.auxiliary_parent = parent;
	child.represented_parent = parent;
	child.edge_to_parent = edge_value;
	Update(child);
	return true;
}

bool RootedDynamicForest::CutFromParent(RootedDynamicForestNode &child) {
	if (!child.represented_parent) {
		return false;
	}
	Expose(child);
	D_ASSERT(child.auxiliary_left);
	child.auxiliary_left->auxiliary_parent = nullptr;
	child.auxiliary_left = nullptr;
	child.represented_parent = nullptr;
	child.edge_to_parent = {};
	Update(child);
	return true;
}

void RootedDynamicForest::SetNodeValue(RootedDynamicForestNode &node, const RootedDynamicForestPathValue &value) {
	Expose(node);
	node.node_value = value;
	Update(node);
}

bool RootedDynamicForest::SetEdgeValue(RootedDynamicForestNode &node, const RootedDynamicForestPathValue &value) {
	if (!node.represented_parent) {
		return false;
	}
	Expose(node);
	node.edge_to_parent = value;
	Update(node);
	return true;
}

RootedDynamicForestPathValue RootedDynamicForest::GetRootPathValue(RootedDynamicForestNode &node) {
	Expose(node);
	return node.auxiliary_value;
}

bool RootedDynamicForest::GetPathValue(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
                                       RootedDynamicForestPathValue &result) {
	if (!IsAncestor(ancestor, descendant)) {
		return false;
	}
	auto old_parent = ancestor.represented_parent;
	auto old_edge = ancestor.edge_to_parent;
	if (!old_parent) {
		result = GetRootPathValue(descendant);
		return true;
	}
	const bool cut = CutFromParent(ancestor);
	D_ASSERT(cut);
	RootedDynamicForestRestoreGuard restore(*this, ancestor, old_parent, old_edge);
	result = GetRootPathValue(descendant);
	return true;
}

optional_ptr<RootedDynamicForestNode>
RootedDynamicForest::FindFirstNodeOnPath(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
                                         const RootedDynamicForestPathValue &node_mask) {
	if (node_mask.flags == 0 || !IsAncestor(ancestor, descendant)) {
		return nullptr;
	}
	auto old_parent = ancestor.represented_parent;
	auto old_edge = ancestor.edge_to_parent;
	if (old_parent) {
		const bool cut = CutFromParent(ancestor);
		D_ASSERT(cut);
	}
	RootedDynamicForestRestoreGuard restore(*this, ancestor, old_parent, old_edge);
	Expose(descendant);
	if ((descendant.auxiliary_node_value.flags & node_mask.flags) == 0) {
		return nullptr;
	}
	auto current = optional_ptr<RootedDynamicForestNode>(descendant);
	while (current) {
		if (current->auxiliary_left && (current->auxiliary_left->auxiliary_node_value.flags & node_mask.flags) != 0) {
			current = current->auxiliary_left;
			continue;
		}
		if ((current->node_value.flags & node_mask.flags) != 0) {
			Splay(*current);
			return current;
		}
		D_ASSERT(current->auxiliary_right &&
		         (current->auxiliary_right->auxiliary_node_value.flags & node_mask.flags) != 0);
		current = current->auxiliary_right;
	}
	return nullptr;
}

optional_ptr<RootedDynamicForestNode>
RootedDynamicForest::FindLastNodeOnPath(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
                                        const RootedDynamicForestPathValue &node_mask) {
	optional_ptr<RootedDynamicForestNode> result;
	if (!FindLastNodeOnPath(ancestor, descendant, node_mask, result)) {
		return nullptr;
	}
	return result;
}

bool RootedDynamicForest::FindLastNodeOnPath(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
                                             const RootedDynamicForestPathValue &node_mask,
                                             optional_ptr<RootedDynamicForestNode> &result) {
	optional_ptr<RootedDynamicForestNode> path_child;
	return FindLastNodeOnPath(ancestor, descendant, node_mask, result, path_child);
}

bool RootedDynamicForest::FindLastNodeOnPath(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
                                             const RootedDynamicForestPathValue &node_mask,
                                             optional_ptr<RootedDynamicForestNode> &result,
                                             optional_ptr<RootedDynamicForestNode> &path_child) {
	result = nullptr;
	path_child = nullptr;
	Expose(ancestor);
	if (Expose(descendant).get() != &ancestor) {
		return false;
	}
	Splay(ancestor);
	path_child = ancestor.auxiliary_right;
	while (path_child && path_child->auxiliary_left) {
		path_child = path_child->auxiliary_left;
	}
	if (node_mask.flags == 0) {
		return true;
	}
	if ((!ancestor.auxiliary_right || (ancestor.auxiliary_right->auxiliary_node_value.flags & node_mask.flags) == 0)) {
		if ((ancestor.node_value.flags & node_mask.flags) != 0) {
			result = ancestor;
		}
		return true;
	}
	auto current = ancestor.auxiliary_right;
	while (current) {
		if (current->auxiliary_right && (current->auxiliary_right->auxiliary_node_value.flags & node_mask.flags) != 0) {
			current = current->auxiliary_right;
			continue;
		}
		if ((current->node_value.flags & node_mask.flags) != 0) {
			Splay(*current);
			result = current;
			return true;
		}
		D_ASSERT(current->auxiliary_left &&
		         (current->auxiliary_left->auxiliary_node_value.flags & node_mask.flags) != 0);
		current = current->auxiliary_left;
	}
	return true;
}

optional_ptr<RootedDynamicForestNode>
RootedDynamicForest::FindLastEdgeOnPath(RootedDynamicForestNode &ancestor, RootedDynamicForestNode &descendant,
                                        const RootedDynamicForestPathValue &edge_mask) {
	if (edge_mask.flags == 0 || !IsAncestor(ancestor, descendant)) {
		return nullptr;
	}
	auto old_parent = ancestor.represented_parent;
	auto old_edge = ancestor.edge_to_parent;
	if (old_parent) {
		const bool cut = CutFromParent(ancestor);
		D_ASSERT(cut);
	}
	RootedDynamicForestRestoreGuard restore(*this, ancestor, old_parent, old_edge);
	Expose(descendant);
	if ((descendant.auxiliary_edge_value.flags & edge_mask.flags) == 0) {
		return nullptr;
	}
	auto current = optional_ptr<RootedDynamicForestNode>(descendant);
	while (current) {
		if (current->auxiliary_right && (current->auxiliary_right->auxiliary_edge_value.flags & edge_mask.flags) != 0) {
			current = current->auxiliary_right;
			continue;
		}
		if ((current->edge_to_parent.flags & edge_mask.flags) != 0) {
			Splay(*current);
			return current;
		}
		D_ASSERT(current->auxiliary_left &&
		         (current->auxiliary_left->auxiliary_edge_value.flags & edge_mask.flags) != 0);
		current = current->auxiliary_left;
	}
	return nullptr;
}

optional_ptr<RootedDynamicForestNode> RootedDynamicForest::GetRepresentedParent(RootedDynamicForestNode &node) const {
	return node.represented_parent;
}

RootedDynamicForestPathValue RootedDynamicForest::GetEdgeValue(RootedDynamicForestNode &node) const {
	return node.edge_to_parent;
}

#ifdef DEBUG
void RootedDynamicForest::VerifyLocal(const RootedDynamicForestNode &node) const {
	if (node.auxiliary_left) {
		D_ASSERT(node.auxiliary_left->auxiliary_parent.get() == &node);
	}
	if (node.auxiliary_right) {
		D_ASSERT(node.auxiliary_right->auxiliary_parent.get() == &node);
	}
	RootedDynamicForestPathValue expected;
	RootedDynamicForestPathValue expected_nodes;
	RootedDynamicForestPathValue expected_edges;
	if (node.auxiliary_left) {
		expected.Merge(node.auxiliary_left->auxiliary_value);
		expected_nodes.Merge(node.auxiliary_left->auxiliary_node_value);
		expected_edges.Merge(node.auxiliary_left->auxiliary_edge_value);
	}
	expected.Merge(node.node_value);
	expected_nodes.Merge(node.node_value);
	expected.Merge(node.edge_to_parent);
	expected_edges.Merge(node.edge_to_parent);
	if (node.auxiliary_right) {
		expected.Merge(node.auxiliary_right->auxiliary_value);
		expected_nodes.Merge(node.auxiliary_right->auxiliary_node_value);
		expected_edges.Merge(node.auxiliary_right->auxiliary_edge_value);
	}
	D_ASSERT(expected == node.auxiliary_value);
	D_ASSERT(expected_nodes == node.auxiliary_node_value);
	D_ASSERT(expected_edges == node.auxiliary_edge_value);
}
#endif

} // namespace duckdb
