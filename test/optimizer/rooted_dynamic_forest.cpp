#include "catch.hpp"

#include "duckdb/planner/detail/rooted_dynamic_forest.hpp"

#include <random>

using namespace duckdb;

static idx_t ForestNodeIndex(const vector<unique_ptr<RootedDynamicForestNode>> &nodes, RootedDynamicForestNode &node) {
	for (idx_t node_idx = 0; node_idx < nodes.size(); node_idx++) {
		if (nodes[node_idx].get() == &node) {
			return node_idx;
		}
	}
	return DConstants::INVALID_INDEX;
}

class NaiveRootedForest {
public:
	explicit NaiveRootedForest(idx_t count)
	    : parents(count, DConstants::INVALID_INDEX), node_values(count, 0), edge_values(count, 0) {
	}

	vector<idx_t> parents;
	vector<uint64_t> node_values;
	vector<uint64_t> edge_values;

public:
	idx_t FindRoot(idx_t node) const {
		while (parents[node] != DConstants::INVALID_INDEX) {
			node = parents[node];
		}
		return node;
	}

	bool Connected(idx_t left, idx_t right) const {
		return FindRoot(left) == FindRoot(right);
	}

	bool IsAncestor(idx_t ancestor, idx_t descendant) const {
		while (descendant != DConstants::INVALID_INDEX) {
			if (ancestor == descendant) {
				return true;
			}
			descendant = parents[descendant];
		}
		return false;
	}

	idx_t LowestCommonAncestor(idx_t left, idx_t right) const {
		if (!Connected(left, right)) {
			return DConstants::INVALID_INDEX;
		}
		unordered_set<idx_t> left_path;
		while (left != DConstants::INVALID_INDEX) {
			left_path.insert(left);
			left = parents[left];
		}
		while (left_path.find(right) == left_path.end()) {
			right = parents[right];
		}
		return right;
	}

	uint64_t RootPathValue(idx_t node) const {
		uint64_t result = 0;
		while (node != DConstants::INVALID_INDEX) {
			result |= node_values[node];
			result |= edge_values[node];
			node = parents[node];
		}
		return result;
	}

	uint64_t PathValue(idx_t ancestor, idx_t descendant) const {
		uint64_t result = 0;
		while (true) {
			result |= node_values[descendant];
			if (descendant == ancestor) {
				return result;
			}
			result |= edge_values[descendant];
			descendant = parents[descendant];
		}
	}

	idx_t FindFirstNodeOnPath(idx_t ancestor, idx_t descendant, uint64_t node_mask) const {
		if (node_mask == 0 || !IsAncestor(ancestor, descendant)) {
			return DConstants::INVALID_INDEX;
		}
		idx_t result = DConstants::INVALID_INDEX;
		while (true) {
			if ((node_values[descendant] & node_mask) != 0) {
				result = descendant;
			}
			if (descendant == ancestor) {
				return result;
			}
			descendant = parents[descendant];
		}
	}

	idx_t FindLastNodeOnPath(idx_t ancestor, idx_t descendant, uint64_t node_mask) const {
		if (node_mask == 0 || !IsAncestor(ancestor, descendant)) {
			return DConstants::INVALID_INDEX;
		}
		while (true) {
			if ((node_values[descendant] & node_mask) != 0) {
				return descendant;
			}
			if (descendant == ancestor) {
				return DConstants::INVALID_INDEX;
			}
			descendant = parents[descendant];
		}
	}
};

static vector<unique_ptr<RootedDynamicForestNode>> CreateForestNodes(idx_t count) {
	vector<unique_ptr<RootedDynamicForestNode>> result;
	for (idx_t node_idx = 0; node_idx < count; node_idx++) {
		result.push_back(make_uniq<RootedDynamicForestNode>());
	}
	return result;
}

TEST_CASE("Rooted dynamic forest handles fixed tree shapes", "[optimizer][rooted_dynamic_forest]") {
	RootedDynamicForest forest;
	auto nodes = CreateForestNodes(15);
	forest.SetNodeValue(*nodes[0], {1});
	REQUIRE(&forest.FindRoot(*nodes[0]) == nodes[0].get());
	REQUIRE(forest.GetRootPathValue(*nodes[0]).flags == 1);
	REQUIRE(forest.LowestCommonAncestor(*nodes[0], *nodes[0]).get() == nodes[0].get());

	for (idx_t child_idx = 1; child_idx <= 5; child_idx++) {
		REQUIRE(forest.Link(*nodes[child_idx], *nodes[child_idx - 1], {uint64_t(1) << (child_idx + 15)}));
		forest.SetNodeValue(*nodes[child_idx], {uint64_t(1) << child_idx});
	}
	REQUIRE(&forest.FindRoot(*nodes[5]) == nodes[0].get());
	REQUIRE(forest.IsAncestor(*nodes[2], *nodes[5]));
	REQUIRE_FALSE(forest.IsAncestor(*nodes[5], *nodes[2]));
	REQUIRE(forest.LowestCommonAncestor(*nodes[4], *nodes[5]).get() == nodes[4].get());
	REQUIRE(&forest.Access(*nodes[5]) == nodes[5].get());
	REQUIRE(&forest.Access(*nodes[5]) == nodes[5].get());
}

TEST_CASE("Rooted dynamic forest handles stars and bushy trees", "[optimizer][rooted_dynamic_forest]") {
	RootedDynamicForest forest;
	auto nodes = CreateForestNodes(15);
	for (idx_t child_idx = 1; child_idx <= 5; child_idx++) {
		REQUIRE(forest.Link(*nodes[child_idx], *nodes[0], {uint64_t(1) << child_idx}));
	}
	REQUIRE(forest.LowestCommonAncestor(*nodes[2], *nodes[5]).get() == nodes[0].get());

	REQUIRE(forest.Link(*nodes[7], *nodes[6]));
	REQUIRE(forest.Link(*nodes[8], *nodes[6]));
	REQUIRE(forest.Link(*nodes[9], *nodes[7]));
	REQUIRE(forest.Link(*nodes[10], *nodes[7]));
	REQUIRE(forest.Link(*nodes[11], *nodes[8]));
	REQUIRE(forest.Link(*nodes[12], *nodes[11]));
	REQUIRE(forest.LowestCommonAncestor(*nodes[9], *nodes[10]).get() == nodes[7].get());
	REQUIRE(forest.LowestCommonAncestor(*nodes[9], *nodes[12]).get() == nodes[6].get());
	REQUIRE_FALSE(forest.Connected(*nodes[1], *nodes[9]));
	REQUIRE_FALSE(forest.LowestCommonAncestor(*nodes[1], *nodes[9]));
}

TEST_CASE("Rooted dynamic forest preserves path values across cuts", "[optimizer][rooted_dynamic_forest]") {
	RootedDynamicForest forest;
	auto nodes = CreateForestNodes(8);
	for (idx_t node_idx = 0; node_idx < nodes.size(); node_idx++) {
		forest.SetNodeValue(*nodes[node_idx], {uint64_t(1) << node_idx});
	}
	for (idx_t child_idx = 1; child_idx < nodes.size(); child_idx++) {
		REQUIRE(forest.Link(*nodes[child_idx], *nodes[child_idx - 1], {uint64_t(1) << (child_idx + 16)}));
	}
	RootedDynamicForestPathValue path;
	auto old_parent = forest.GetRepresentedParent(*nodes[3]);
	auto old_edge = forest.GetEdgeValue(*nodes[3]);
	REQUIRE(forest.GetPathValue(*nodes[3], *nodes[7], path));
	uint64_t expected = 0;
	for (idx_t node_idx = 3; node_idx <= 7; node_idx++) {
		expected |= uint64_t(1) << node_idx;
		if (node_idx > 3) {
			expected |= uint64_t(1) << (node_idx + 16);
		}
	}
	REQUIRE(path.flags == expected);
	REQUIRE(forest.GetRepresentedParent(*nodes[3]) == old_parent);
	REQUIRE(forest.GetEdgeValue(*nodes[3]) == old_edge);
	REQUIRE_FALSE(forest.GetPathValue(*nodes[7], *nodes[3], path));

	for (idx_t cut_idx = 1; cut_idx < nodes.size(); cut_idx++) {
		REQUIRE(forest.CutFromParent(*nodes[cut_idx]));
		REQUIRE_FALSE(forest.Connected(*nodes[cut_idx - 1], *nodes[cut_idx]));
		REQUIRE(forest.GetEdgeValue(*nodes[cut_idx]).flags == 0);
		REQUIRE(forest.Link(*nodes[cut_idx], *nodes[cut_idx - 1], {uint64_t(1) << (cut_idx + 16)}));
	}
	REQUIRE_FALSE(forest.CutFromParent(*nodes[0]));
	REQUIRE_FALSE(forest.Link(*nodes[7], *nodes[0]));
}

TEST_CASE("Rooted dynamic forest finds the first matching path node", "[optimizer][rooted_dynamic_forest]") {
	RootedDynamicForest forest;
	auto nodes = CreateForestNodes(7);
	for (idx_t child_idx = 1; child_idx < 6; child_idx++) {
		REQUIRE(forest.Link(*nodes[child_idx], *nodes[child_idx - 1], {uint64_t(1) << child_idx}));
	}
	forest.SetNodeValue(*nodes[0], {1});
	forest.SetNodeValue(*nodes[2], {2});
	forest.SetNodeValue(*nodes[4], {2});
	forest.SetNodeValue(*nodes[5], {4});

	REQUIRE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[5], {1}).get() == nodes[0].get());
	REQUIRE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[5], {2}).get() == nodes[2].get());
	REQUIRE(forest.FindFirstNodeOnPath(*nodes[2], *nodes[5], {2}).get() == nodes[2].get());
	REQUIRE(forest.FindFirstNodeOnPath(*nodes[3], *nodes[5], {2}).get() == nodes[4].get());
	REQUIRE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[5], {4}).get() == nodes[5].get());
	REQUIRE_FALSE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[5], {8}));
	REQUIRE_FALSE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[5], {uint64_t(1) << 3}));
	REQUIRE_FALSE(forest.FindFirstNodeOnPath(*nodes[5], *nodes[0], {1}));
	REQUIRE_FALSE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[6], {1}));
	REQUIRE_FALSE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[5], {}));
	REQUIRE(forest.GetRepresentedParent(*nodes[0]) == nullptr);
	REQUIRE(forest.GetRepresentedParent(*nodes[2]).get() == nodes[1].get());
}

TEST_CASE("Rooted dynamic forest finds the last matching path node", "[optimizer][rooted_dynamic_forest]") {
	RootedDynamicForest forest;
	auto nodes = CreateForestNodes(7);
	for (idx_t child_idx = 1; child_idx < 6; child_idx++) {
		REQUIRE(forest.Link(*nodes[child_idx], *nodes[child_idx - 1], {uint64_t(1) << child_idx}));
	}
	forest.SetNodeValue(*nodes[0], {1});
	forest.SetNodeValue(*nodes[2], {2});
	forest.SetNodeValue(*nodes[4], {2});
	forest.SetNodeValue(*nodes[5], {4});

	REQUIRE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {1}).get() == nodes[0].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {2}).get() == nodes[4].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[2], *nodes[5], {2}).get() == nodes[4].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[3], *nodes[5], {2}).get() == nodes[4].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {4}).get() == nodes[5].get());
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {8}));
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {uint64_t(1) << 3}));
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[5], *nodes[0], {1}));
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[0], *nodes[6], {1}));
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {}));
	optional_ptr<RootedDynamicForestNode> result;
	REQUIRE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {8}, result));
	REQUIRE_FALSE(result);
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[5], *nodes[0], {1}, result));
	REQUIRE_FALSE(result);
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[0], *nodes[6], {1}, result));
	REQUIRE_FALSE(result);
	optional_ptr<RootedDynamicForestNode> path_child;
	REQUIRE(forest.FindLastNodeOnPath(*nodes[0], *nodes[5], {2}, result, path_child));
	REQUIRE(result.get() == nodes[4].get());
	REQUIRE(path_child.get() == nodes[1].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[2], *nodes[5], {8}, result, path_child));
	REQUIRE_FALSE(result);
	REQUIRE(path_child.get() == nodes[3].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[5], *nodes[5], {4}, result, path_child));
	REQUIRE(result.get() == nodes[5].get());
	REQUIRE_FALSE(path_child);
	REQUIRE_FALSE(forest.FindLastNodeOnPath(*nodes[5], *nodes[0], {1}, result, path_child));
	REQUIRE_FALSE(result);
	REQUIRE_FALSE(path_child);
	REQUIRE(forest.GetRepresentedParent(*nodes[0]) == nullptr);
	REQUIRE(forest.GetRepresentedParent(*nodes[2]).get() == nodes[1].get());
}

TEST_CASE("Rooted dynamic forest preserves state after rejected mutations", "[optimizer][rooted_dynamic_forest]") {
	RootedDynamicForest forest;
	auto nodes = CreateForestNodes(4);
	for (idx_t node_idx = 0; node_idx < nodes.size(); node_idx++) {
		forest.SetNodeValue(*nodes[node_idx], {uint64_t(1) << node_idx});
	}
	REQUIRE(forest.Link(*nodes[1], *nodes[0], {uint64_t(1) << 16}));
	REQUIRE(forest.Link(*nodes[2], *nodes[1], {uint64_t(1) << 17}));

	auto RequireOriginalChain = [&]() {
		REQUIRE(forest.GetRepresentedParent(*nodes[0]) == nullptr);
		REQUIRE(forest.GetRepresentedParent(*nodes[1]).get() == nodes[0].get());
		REQUIRE(forest.GetRepresentedParent(*nodes[2]).get() == nodes[1].get());
		REQUIRE(&forest.FindRoot(*nodes[2]) == nodes[0].get());
		REQUIRE(
		    forest.GetRootPathValue(*nodes[2]).flags ==
		    ((uint64_t(1) << 0) | (uint64_t(1) << 1) | (uint64_t(1) << 2) | (uint64_t(1) << 16) | (uint64_t(1) << 17)));
	};

	REQUIRE_FALSE(forest.Link(*nodes[2], *nodes[3]));
	RequireOriginalChain();
	REQUIRE_FALSE(forest.Link(*nodes[0], *nodes[2]));
	RequireOriginalChain();
	REQUIRE_FALSE(forest.SetEdgeValue(*nodes[0], {uint64_t(1) << 18}));
	REQUIRE_FALSE(forest.CutFromParent(*nodes[0]));
	RequireOriginalChain();

	REQUIRE(forest.CutFromParent(*nodes[2]));
	REQUIRE(forest.Link(*nodes[2], *nodes[3], {uint64_t(1) << 18}));
	REQUIRE(&forest.FindRoot(*nodes[2]) == nodes[3].get());
	REQUIRE(forest.GetRootPathValue(*nodes[2]).flags ==
	        ((uint64_t(1) << 2) | (uint64_t(1) << 3) | (uint64_t(1) << 18)));
}

TEST_CASE("Rooted dynamic forest handles a deep chain iteratively", "[optimizer][rooted_dynamic_forest]") {
	constexpr idx_t NODE_COUNT = 10000;
	RootedDynamicForest forest;
	auto nodes = CreateForestNodes(NODE_COUNT);
	for (idx_t node_idx = 1; node_idx < NODE_COUNT; node_idx++) {
		REQUIRE(forest.Link(*nodes[node_idx], *nodes[node_idx - 1]));
	}
	forest.SetNodeValue(*nodes[0], {1});
	forest.SetNodeValue(*nodes[NODE_COUNT - 1], {2});
	REQUIRE(&forest.FindRoot(*nodes[NODE_COUNT - 1]) == nodes[0].get());
	REQUIRE(forest.LowestCommonAncestor(*nodes[NODE_COUNT / 2], *nodes[NODE_COUNT - 1]).get() ==
	        nodes[NODE_COUNT / 2].get());
	REQUIRE(forest.GetRootPathValue(*nodes[NODE_COUNT - 1]).flags == 3);
	REQUIRE(forest.FindFirstNodeOnPath(*nodes[0], *nodes[NODE_COUNT - 1], {2}).get() == nodes[NODE_COUNT - 1].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[0], *nodes[NODE_COUNT - 1], {1}).get() == nodes[0].get());
	REQUIRE(forest.FindLastNodeOnPath(*nodes[0], *nodes[NODE_COUNT - 1], {2}).get() == nodes[NODE_COUNT - 1].get());
}

TEST_CASE("Rooted dynamic forest matches a naive randomized oracle", "[optimizer][rooted_dynamic_forest]") {
	const vector<uint64_t> RANDOM_SEEDS {UINT64_C(0x9f2d7b41c36a850e), UINT64_C(0x0123456789abcdef),
	                                     UINT64_C(0xd1b54a32d192ed03), UINT64_C(0x94d049bb133111eb)};
	constexpr idx_t NODE_COUNT = 32;
	constexpr idx_t OPERATION_COUNT = 5000;
	for (auto random_seed : RANDOM_SEEDS) {
		RootedDynamicForest forest;
		auto nodes = CreateForestNodes(NODE_COUNT);
		NaiveRootedForest oracle(NODE_COUNT);
		std::mt19937_64 random(random_seed);

		auto ComparePair = [&](idx_t left, idx_t right) {
			INFO("seed=" << random_seed << " left=" << left << " right=" << right);
			REQUIRE(forest.Connected(*nodes[left], *nodes[right]) == oracle.Connected(left, right));
			auto actual_lca = forest.LowestCommonAncestor(*nodes[left], *nodes[right]);
			auto expected_lca = oracle.LowestCommonAncestor(left, right);
			REQUIRE((actual_lca ? ForestNodeIndex(nodes, *actual_lca) : DConstants::INVALID_INDEX) == expected_lca);
			REQUIRE(forest.IsAncestor(*nodes[left], *nodes[right]) == oracle.IsAncestor(left, right));
			const uint64_t node_mask = uint64_t(1) << ((left + right) % 63);
			auto actual_first = forest.FindFirstNodeOnPath(*nodes[left], *nodes[right], {node_mask});
			auto expected_first = oracle.FindFirstNodeOnPath(left, right, node_mask);
			REQUIRE((actual_first ? ForestNodeIndex(nodes, *actual_first) : DConstants::INVALID_INDEX) ==
			        expected_first);
			auto actual_last = forest.FindLastNodeOnPath(*nodes[left], *nodes[right], {node_mask});
			auto expected_last = oracle.FindLastNodeOnPath(left, right, node_mask);
			REQUIRE((actual_last ? ForestNodeIndex(nodes, *actual_last) : DConstants::INVALID_INDEX) == expected_last);
		};

		for (idx_t operation_idx = 0; operation_idx < OPERATION_COUNT; operation_idx++) {
			INFO("seed=" << random_seed << " operation=" << operation_idx);
			const idx_t node = random() % NODE_COUNT;
			const idx_t other = random() % NODE_COUNT;
			switch (random() % 6) {
			case 0:
				if (node != other && oracle.parents[node] == DConstants::INVALID_INDEX &&
				    !oracle.Connected(node, other)) {
					const uint64_t edge = uint64_t(1) << (random() % 63);
					REQUIRE(forest.Link(*nodes[node], *nodes[other], {edge}));
					oracle.parents[node] = other;
					oracle.edge_values[node] = edge;
				}
				break;
			case 1: {
				const bool expected = oracle.parents[node] != DConstants::INVALID_INDEX;
				REQUIRE(forest.CutFromParent(*nodes[node]) == expected);
				if (expected) {
					oracle.parents[node] = DConstants::INVALID_INDEX;
					oracle.edge_values[node] = 0;
				}
				break;
			}
			case 2: {
				const uint64_t value = uint64_t(1) << (random() % 63);
				forest.SetNodeValue(*nodes[node], {value});
				oracle.node_values[node] = value;
				break;
			}
			case 3: {
				const uint64_t value = uint64_t(1) << (random() % 63);
				const bool expected = oracle.parents[node] != DConstants::INVALID_INDEX;
				REQUIRE(forest.SetEdgeValue(*nodes[node], {value}) == expected);
				if (expected) {
					oracle.edge_values[node] = value;
				}
				break;
			}
			case 4:
				REQUIRE(ForestNodeIndex(nodes, forest.FindRoot(*nodes[node])) == oracle.FindRoot(node));
				REQUIRE(forest.GetRootPathValue(*nodes[node]).flags == oracle.RootPathValue(node));
				break;
			case 5: {
				RootedDynamicForestPathValue path;
				const bool expected = oracle.IsAncestor(node, other);
				REQUIRE(forest.GetPathValue(*nodes[node], *nodes[other], path) == expected);
				if (expected) {
					REQUIRE(path.flags == oracle.PathValue(node, other));
				}
				break;
			}
			}
			ComparePair(node, other);
			if (operation_idx % 100 == 0) {
				for (idx_t left = 0; left < NODE_COUNT; left++) {
					for (idx_t right = 0; right < NODE_COUNT; right++) {
						ComparePair(left, right);
					}
				}
			}
		}
	}
}
