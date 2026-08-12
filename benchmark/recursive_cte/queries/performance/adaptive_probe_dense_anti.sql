WITH RECURSIVE state(key, depth) USING KEY (key) AS (
	SELECT key, 0 FROM range(20000) keys(key)
	UNION
	SELECT candidate, next_depth
	FROM (
		SELECT key + 20000 AS candidate, depth + 1 AS next_depth
		FROM state
		WHERE depth < 8
	) frontier
	CROSS JOIN range(4) duplicates
	ANTI JOIN recurring.state recurring_state ON recurring_state.key = candidate
)
SELECT count(*) AS keys, sum(depth)::BIGINT AS depth_sum, max(depth) AS maximum_depth
FROM state;
