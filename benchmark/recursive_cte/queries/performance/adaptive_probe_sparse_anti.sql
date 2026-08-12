WITH RECURSIVE state(key, depth) USING KEY (key) AS (
	SELECT key, CASE WHEN key = 0 THEN 0 ELSE 100 END
	FROM range(1000000) keys(key)
	UNION ALL
	SELECT candidate, next_depth
	FROM (
		SELECT 1000000::BIGINT + depth AS candidate, depth + 1 AS next_depth
		FROM state
		WHERE depth < 20
	) frontier
	ANTI JOIN recurring.state recurring_state ON recurring_state.key = candidate
)
SELECT count(*) AS keys, sum(depth)::BIGINT AS depth_sum, max(key) AS maximum_key
FROM state;
