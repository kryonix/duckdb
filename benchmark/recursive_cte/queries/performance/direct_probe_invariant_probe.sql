WITH RECURSIVE
fanout(key, weight) AS (
	SELECT (i % 50000)::INTEGER, count(*) OVER (PARTITION BY i % 50000)::INTEGER FROM range(400000) t(i)
),
state(key, value) USING KEY (key) AS (
	SELECT key::INTEGER, 0::BIGINT FROM range(50000) keys(key)
	UNION ALL
	SELECT recurring_state.key, recurring_state.value + sum(fanout.weight)
	FROM recurring.state recurring_state
	JOIN fanout ON fanout.key = recurring_state.key
	WHERE recurring_state.value < 20 * 64
	GROUP BY recurring_state.key, recurring_state.value
)
SELECT count(*) AS keys, sum(value)::BIGINT AS value_sum, max(value) AS maximum_value
FROM state;
