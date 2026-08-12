WITH RECURSIVE state(key, value) USING KEY (key) AS (
	SELECT key, 0 FROM range(1000000) keys(key)
	UNION ALL
	SELECT candidate.key, coalesce(recurring_state.value, 0) + 1
	FROM range(1000000) candidate(key)
	LEFT JOIN recurring.state recurring_state USING (key)
	WHERE coalesce(recurring_state.value, 0) < 8
)
SELECT count(*) AS keys, sum(value)::BIGINT AS value_sum
FROM state;
