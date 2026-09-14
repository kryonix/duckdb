WITH RECURSIVE t(i, level) AS (
	SELECT i, 0 FROM range(1000000) r(i)
	UNION
	SELECT (i * 7 + 13) % 2000000, level + 1 FROM t WHERE level < 5
)
SELECT count(*) AS row_count, max(level) AS maximum_level, sum(i)::BIGINT AS value_sum
FROM t;
