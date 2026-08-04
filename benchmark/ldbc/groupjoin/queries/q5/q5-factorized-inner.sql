WITH tagged_message AS (
	SELECT m.m_messageid, m.m_creatorid
	FROM tag t
	JOIN message_tag mt ON t.t_tagid = mt.mt_tagid
	JOIN message m ON mt.mt_messageid = m.m_messageid
	WHERE t.t_name = {{tag}}
)
SELECT tm.m_messageid,
	   tm.m_creatorid,
	   count(*) AS pair_count,
	   count(r.m_messageid) AS reply_rows,
	   count(l.l_messageid) AS like_rows
FROM tagged_message tm
JOIN message r ON tm.m_messageid = r.m_c_replyof
JOIN likes l ON tm.m_messageid = l.l_messageid
GROUP BY tm.m_messageid, tm.m_creatorid
ORDER BY tm.m_messageid;
