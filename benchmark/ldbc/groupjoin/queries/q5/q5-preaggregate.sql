WITH tagged_message AS (
	SELECT m.m_messageid, m.m_creatorid
	FROM tag t
	JOIN message_tag mt ON t.t_tagid = mt.mt_tagid
	JOIN message m ON mt.mt_messageid = m.m_messageid
	WHERE t.t_name = {{tag}}
), reply_count AS (
	SELECT tm.m_messageid, count(DISTINCT r.m_messageid) AS reply_count
	FROM tagged_message tm
	LEFT JOIN message r ON tm.m_messageid = r.m_c_replyof
	GROUP BY tm.m_messageid
), like_count AS (
	SELECT tm.m_messageid,
	       count(DISTINCT l.l_messageid || ' ' || l.l_personid) AS like_count
	FROM tagged_message tm
	LEFT JOIN likes l ON tm.m_messageid = l.l_messageid
	GROUP BY tm.m_messageid
), detail AS (
	SELECT p.p_personid AS person_id,
	       sum(rc.reply_count) AS replyCount,
	       sum(lc.like_count) AS likeCount,
	       count(DISTINCT tm.m_messageid) AS messageCount
	FROM tagged_message tm
	JOIN person p ON tm.m_creatorid = p.p_personid
	JOIN reply_count rc ON tm.m_messageid = rc.m_messageid
	JOIN like_count lc ON tm.m_messageid = lc.m_messageid
	GROUP BY p.p_personid
)
SELECT person_id AS "person.id",
	   replyCount,
	   likeCount,
	   messageCount,
	   messageCount + 2 * replyCount + 10 * likeCount AS score
FROM detail
ORDER BY score DESC, person_id
LIMIT 100;
