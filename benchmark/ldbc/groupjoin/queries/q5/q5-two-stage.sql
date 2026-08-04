WITH tagged_message AS (
	SELECT m.m_messageid, m.m_creatorid
	FROM tag t
	JOIN message_tag mt ON t.t_tagid = mt.mt_tagid
	JOIN message m ON mt.mt_messageid = m.m_messageid
	WHERE t.t_name = {{tag}}
), message_detail AS (
	SELECT tm.m_messageid,
	       tm.m_creatorid,
	       count(DISTINCT r.m_messageid) AS reply_count,
	       count(DISTINCT l.l_messageid || ' ' || l.l_personid) AS like_count
	FROM tagged_message tm
	LEFT JOIN message r ON tm.m_messageid = r.m_c_replyof
	LEFT JOIN likes l ON tm.m_messageid = l.l_messageid
	GROUP BY tm.m_messageid, tm.m_creatorid
), detail AS (
	SELECT p.p_personid AS person_id,
	       sum(md.reply_count) AS replyCount,
	       sum(md.like_count) AS likeCount,
	       count(*) AS messageCount
	FROM message_detail md
	JOIN person p ON md.m_creatorid = p.p_personid
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
