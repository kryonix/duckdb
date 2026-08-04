INSERT INTO organisation
SELECT id, type, name, url, LocationPlaceId
FROM read_parquet('{{DATA}}/static/Organisation/*.parquet');

INSERT INTO place
SELECT id, name, url, type, PartOfPlaceId
FROM read_parquet('{{DATA}}/static/Place/*.parquet');

INSERT INTO tagclass
SELECT id, name, url, SubclassOfTagClassId
FROM read_parquet('{{DATA}}/static/TagClass/*.parquet');

INSERT INTO tag
SELECT id, name, url, TypeTagClassId
FROM read_parquet('{{DATA}}/static/Tag/*.parquet');

INSERT INTO forum
SELECT epoch_ms(creationDate), id, title, ModeratorPersonId
FROM read_parquet('{{DATA}}/dynamic/Forum/*.parquet');

INSERT INTO forum_person
SELECT epoch_ms(creationDate), ForumId, PersonId
FROM read_parquet('{{DATA}}/dynamic/Forum_hasMember_Person/*.parquet');

INSERT INTO forum_tag
SELECT epoch_ms(creationDate), ForumId, TagId
FROM read_parquet('{{DATA}}/dynamic/Forum_hasTag_Tag/*.parquet');

INSERT INTO person
SELECT epoch_ms(creationDate), id, firstName, lastName, gender, epoch_ms(birthday)::DATE,
       locationIP, browserUsed, LocationCityId
FROM read_parquet('{{DATA}}/dynamic/Person/*.parquet');

INSERT INTO person_email
SELECT epoch_ms(creationDate), id, unnest(string_split(email, ';'))
FROM read_parquet('{{DATA}}/dynamic/Person/*.parquet');

INSERT INTO person_language
SELECT epoch_ms(creationDate), id, unnest(string_split(language, ';'))
FROM read_parquet('{{DATA}}/dynamic/Person/*.parquet');

INSERT INTO person_tag
SELECT epoch_ms(creationDate), personId, interestId
FROM read_parquet('{{DATA}}/dynamic/Person_hasInterest_Tag/*.parquet');

INSERT INTO person_university
SELECT epoch_ms(creationDate), PersonId, UniversityId, classYear
FROM read_parquet('{{DATA}}/dynamic/Person_studyAt_University/*.parquet');

INSERT INTO person_company
SELECT epoch_ms(creationDate), PersonId, CompanyId, workFrom
FROM read_parquet('{{DATA}}/dynamic/Person_workAt_Company/*.parquet');

INSERT INTO knows
SELECT epoch_ms(creationDate), Person1Id, Person2Id
FROM read_parquet('{{DATA}}/dynamic/Person_knows_Person/*.parquet');

INSERT INTO knows
SELECT epoch_ms(creationDate), Person2Id, Person1Id
FROM read_parquet('{{DATA}}/dynamic/Person_knows_Person/*.parquet');

INSERT INTO likes
SELECT epoch_ms(creationDate), PersonId, PostId
FROM read_parquet('{{DATA}}/dynamic/Person_likes_Post/*.parquet');

INSERT INTO likes
SELECT epoch_ms(creationDate), PersonId, CommentId
FROM read_parquet('{{DATA}}/dynamic/Person_likes_Comment/*.parquet');

INSERT INTO message_tag
SELECT epoch_ms(creationDate), PostId, TagId
FROM read_parquet('{{DATA}}/dynamic/Post_hasTag_Tag/*.parquet');

INSERT INTO message_tag
SELECT epoch_ms(creationDate), CommentId, TagId
FROM read_parquet('{{DATA}}/dynamic/Comment_hasTag_Tag/*.parquet');

INSERT INTO post
SELECT epoch_ms(creationDate), id, imageFile, locationIP, browserUsed, language, content,
       length, CreatorPersonId, ContainerForumId, LocationCountryId
FROM read_parquet('{{DATA}}/dynamic/Post/*.parquet');

INSERT INTO comment
SELECT epoch_ms(creationDate), id, locationIP, browserUsed, content, length,
       CreatorPersonId, LocationCountryId, ParentPostId, ParentCommentId
FROM read_parquet('{{DATA}}/dynamic/Comment/*.parquet');
