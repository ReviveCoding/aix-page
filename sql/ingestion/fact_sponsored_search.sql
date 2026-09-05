CREATE TABLE fact_sponsored_search AS
SELECT Click AS click, Impression AS impression, DisplayURL AS display_url_id,
       AdID AS ad_id, AdvertiserID AS advertiser_id, Depth AS depth,
       Position AS position, QueryID AS query_id, KeywordID AS keyword_id,
       TitleID AS title_id, DescriptionID AS description_id, UserID AS user_id
FROM staged_sponsored_search;
