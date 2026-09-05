CREATE VIEW dim_ad AS SELECT DISTINCT ad_id, advertiser_id, title_id, description_id FROM fact_sponsored_search;
CREATE VIEW dim_advertiser AS SELECT DISTINCT advertiser_id FROM fact_sponsored_search;
CREATE VIEW dim_display_url AS SELECT DISTINCT display_url_id FROM fact_sponsored_search;
CREATE VIEW dim_query AS SELECT DISTINCT query_id, keyword_id FROM fact_sponsored_search;
CREATE VIEW dim_user AS SELECT DISTINCT user_id, CASE WHEN user_id = 0 THEN 1 ELSE 0 END AS is_anonymous FROM fact_sponsored_search;
CREATE VIEW mart_ctr_features AS SELECT *, 1.0 * position / depth AS normalized_position FROM fact_sponsored_search;
CREATE VIEW mart_model_context AS SELECT * FROM mart_ctr_features;
