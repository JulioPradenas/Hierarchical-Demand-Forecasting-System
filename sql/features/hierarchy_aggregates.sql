-- Hierarchical aggregations: store and category totals.
-- Input table: sales (item_id, dept_id, cat_id, store_id, state_id, date, sales)
SELECT
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    date,
    sales,
    SUM(sales) OVER (PARTITION BY store_id, date)          AS store_daily_total,
    SUM(sales) OVER (PARTITION BY cat_id, store_id, date)  AS cat_store_daily_total,
    SUM(sales) OVER (PARTITION BY state_id, date)          AS state_daily_total
FROM sales
ORDER BY item_id, store_id, date
