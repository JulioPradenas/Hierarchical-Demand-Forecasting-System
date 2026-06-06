-- Lag features for horizon H=28: minimum lag is 28 to prevent leakage.
-- Input table: sales (item_id, store_id, date, sales)
SELECT
    item_id,
    store_id,
    date,
    sales,
    LAG(sales, 28)  OVER w AS lag_28,
    LAG(sales, 35)  OVER w AS lag_35,
    LAG(sales, 42)  OVER w AS lag_42,
    LAG(sales, 56)  OVER w AS lag_56,
    LAG(sales, 91)  OVER w AS lag_91,
    LAG(sales, 182) OVER w AS lag_182,
    LAG(sales, 364) OVER w AS lag_364,
    LAG(sales, 365) OVER w AS lag_365
FROM sales
WINDOW w AS (PARTITION BY item_id, store_id ORDER BY date)
ORDER BY item_id, store_id, date
