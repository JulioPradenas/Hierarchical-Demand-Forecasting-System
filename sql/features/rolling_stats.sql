-- Rolling stats computed over lag_28 (not raw sales) to avoid leakage.
-- Input table: sales (item_id, store_id, date, sales)
WITH lagged AS (
    SELECT
        item_id,
        store_id,
        date,
        sales,
        LAG(sales, 28) OVER (PARTITION BY item_id, store_id ORDER BY date) AS lag_28
    FROM sales
)
SELECT
    item_id,
    store_id,
    date,
    sales,
    lag_28,
    AVG(lag_28) OVER (
        PARTITION BY item_id, store_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_7,
    AVG(lag_28) OVER (
        PARTITION BY item_id, store_id
        ORDER BY date
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_28,
    STDDEV_POP(lag_28) OVER (
        PARTITION BY item_id, store_id
        ORDER BY date
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ) AS rolling_std_28
FROM lagged
ORDER BY item_id, store_id, date
