-- Aggregate transaction-level history into SKU-level statistics.
SELECT
    sku,
    eclass,
    manufacturer,
    AVG(price) AS avg_price,
    COUNT(*) AS purchase_count,
    MAX([date]) AS last_seen_date
FROM transactions
GROUP BY
    sku,
    eclass,
    manufacturer;
