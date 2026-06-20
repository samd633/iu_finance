DECLARE @TICKER VARCHAR(50) = 'AAPL';

-- 1. Confirm the column names + types in dbo.share_split
--    (so we can set SPLIT_DATE_COL / SPLIT_FACTOR_COL in db.py correctly)
SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'share_split'
ORDER BY ORDINAL_POSITION;

-- 2. Inspect the actual split rows for the ticker
--    (confirm the factor convention: a 4-for-1 split should be stored as 4.0)
SELECT *
FROM dbo.share_split
WHERE ticker = @TICKER
ORDER BY 1;
