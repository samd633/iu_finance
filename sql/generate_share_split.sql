IF OBJECT_ID('dbo.share_split', 'U') IS NULL
CREATE TABLE [dbo].[share_split] (
    id           INT IDENTITY(1,1) PRIMARY KEY,
    ticker       VARCHAR(10)    NOT NULL,
    split_dt     DATE           NOT NULL,
    split_factor NUMERIC(20,16) NOT NULL
);

SELECT * FROM [dbo].[share_split];