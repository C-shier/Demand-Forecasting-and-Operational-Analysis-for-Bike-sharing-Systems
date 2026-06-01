-- 共享单车需求预测与运营分析：核心分析 SQL
-- 建议先将原始 hour.csv 导入到名为 bike_hour 的表中。

-- 1. 项目基础概览
SELECT
    COUNT(*) AS 记录数,
    MIN(dteday) AS 开始日期,
    MAX(dteday) AS 结束日期,
    SUM(cnt) AS 总租赁量,
    ROUND(AVG(cnt), 2) AS 小时平均租赁量,
    ROUND(SUM(registered) * 1.0 / SUM(cnt), 4) AS 注册用户租赁占比,
    ROUND(SUM(casual) * 1.0 / SUM(cnt), 4) AS 非注册用户租赁占比
FROM bike_hour;

-- 2. 月度租赁量趋势
SELECT
    strftime('%Y-%m', dteday) AS 月份,
    SUM(cnt) AS 总租赁量
FROM bike_hour
GROUP BY strftime('%Y-%m', dteday)
ORDER BY 月份;

-- 3. 24 小时平均租赁量
SELECT
    hr AS 小时,
    ROUND(AVG(cnt), 2) AS 平均租赁量
FROM bike_hour
GROUP BY hr
ORDER BY hr;

-- 4. 工作日与非工作日分时差异
SELECT
    hr AS 小时,
    CASE WHEN workingday = 1 THEN '工作日' ELSE '非工作日' END AS 是否工作日,
    ROUND(AVG(cnt), 2) AS 平均租赁量
FROM bike_hour
GROUP BY hr, workingday
ORDER BY hr, 是否工作日;

-- 5. 天气维度平均租赁量
SELECT
    CASE weathersit
        WHEN 1 THEN '晴天/少云'
        WHEN 2 THEN '多云/薄雾'
        WHEN 3 THEN '小雨/小雪'
        WHEN 4 THEN '恶劣天气'
    END AS 天气,
    ROUND(AVG(cnt), 2) AS 平均租赁量
FROM bike_hour
GROUP BY weathersit
ORDER BY 平均租赁量 DESC;

-- 6. 季节维度平均租赁量
SELECT
    CASE season
        WHEN 1 THEN '春季'
        WHEN 2 THEN '夏季'
        WHEN 3 THEN '秋季'
        WHEN 4 THEN '冬季'
    END AS 季节,
    ROUND(AVG(cnt), 2) AS 平均租赁量
FROM bike_hour
GROUP BY season
ORDER BY 平均租赁量 DESC;
