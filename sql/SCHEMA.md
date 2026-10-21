# Database schema reference

Database: `data/processed/superstore.duckdb` (rebuild with `python -m src.build_database`).
Run a query file with `python -m src.run_sql sql/<file>.sql`.

## Quick guide

| Table / view | Grain (one row = …) | Rows | Use it for |
|---|---|---|---|
| `orders_clean` | order line | 9,993 | most queries: every column in one wide table, no joins needed |
| `fact_order_lines` | order line | 9,993 | star-schema queries, joined to the `dim_*` tables |
| `orders` (view) | order | 5,009 | order-level questions: RFM, basket size |
| `dim_customer` | customer | 793 | customer attributes |
| `dim_product` | product_id | 1,862 | product attributes |
| `dim_location` | location | 632 | geography |
| `dim_date` | calendar day | 1,464 | calendar attributes, days with no orders |
| `product_name_variants` | product_id + name | 64 | the 32 ids with two names |
| `data_quality_log` | cleaning check | 12 | what the cleaning changed |
| `dropped_rows` | dropped row | 1 | row_id removed as a duplicate |
| `customer_features` | customer | 793 | step 3: RFM, profit, discounts, high-volume groups |
| `customer_mix` | customer × dimension value | 17,050 | step 3: each customer's mix of discount bands, products, regions, states |

**Joins:** `fact_order_lines` → `dim_customer` on `customer_id`, `dim_product` on
`product_id`, `dim_location` on `location_id`, `dim_date` on `order_date = date`
(or `ship_date = date`).

## Things to know before writing queries

- **Dates are `TIMESTAMP`** (time is always 00:00). Use `date_trunc('month', order_date)`,
  `year(order_date)` or `CAST(order_date AS DATE)` as needed.
- **Line vs order:** `orders_clean` has one row per product in an order. Use
  `COUNT(DISTINCT order_id)` to count orders, or query the `orders` view.
- **Margins:** aggregate first, then divide: `SUM(profit) / SUM(sales)`, not
  `AVG(profit_margin)` (the average of ratios over-weights small lines).
- **Assumption:** `gross_sales`, `discount_amount` and `unit_price` assume `sales` is
  recorded *after* discount (tested in `notebooks/01_data_quality.ipynb`, not stated
  in the data).
- **Product names:** 32 `product_id`s carry two different names (some are different
  items). `orders_clean.product_name` is the original name per line;
  `dim_product.product_name` is the most frequent one. Group by `product_id`, not
  by name.
- **Outlier flags** (`is_outlier_*`) are for analysis only; never use them as ML features.
- **Reserved words:** `data_quality_log` columns `check` and `column` must be quoted:
  `SELECT "check", "column" FROM data_quality_log`.

## Value domains

| Column | Values |
|---|---|
| `ship_mode` | First Class, Same Day, Second Class, Standard Class |
| `segment` | Consumer, Corporate, Home Office |
| `region` | Central, East, South, West |
| `category` | Furniture, Office Supplies, Technology (17 sub-categories) |
| `discount` | 0, 0.10, 0.15, 0.20, 0.30, 0.32, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80 |
| `discount_band` | `0%`, `1-20%`, `21-40%`, `>40%` (text; sorts alphabetically, not numerically) |
| Order dates | 2014-01-03 to 2017-12-30 (ship dates to 2018-01-05) |

---

## `orders_clean`
Wide, cleaned order-line table. One row per product in an order.

| Column | Type | Description |
|---|---|---|
| `row_id` | BIGINT | original row number from the raw file; unique |
| `order_id` | VARCHAR | order identifier, e.g. `CA-2016-152156` |
| `order_date` | TIMESTAMP | date the order was placed |
| `ship_date` | TIMESTAMP | date the order was shipped |
| `ship_mode` | VARCHAR | shipping class |
| `customer_id` | VARCHAR | customer identifier |
| `customer_name` | VARCHAR | customer full name |
| `segment` | VARCHAR | customer segment |
| `city` | VARCHAR | delivery city |
| `state` | VARCHAR | delivery state (49 states) |
| `postal_code` | VARCHAR | 5-digit postal code, kept as text |
| `region` | VARCHAR | US region |
| `product_id` | VARCHAR | product identifier, e.g. `FUR-BO-10001798` |
| `category` | VARCHAR | product category |
| `sub_category` | VARCHAR | product sub-category |
| `product_name` | VARCHAR | product name as written on this line |
| `sales` | DOUBLE | revenue of the line in USD, after discount |
| `quantity` | BIGINT | units sold (1–14) |
| `discount` | DOUBLE | discount rate as a fraction (0–0.8) |
| `profit` | DOUBLE | profit of the line in USD; negative = loss |
| `order_year` | INTEGER | year of `order_date` |
| `order_quarter` | INTEGER | quarter of `order_date` (1–4) |
| `order_month` | INTEGER | month of `order_date` (1–12) |
| `order_ym` | VARCHAR | year-month of `order_date`, e.g. `2016-11` |
| `ship_days` | BIGINT | days from order to shipment (0–7) |
| `profit_margin` | DOUBLE | `profit / sales` for this line |
| `is_loss` | BOOLEAN | true if `profit < 0` |
| `discount_band` | VARCHAR | discount grouped into four bands |
| `gross_sales` | DOUBLE | estimated pre-discount revenue: `sales / (1 - discount)` (assumption) |
| `discount_amount` | DOUBLE | estimated value given away: `gross_sales - sales` (assumption) |
| `unit_price` | DOUBLE | estimated pre-discount price per unit: `gross_sales / quantity` (assumption) |
| `is_outlier_sales` | BOOLEAN | `sales` outside 1.5 × IQR; analysis only |
| `is_outlier_profit` | BOOLEAN | `profit` outside 1.5 × IQR; analysis only |
| `location_id` | VARCHAR | `postal_code\|city\|state`; key to `dim_location` |

## `fact_order_lines`
Star-schema fact table: keys and measures only. One row per order line.

| Column | Type | Description |
|---|---|---|
| `row_id` | BIGINT | original row number; unique |
| `order_id` | VARCHAR | order identifier |
| `order_date` | TIMESTAMP | order date; joins to `dim_date.date` |
| `ship_date` | TIMESTAMP | ship date; joins to `dim_date.date` |
| `ship_mode` | VARCHAR | shipping class |
| `customer_id` | VARCHAR | key to `dim_customer` |
| `product_id` | VARCHAR | key to `dim_product` |
| `location_id` | VARCHAR | key to `dim_location` |
| `sales` | DOUBLE | revenue in USD, after discount |
| `quantity` | BIGINT | units sold |
| `discount` | DOUBLE | discount rate (0–0.8) |
| `profit` | DOUBLE | profit in USD; negative = loss |
| `profit_margin` | DOUBLE | `profit / sales` for this line |
| `is_loss` | BOOLEAN | true if `profit < 0` |
| `discount_band` | VARCHAR | discount grouped into four bands |
| `gross_sales` | DOUBLE | estimated pre-discount revenue (assumption) |
| `discount_amount` | DOUBLE | estimated value given away (assumption) |
| `unit_price` | DOUBLE | estimated pre-discount price per unit (assumption) |
| `ship_days` | BIGINT | days from order to shipment |

## `orders` (view)
One row per order, aggregated from `fact_order_lines`.

| Column | Type | Description |
|---|---|---|
| `order_id` | VARCHAR | order identifier; unique |
| `order_date` | TIMESTAMP | date the order was placed |
| `ship_date` | TIMESTAMP | date the order was shipped |
| `ship_mode` | VARCHAR | shipping class |
| `customer_id` | VARCHAR | key to `dim_customer` |
| `location_id` | VARCHAR | key to `dim_location` |
| `line_count` | BIGINT | number of order lines (products) |
| `quantity` | HUGEINT | total units in the order |
| `sales` | DOUBLE | total order revenue in USD |
| `profit` | DOUBLE | total order profit in USD |
| `discount_amount` | DOUBLE | total estimated discount value (assumption) |
| `is_loss` | BOOLEAN | true if the whole order lost money |

## `dim_customer`
| Column | Type | Description |
|---|---|---|
| `customer_id` | VARCHAR | customer identifier; primary key |
| `customer_name` | VARCHAR | customer full name |
| `segment` | VARCHAR | Consumer, Corporate or Home Office |

## `dim_product`
| Column | Type | Description |
|---|---|---|
| `product_id` | VARCHAR | product identifier; primary key |
| `product_name` | VARCHAR | most frequent name for this id (ties: alphabetical) |
| `category` | VARCHAR | product category |
| `sub_category` | VARCHAR | product sub-category |
| `name_variant_count` | BIGINT | number of names seen for this id (1 or 2) |

## `dim_location`
| Column | Type | Description |
|---|---|---|
| `location_id` | VARCHAR | `postal_code\|city\|state`; primary key |
| `postal_code` | VARCHAR | 5-digit postal code (not unique on its own) |
| `city` | VARCHAR | city |
| `state` | VARCHAR | state |
| `region` | VARCHAR | US region |

## `dim_date`
Every calendar day from 2014-01-03 to 2018-01-05, including days with no orders.

| Column | Type | Description |
|---|---|---|
| `date` | TIMESTAMP | calendar day; primary key |
| `year` | INTEGER | year |
| `quarter` | INTEGER | quarter (1–4) |
| `month` | INTEGER | month (1–12) |
| `month_name` | VARCHAR | month name, e.g. `January` |
| `year_month` | VARCHAR | e.g. `2016-11` |
| `day_of_week` | INTEGER | 1 = Monday … 7 = Sunday |
| `day_name` | VARCHAR | weekday name, e.g. `Monday` |
| `is_weekend` | BOOLEAN | true on Saturday and Sunday |

## `product_name_variants`
The 32 `product_id`s that appear with two names (64 rows).

| Column | Type | Description |
|---|---|---|
| `product_id` | VARCHAR | product identifier |
| `product_name` | VARCHAR | one of the names used for this id |
| `line_count` | BIGINT | order lines using this name |
| `is_chosen` | BOOLEAN | true for the name used in `dim_product` |
| `is_tie` | BOOLEAN | true if this name tied for most frequent |

## `data_quality_log`
| Column | Type | Description |
|---|---|---|
| `check` | VARCHAR | name of the check (quote it: `"check"`) |
| `column` | VARCHAR | column(s) checked (quote it: `"column"`) |
| `rows_affected` | BIGINT | rows found or changed |
| `action` | VARCHAR | what the pipeline did |

## `dropped_rows`
| Column | Type | Description |
|---|---|---|
| `row_id` | BIGINT | raw row removed as a near-duplicate (3407) |

---

## `customer_features`
One row per customer, built from `orders_clean` (step 3). **Every column summarizes
the customer's full 2014–2017 history: do not use them as features in the step 4
order-line model (leakage).** Reference date = last order date in the data + 1 day
(2017-12-31).

| Column | Type | Description |
|---|---|---|
| `customer_id` | VARCHAR | customer identifier; primary key |
| `customer_name` | VARCHAR | customer full name |
| `segment` | VARCHAR | Consumer, Corporate or Home Office |
| `first_order_date` | TIMESTAMP | first order date |
| `last_order_date` | TIMESTAMP | last order date |
| `recency_days` | BIGINT | days from last order to the reference date (R) |
| `tenure_days` | BIGINT | days from first order to the reference date |
| `frequency` | BIGINT | number of distinct orders (F) |
| `order_lines` | BIGINT | number of order lines |
| `quantity` | HUGEINT | total units |
| `sales` | DOUBLE | total revenue after discount (M) |
| `gross_sales` | DOUBLE | total revenue before discount (assumption); defines high-volume |
| `profit` | DOUBLE | total profit; negative = net loss |
| `profit_margin` | DOUBLE | `profit / sales` |
| `avg_discount` | DOUBLE | mean discount over the customer's lines |
| `high_discount_line_share` | DOUBLE | share of lines with discount > 20% |
| `high_discount_gross_share` | DOUBLE | share of gross sales with discount > 20% |
| `loss_lines` | BIGINT | number of loss-making lines |
| `worst_line_profit` | DOUBLE | profit of the customer's worst line |
| `worst_line_share_of_losses` | DOUBLE | worst line's share of all line-level losses (1 = one line explains all losses); NULL if no loss lines |
| `is_high_volume` | BOOLEAN | `gross_sales` at or above the 75th percentile of customers |
| `is_net_loss` | BOOLEAN | `profit < 0` |
| `is_near_zero_profit` | BOOLEAN | \|profit\| < $100 (sensitivity check) |
| `volume_group` | VARCHAR | `HV-loss`, `HV-profit` or `Other` |
| `hv_gross_p80` | BOOLEAN | sensitivity: gross sales at or above the 80th percentile |
| `hv_net_sales_p75` | BOOLEAN | sensitivity: net sales at or above the 75th percentile |
| `hv_quantity_p75` | BOOLEAN | sensitivity: quantity at or above the 75th percentile (ties included) |

## `customer_mix`
One row per customer and value of five dimensions. Use it to compare *what* and
*where* customers buy. Filter on `dimension` first.

| Column | Type | Description |
|---|---|---|
| `customer_id` | VARCHAR | key to `customer_features` |
| `dimension` | VARCHAR | `discount_band`, `category`, `sub_category`, `region` or `state` |
| `value` | VARCHAR | the value within that dimension, e.g. `Tables` or `Texas` |
| `order_lines` | BIGINT | the customer's lines with this value |
| `gross_sales` | DOUBLE | gross sales of those lines (assumption) |
| `sales` | DOUBLE | net sales of those lines |
| `profit` | DOUBLE | profit of those lines |
| `gross_share` | DOUBLE | share of the customer's gross sales; sums to 1 per customer and dimension |

Example: average share of gross sales shipped to Texas, by volume group
(customers with no Texas lines count as 0):

```sql
SELECT f.volume_group,
       AVG(COALESCE(m.gross_share, 0)) AS avg_texas_share
FROM customer_features AS f
LEFT JOIN customer_mix AS m
       ON m.customer_id = f.customer_id AND m.dimension = 'state' AND m.value = 'Texas'
GROUP BY f.volume_group;
```
