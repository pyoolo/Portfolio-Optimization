# Input data format

The loaders accept a single `.xlsx` workbook with up to three sheets. The
parsers are locale-tolerant: they accept Italian-style numbers (`12,5%`,
`1.234,56`) and dates (`dd/mm/yyyy`), and coerce unparseable cells to `NaN`
rather than failing.

## Sheet `Composizioni` (required)

Long format, one row per holding per rebalance date.

| Data Di Ribilanciamento | Ticker | Peso |
|---|---|---|
| 27/01/2022 | FUND01 LX Equity | 12,5 |
| 27/01/2022 | FUND02 LX Equity | 8,0 |
| 05/05/2022 | FUND01 LX Equity | 10,0 |

- Column names are configurable via `load_compositions(col_date=…, col_ticker=…, col_weight=…)`.
- `Peso` may be a percentage (`12,5` → 12.5%, the default) or a fraction;
  control with `weights_are_percent`.

## Sheet `NAV` (required)

Wide format with **repeated** `Data | TICKER` column pairs, one pair per fund,
with potentially asynchronous dates. Because the date columns share the header
`Data`, pandas reads them as `Data, Data.1, Data.2, …`; the loader handles this
by treating any column whose header starts with `data` as a date paired with the
next (price) column.

| Data | FUND01 LX Equity | Data | FUND02 LX Equity | … |
|---|---|---|---|---|
| 27/01/2022 | 100,12 | 28/01/2022 | 99,80 | … |
| 28/01/2022 | 100,45 | 31/01/2022 | 100,10 | … |

The price column's header is taken as the ticker, so it must match the tickers
used in `Composizioni`.

## Sheet `ISIN` (optional)

Maps tickers to readable names for labeling.

| Ticker | Nome | Codice ISIN |
|---|---|---|
| FUND01 LX Equity | Global Equity Fund | LU0000000001 |

If the sheet is missing or malformed, name labeling is silently skipped.

## Generating a sample

`python -m diworsification.make_sample_data` writes
`data/sample_portfolio.xlsx` in exactly this format, so you can inspect a
concrete example and run the whole pipeline without proprietary data.
