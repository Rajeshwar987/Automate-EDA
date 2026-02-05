# GenAI Usage Log for EDA Script

## Tool Used

ChatGPT (OpenAI API via Python SDK)

------------------------------------------------------------------------

## Generic Prompt Used in Code

**Prompt Template:**

"Analyze this dataset: {rows} rows x {columns} columns. First columns:
{column_list}. Provide a sharp 2-sentence business insight."

This prompt is dynamically populated at runtime using: - Number of rows
in the dataset - Number of columns in the dataset - First few column
names

The goal is to produce a concise, business-oriented understanding of any
dataset loaded into the EDA system, regardless of domain.

------------------------------------------------------------------------

## Example Instantiated Prompt

"Analyze this dataset: 100000 rows x 20 columns. First columns:
\['col1', 'col2', 'col3', 'col4', 'col5'\]. Provide a sharp 2-sentence
business insight."

------------------------------------------------------------------------

## Key Expected Response Pattern

A short, high-level interpretation of: - What the dataset likely
represents - Any structural or trend-based observation inferred from
schema and size

No domain assumptions. No deep statistics. Just contextual business
framing.

------------------------------------------------------------------------

## What Was Verified Before Using GenAI

1.  Updated deprecated API calls to the current `responses.create`
    method.

2.  Switched to a supported model (`gpt-4.1-mini`).

3.  Confirmed environment variable `OPENAI_API_KEY` is accessible within
    Python.

4.  Resolved terminal environment caching issues on Windows.

5.  Enabled API billing to avoid quota errors.

6.  Cleaned dataset column names using:

    ``` python
    df.columns = df.columns.str.strip()
    ```

7.  Ensured plots do not block execution by saving instead of showing.

8.  Verified correct script execution path with:

    ``` python
    print(os.path.abspath(__file__))
    ```

------------------------------------------------------------------------

## Outcome

The EDA script can now load any CSV file, perform automated analysis,
generate visualizations, and produce a GenAI-powered business summary
reliably for any dataset.
