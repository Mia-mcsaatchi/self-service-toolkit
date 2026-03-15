# Self-Service Toolkit

A no-code data enrichment pipeline that processes tabular data through OpenAI's GPT-4o-mini and exports structured results. Built for Jupyter / Google Colab.

## What It Does

Upload a spreadsheet, define output fields with plain-English prompts, and the tool sends each row through GPT-4o-mini — returning structured data you can export back to CSV, Excel, or Google Sheets.

## Blocks

| Block | Name | Purpose |
|-------|------|---------|
| **Block 1** | Data Loader | Upload CSV/XLSX or connect a Google Sheet |
| **Block K** | API Key Setup | Set your OpenAI API key for the session |
| **Block P** | Base Prompt | System-level instruction used by all LLM calls |
| **Block C** | Field Config Builder | Define output columns and their prompts interactively |
| **Block R** | Run & Export | Process all rows asynchronously and export results |
| **Block S** | Web Server *(optional)* | Expose the pipeline as a public API via Flask + Cloudflare tunnel |

## Getting Started

### 1. Open in Google Colab

Upload `Packed_Updated (2).ipynb` to [Google Colab](https://colab.research.google.com).

### 2. Load your data (Block 1)

- **Upload a file:** drag in a `.csv` or `.xlsx`
- **Google Sheet:** paste the sheet URL, click "Fetch tabs", select a worksheet, then click "Load table"

### 3. Set your OpenAI API key (Block K)

Paste your key (starts with `sk-`) and click **Set key**. The key is stored in the session only.

### 4. Configure output fields (Block C)

1. Click **Refresh from DataFrame** to load your column names
2. Click **+ Add field** for each new column you want
3. For each field, fill in:
   - **Field name** — the output column name
   - **Reads from** — which source columns to include as context
   - **Prompt** — plain-English instruction (e.g. *"Classify the sentiment as positive, negative, or neutral"*)
4. Use **Dependent group** mode when multiple output fields share one prompt

### 5. Run and export (Block R)

Configure the output settings at the top of Block R:

```python
OUTPUT_TARGET  = "local"          # "local" or "gsheet"
OUTPUT_PATH    = "/content/output.csv"
OUTPUT_FORMAT  = "csv"            # "csv" or "xlsx"
MAX_ROWS       = 0                # 0 = all rows
MAX_CONCURRENT = 10               # parallel API calls
```

Then run the block. Results are saved to the path you specified.

## Optional: Web API (Block S)

Block S starts a Flask server and creates a public Cloudflare tunnel URL. Use this to connect the pipeline to a frontend without exposing your Colab session directly.

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Check server status |
| POST | `/api/upload-data` | Send parsed row data |
| POST | `/api/upload-config` | Send field config JSON |
| POST | `/api/set-api-key` | Set OpenAI key remotely |
| POST | `/api/process` | Run the pipeline |
| GET | `/api/export-csv` | Download results as CSV |
| GET | `/api/export-xlsx` | Download results as XLSX |

## Requirements

The notebook auto-detects the environment. Install missing packages in Colab with:

```
!pip install ipywidgets gspread google-auth aiohttp nest-asyncio tqdm openpyxl flask flask-cors
```

## Notes

- Each row makes one API call returning all fields as a JSON object
- If a field is ambiguous or missing data, the model returns `"unsure"`
- Google Sheets support requires Colab OAuth — click "Authorize Google" when prompted
