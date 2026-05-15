# LEAD: LLM-Enhanced Authority-driven Discovery

**LEAD** is a hybrid causal discovery framework designed to bridge the gap between purely statistical methods and real-world scientific knowledge. By leveraging Large Language Models (LLMs) and real-time retrieval from authoritative web domains (e.g., `.gov`, `.edu`, `.org`), LEAD identifies causal relationships with higher precision and interpretability.

## 📂 Project Structure

```text
LEAD/
├── core/                   # Core methodology implementation
│   ├── discovery/          # Statistical algorithms (PC, GES, NOTEARS) with LLM-priors
│   ├── extraction/         # LLM-based value extraction and causal verification
│   └── retriever/          # Authority-driven web retrieval & scraping logic
├── datasets/               # Input JSON datasets (nodes, synonyms, edges)
│   └── processed/          # Cached evidence and extracted CSV data
├── log/                    # System logs
│   └── experiment.log      # Detailed runtime tracing and LLM reasoning logs
├── results/                # Quantitative and qualitative outputs
│   ├── metrics/            # Detailed performance reports (JSON)
│   ├── plots/              # Visualized causal DAGs (PNG)
│   └── all_experiments_summary.csv  # Unified experimental logs (CSV)
├── config.yaml             # Centralized environment & hyperparameter configuration
├── main.py                 # Primary entry point for the discovery pipeline
├── metrics.py              # Specialized evaluation metrics (SHD, SID, NHD)
├── utils.py                # Adjacency matrix and graph utility functions
├── requirements.txt        # System dependencies
└── README.md               # Project documentation
```

## 🌟 Key Features

- **Authority-Driven Retrieval**: Unlike generic search, LEAD restricts its knowledge base to high-credibility domains to minimize noise and misinformation.
- **Adaptive Search Intensity**: The system dynamically scales its search limit based on the graph complexity ($L = |V|$), ensuring robust coverage for large datasets.
- **Consensus Verification**: A multi-document voting mechanism verifies causal claims, filtering out low-confidence relations before graph construction.
- **Integrated Evaluation**: Built-in support for standard causal metrics, including Structural Hamming Distance (SHD) and Structural Interventional Distance (SID).

## 🚀 Getting Started

### 1. Installation
The system requires Python 3.9–3.12 (**3.13 is not yet supported** by causal-learn). Install dependencies via `pip`:
```bash
pip install -r requirements.txt
```

### 2. Configuration
LEAD uses a centralized configuration for security and reproducibility. **Edit `config.yaml` before running**:
- **API Settings**: Provide your LLM API key and base URL.
- **Network**: Configure local proxy settings for web retrieval.
- **Logging**: Toggle between `INFO` (standard) and `DEBUG` (detailed reasoning) modes.

#### Managing API Keys Across Computers

**Recommended**: Use environment variables (takes priority over `config.yaml`):

```bash
# Linux/macOS
export LEAD_API_KEY="your-api-key"
export LEAD_API_BASE="https://api.example.com/v1/"

# Windows (Command Prompt)
set LEAD_API_KEY=your-api-key
set LEAD_API_BASE=https://api.example.com/v1/

# Windows (PowerShell)
$env:LEAD_API_KEY="your-api-key"
$env:LEAD_API_BASE="https://api.example.com/v1/"
```

On each computer, set these in your shell profile (`.bashrc`, `.zshrc`, or system environment variables) — then `config.yaml` stays the same everywhere with the placeholder `YOUR_API_KEY_HERE`.

#### Clearing Cached Results

To force a fresh re-run, delete the cached files under `datasets/processed/`:

| Cache File Pattern | Pipeline Stage |
|---|---|
| `{dataset}_local_docs_cache.json` | Part 1: Document retrieval |
| `{dataset}_{model}_extracted_table_data.csv` | Part 2: Variable value extraction |
| `{dataset}_explicit_causal_relation_evidence.json` | Part 3: Targeted causal evidence search |
| `{dataset}_{model}_causal_relation_verification_results.json` | Part 4: LLM causal claim verification |

Example — reset everything for the `cancer` dataset:
```bash
rm datasets/processed/cancer_*.json datasets/processed/cancer_*.csv
```

Or delete the entire `datasets/processed/` directory to clear all caches.

### 3. Running the Pipeline
Execute the full discovery process (Knowledge Retrieval $\rightarrow$ Value Extraction $\rightarrow$ Causal Discovery):

```bash
# Basic usage: Dataset (cancer), Algorithm (pc), Model (glm-4)
python main.py --dataset cancer --alg pc --llm glm-4-flash

# Advanced usage: Using GES on the ADNI dataset
python main.py --dataset adni --alg ges --llm glm-4-flash
```

## 📊 Experimental Evaluation

After each run, the system populates the `results/` directory:

- **Quantitative Metrics**: Each experiment generates a JSON file in `results/metrics/` containing Precision, Recall, F1-score, **SHD**, and **SID**.
- **Unified Summary**: All results are appended to `all_experiments_summary.csv`, allowing for immediate cross-algorithm performance analysis.
- **Visualization**: Predicted causal structures are rendered as PNG files in `results/plots/`.

## 🔬 Scientific Context
LEAD is developed to address the limitations of data-starved causal discovery. By integrating "Authority-driven" evidence, it provides a principled way to incorporate human-level expert knowledge into automated discovery pipelines.

---

### 📄 Citation
If you use LEAD in your research, please cite:
*(Insert your conference paper citation here)*

### ⚠️ Note on Reproducibility
For the purpose of anonymous review, ensure that `config.yaml` is stripped of private API credentials when sharing the repository. The provided `processed/` folder contains cached evidence to allow for result verification even without an active internet connection.