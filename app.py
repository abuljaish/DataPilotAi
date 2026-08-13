import json
import os
import re
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from google import genai
from google.genai import types

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, 'backend', '.env'))

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable Cross-Origin Resource Sharing for local frontend communication

# Configuration
UPLOAD_FOLDER = os.path.join(PROJECT_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Single-user active dataset storage in memory (for MVP development)
ACTIVE_DATASET = {
    "filename": None,
    "filepath": None,
    "df": None
}


def _normalize_col_name(name: str) -> str:
    """Normalize a column name for identifier pattern matching."""
    if not isinstance(name, str):
        return ""
    # Lowercase, remove spaces, underscores, dots, hyphens and non-alphanumerics
    s = name.lower()
    s = re.sub(r"[\s_\.-]", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def detect_identifier_columns(df: pd.DataFrame, numeric_cols: list) -> list:
    """Return a list of columns from numeric_cols that are confidently identifiers/indexes.

    Detection uses multiple signals:
    - name patterns (normalized)
    - sequential numeric behavior (constant or near-constant step)
    - high uniqueness combined with other signals
    - monotonic row-order behavior
    """
    strong_name_patterns = {
        'id', 'sno', 'serialno', 'serialnumber', 'recordid', 'employeeid',
        'userid', 'customerid', 'index', 'rownumber', 'rowno', 'number', 'serial'
    }
    # safe words that should never be auto-classified as identifiers
    safe_terms = {
        'age', 'salary', 'revenue', 'price', 'score', 'temperature', 'experience',
        'experienceyears', 'performance', 'performance_score', 'performancescore',
        'amount', 'total', 'value', 'quantity', 'qty'
    }

    excluded = []
    for col in numeric_cols:
        series = df[col].dropna()
        non_null_count = len(series)
        if non_null_count == 0:
            continue

        norm = _normalize_col_name(col)

        # Name-based signals
        name_strong = norm in strong_name_patterns
        name_ends_id = norm.endswith('id') and len(norm) > 2 and norm not in safe_terms

        # Uniqueness
        unique_count = int(df[col].nunique(dropna=True))
        unique_prop = unique_count / non_null_count if non_null_count else 0.0

        # Sequential detection: diffs between consecutive rows
        seq_signal = False
        median_step = None
        try:
            # operate on original row order
            vals = pd.to_numeric(series, errors='coerce').dropna()
            if len(vals) >= 3 and all(vals.apply(float).notnull()):
                diffs = vals.astype('float64').diff().dropna()
                if len(diffs) >= 2:
                    # mode-like behavior: most diffs close to a single value
                    median_step = float(diffs.median())
                    # proportion of diffs within small tolerance of median_step
                    tol = max(abs(median_step) * 0.001, 1e-6)
                    close = diffs.apply(lambda d: abs(d - median_step) <= tol)
                    prop_close = close.sum() / len(diffs)
                    if prop_close >= 0.9 and median_step != 0:
                        seq_signal = True
        except Exception:
            seq_signal = False

        # Monotonic behavior across rows
        monotonic = False
        try:
            if len(series) >= 3:
                monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing
        except Exception:
            monotonic = False

        # Decision rules (conservative): require combination of signals unless name is an exact strong match
        is_identifier = False

        if norm in {'id', 'index', 'sno', 'serialno', 'serialnumber', 'recordid'}:
            is_identifier = True
        elif name_strong or name_ends_id:
            # If name looks identifier-like, require either high uniqueness or sequential/monotonic evidence
            if unique_prop >= 0.85 or seq_signal or monotonic:
                is_identifier = True
        # NOTE: sequential behavior alone (without an identifier-like name) is not sufficient
        # to mark a column as an identifier to avoid incorrectly excluding measurements
        # such as experience, salary, etc.

        # Extra safety: never auto-exclude clearly safe terms
        if norm in safe_terms:
            is_identifier = False

        if is_identifier:
            excluded.append(col)

    return excluded


@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# 1. CSV UPLOAD ENDPOINT
# ==========================================
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'csv-file' not in request.files and 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Please select a CSV file."}), 400

    file = request.files.get('csv-file') or request.files.get('file')

    if not file or file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Invalid file format. Only CSV files (.csv) are allowed."}), 400

    try:
        # Save file to uploads directory
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read CSV dataset using Pandas
        df = pd.read_csv(filepath)

        if df.empty:
            return jsonify({"error": "The uploaded CSV file is empty."}), 400

        # Store dataset in active memory
        ACTIVE_DATASET["filename"] = filename
        ACTIVE_DATASET["filepath"] = filepath
        ACTIVE_DATASET["df"] = df

        # Column data classification
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        total_missing = int(df.isnull().sum().sum())
        missing_values = {
            col: int(count)
            for col, count in df.isnull().sum().items()
            if count > 0
        }

        # Build histogram data for each numeric column without hardcoding names.
        distribution_data = {}
        for col in numeric_cols:
            values = df[col].dropna()
            if values.empty:
                continue

            unique_count = int(values.nunique())
            if unique_count == 1:
                distribution_data[col] = {
                    "labels": [str(values.iloc[0])],
                    "values": [int(len(values))]
                }
                continue

            bin_count = min(10, unique_count)
            bins = pd.cut(values, bins=bin_count, include_lowest=True)
            counts = bins.value_counts(sort=False)
            distribution_data[col] = {
                "labels": [str(interval) for interval in counts.index],
                "values": [int(count) for count in counts.values]
            }

        # Calculate correlations only when there are enough numeric columns.
        correlation_data = {"columns": [], "matrix": []}
        excluded_identifier_columns = detect_identifier_columns(df, numeric_cols)
        # Use filtered numeric columns for correlation calculations
        correlation_numeric_cols = [c for c in numeric_cols if c not in excluded_identifier_columns]

        if len(correlation_numeric_cols) >= 2:
            correlation_matrix = df[correlation_numeric_cols].corr()
            correlation_data = {
                "columns": correlation_numeric_cols,
                "matrix": [
                    [None if pd.isna(value) else round(float(value), 3) for value in row]
                    for row in correlation_matrix.values
                ]
            }

        # Perform Exploratory Data Analysis (EDA) per column
        eda_summary = {}
        for col in df.columns:
            is_num = col in numeric_cols
            missing_cnt = int(df[col].isnull().sum())
            unique_cnt = int(df[col].nunique())
            
            if is_num:
                mean_val = round(float(df[col].mean()), 2) if not df[col].isnull().all() else "--"
                min_val = round(float(df[col].min()), 2) if not df[col].isnull().all() else "--"
                max_val = round(float(df[col].max()), 2) if not df[col].isnull().all() else "--"
            else:
                mean_val, min_val, max_val = "--", "--", "--"

            eda_summary[col] = {
                "is_numeric": is_num,
                "data_type": str(df[col].dtype),
                "mean": mean_val,
                "min": min_val,
                "max": max_val,
                "missing": missing_cnt,
                "unique": unique_cnt
            }

        # First 10 rows preview
        preview_data = df.head(10).fillna('').to_dict(orient='records')

        response_data = {
            "filename": filename,
            "total_rows": int(len(df)),
            "total_cols": int(len(df.columns)),
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "correlation_numeric_columns": correlation_data.get("columns", []),
            "categorical_columns": categorical_cols,
            "total_missing": total_missing,
            "missing_values": missing_values,
            "distribution_data": distribution_data,
            "correlation_data": correlation_data,
            "excluded_identifier_columns": excluded_identifier_columns,
            "preview_data": preview_data,
            "eda_summary": eda_summary
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"Error parsing CSV file: {str(e)}"}), 500


# ==========================================
# 2. ASK YOUR DATA ENDPOINT
# ==========================================
@app.route('/api/ask', methods=['POST'])
def ask_question():
    df = ACTIVE_DATASET.get("df")
    if df is None:
        return jsonify({"error": "No dataset uploaded yet. Please upload a CSV file first."}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question to analyze."}), 400

    try:
        dataset_info = build_dataset_info(df)
        fallback_used = False

        # Ask Gemini (or fallback) for a structured plan, then execute it dynamically
        try:
            plan = ask_llm_plan(question, dataset_info)
            fallback_used = bool(plan.pop("_fallback_used", False))
        except RuntimeError as llm_error:
            err_text = str(llm_error).lower()
            if "quota" in err_text or "429" in err_text or "rate limit" in err_text or "billing" in err_text:
                plan = build_fallback_plan(question, df)
                fallback_used = True
            else:
                raise

        exec_result = execute_plan(plan, df)

        # Aggregate natural-language answers into a single answer string for compatibility with frontend
        answers = exec_result.get('answers', [])
        combined_answer = ' '.join([a.get('answer', '') for a in answers if a.get('answer')])
        # pick first non-empty data payload for backward-compatibility
        data_payload = next((a.get('data', []) for a in answers if a.get('data')), [])

        return jsonify({
            "success": True,
            "question": question,
            "dataset": ACTIVE_DATASET["filename"],
            "answer": combined_answer,
            "data": data_payload,
            "insight": combined_answer,
            "detailed": answers,
            "charts": exec_result.get('charts', []),
            "ai_mode": "fallback" if fallback_used else "gemini"
        }), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Error analyzing query: {str(e)}"}), 500


# ==========================================
# 3. LLM + PANDAS ANALYSIS ENGINE
# ==========================================
GEMINI_MODEL = "gemini-3.5-flash-lite"

ANALYSIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["mean", "sum", "min", "max", "count", "filter", "unsupported"]
        },
        "column": {"type": ["string", "null"]},
        "group_by": {"type": ["array", "null"], "items": {"type": "string"}},
        "order": {"type": ["string", "null"], "enum": ["asc", "desc", None]},
        "limit": {"type": ["integer", "null"]},
        "select": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {"type": "string", "enum": ["==", "!=", ">", "<", ">=", "<="]},
                    "value": {"type": ["string", "number", "boolean", "null"]}
                },
                "required": ["column", "operator", "value"]
            }
        }
    },
    "required": ["operation"]
}


# New JSON schema for Gemini to return a structured multi-question plan
PLAN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["answer", "chart", "both"]},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["count","sum","mean","min","max","list","group_by","comparison"]},
                    "target_column": {"type": ["string", "null"]},
                    "entity_column": {"type": ["string", "null"]},
                    "group_by": {"type": ["string", "array", "null"]},
                    "groups": {"type": ["array", "null"], "items": {"type":"string"}},
                    "filters": {"type": ["array", "null"]},
                    "sort_order": {"type": ["string", "null"], "enum": ["asc", "desc", None]},
                    "limit": {"type": ["integer","null"]},
                    "chart_type": {"type": ["string","null"]},
                    "aggregation": {"type": ["string", "null"], "enum": ["count", "sum", "mean", "min", "max", None]},
                    "question_text": {"type": "string"}
                }
            }
        }
    },
    "required": ["intent","questions"]
}


def ask_llm_plan(question: str, dataset_info: Dict[str, Any]):
    """Ask Gemini to parse the user's natural language into a structured plan.

    Returns a dict matching PLAN_RESPONSE_SCHEMA.
    """
    prompt = (
        "You are a data-analysis planner that returns a structured plan (JSON) for executing on a tabular dataset.\n"
        "Return only JSON that conforms to the schema. Do not include any explanation.\n"
        "Fields: intent (answer|chart|both), questions (array). Each question should include operation, target_column, group_by, filters, sort_order, limit, entity_column (for min/max), groups (for comparisons), chart_type when requested, and question_text.\n"
        "Use only column names present in the dataset_info.columns. If a referenced column is not present, still include the intended name and we will validate server-side.\n"
    )

    user_input = json.dumps({"question": question, "dataset_info": dataset_info}) + "\n\n" + prompt
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction="Return a JSON plan only.",
                response_mime_type="application/json",
                response_json_schema=PLAN_RESPONSE_SCHEMA,
            ),
        )
        content = response.text.strip()
        if not content:
            raise ValueError("Empty response from Gemini API when requesting plan.")
        parsed = json.loads(content)
        return parsed
    except Exception as exc:
        message = str(exc).lower()
        if "429" in message or "quota" in message or "rate limit" in message or "billing" in message:
            raise RuntimeError("Gemini API quota exceeded. Please add billing or retry later.") from exc
        return build_fallback_plan(question, dataset_info)


def build_fallback_plan(question: str, dataset: Any):
    """Build a safe, useful plan when Gemini is unavailable.

    ``dataset`` may be a DataFrame (endpoint fallback) or the serializable
    metadata returned by ``build_dataset_info`` (LLM helper fallback).
    """
    if isinstance(dataset, pd.DataFrame):
        columns = dataset.columns.tolist()
        numeric_cols = dataset.select_dtypes(include=["number"]).columns.tolist()
    else:
        dataset = dataset or {}
        columns = dataset.get("columns", [])
        numeric_cols = dataset.get("numeric_columns", [])

    q = question.casefold()
    target = next((c for c in numeric_cols if c.casefold() in q), numeric_cols[0] if numeric_cols else None)
    grouping = next((c for c in columns if c not in numeric_cols and c.casefold() in q), None)
    entity = next((c for c in columns if c.casefold() in q and c != target), None)
    if entity is None and any(word in q for word in ("who", "which person", "employee")):
        entity = next((c for c in columns if c.casefold() in {"name", "employee", "employee name"}), None)
    operation = "count"
    if any(word in q for word in ("average", "mean", "avg")):
        operation = "mean"
    elif any(word in q for word in ("total", "sum")):
        operation = "sum"
    elif any(word in q for word in ("minimum", "lowest", "smallest", "min")):
        operation = "min"
    elif any(word in q for word in ("maximum", "highest", "largest", "max", "oldest")):
        operation = "max"

    question_plan = {
        "operation": operation,
        "target_column": target if operation != "count" else None,
        "entity_column": entity if operation in {"min", "max"} else None,
        "filters": [],
        "question_text": question,
    }
    if grouping and (" by " in q or "compare" in q):
        question_plan.update({"operation": "comparison", "target_column": target, "group_by": grouping, "aggregation": operation})
    return {"intent": "answer", "questions": [question_plan], "_fallback_used": True}


def _format_currency_like(x):
    try:
        if abs(x) >= 1e6:
            return f"{x:,.2f}"
        if abs(x) >= 1e3:
            return f"{x:,.2f}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def execute_plan(plan: Dict[str, Any], df: pd.DataFrame):
    """Execute the structured plan on the dataframe and return natural-language answers and optional charts.

    Returns: {"answers": [..], "charts": [..]} where answers are dicts {question_text, answer, data}
    """
    answers = []
    charts = []

    for q in plan.get('questions', []):
        qtext = q.get('question_text') or ''
        op = (q.get('operation') or '').lower()
        target = q.get('target_column')
        entity_col = q.get('entity_column')
        group_by = q.get('group_by')
        groups = q.get('groups') or []
        filters = q.get('filters') or []
        chart_type = q.get('chart_type')
        sort_order = (q.get('sort_order') or '').lower() if q.get('sort_order') else None
        limit = q.get('limit')

        # Validate columns exist
        missing_cols = []
        if target and target not in df.columns:
            missing_cols.append(target)
        if entity_col and entity_col not in df.columns:
            missing_cols.append(entity_col)
        if isinstance(group_by, str) and group_by and group_by not in df.columns:
            missing_cols.append(group_by)
        if isinstance(group_by, list):
            for g in group_by:
                if g not in df.columns:
                    missing_cols.append(g)

        if missing_cols:
            answers.append({"question_text": qtext, "answer": f"Could not find column(s): {', '.join(sorted(set(missing_cols)))}.", "data": []})
            continue

        # Apply filters using existing helper
        try:
            working = apply_filters(df, filters)
        except Exception as e:
            answers.append({"question_text": qtext, "answer": f"Error applying filters: {str(e)}", "data": []})
            continue

        # Implement operations
        try:
            gb_cols = group_by if isinstance(group_by, list) else ([group_by] if group_by else [])
            # A normal aggregate with group_by is a grouped aggregate too; this
            # lets Gemini express count/sum/mean/min/max directly.
            if gb_cols and op in {"count", "sum", "mean", "min", "max"}:
                if op == "count":
                    grouped = working.groupby(gb_cols, dropna=False).size().reset_index(name="count")
                    value_name = "count"
                else:
                    if not target or not pd.api.types.is_numeric_dtype(working[target]):
                        raise ValueError(f"Column to aggregate '{target}' is missing or not numeric.")
                    grouped = working.groupby(gb_cols, dropna=False)[target].agg(op).reset_index(name=op)
                    value_name = op
                if groups:
                    grouped = grouped[grouped[gb_cols[0]].astype(str).isin([str(g) for g in groups])]
                grouped = grouped.sort_values(value_name, ascending=(sort_order == "asc"))
                if limit:
                    grouped = grouped.head(int(limit))
                rows = grouped.to_dict(orient="records")
                answers.append({"question_text": qtext, "answer": f"Grouped {op} by {', '.join(gb_cols)}.", "data": rows})
                if chart_type and plan.get("intent") in {"chart", "both"}:
                    charts.append({"title": qtext or f"{op.title()} by {gb_cols[0]}", "labels": [r[gb_cols[0]] for r in rows], "values": [r[value_name] for r in rows], "type": chart_type})
                continue

            if op == 'count':
                if target:
                    val = int(working[target].count())
                    ans = f"The count of '{target}' is {val:,}."
                    data = [{"column": target, "count": val}]
                else:
                    val = int(len(working))
                    ans = f"The dataset has {val:,} rows after filters."
                    data = [{"count": val}]
                answers.append({"question_text": qtext, "answer": ans, "data": data})

            elif op in {'sum', 'mean', 'min', 'max'}:
                if not target:
                    answers.append({"question_text": qtext, "answer": "No target numeric column provided.", "data": []})
                    continue
                if not pd.api.types.is_numeric_dtype(working[target]):
                    answers.append({"question_text": qtext, "answer": f"Column '{target}' is not numeric.", "data": []})
                    continue

                if op == 'sum':
                    val = float(working[target].sum())
                    ans = f"The sum of '{target}' is {val:,.2f}."
                elif op == 'mean':
                    val = float(working[target].mean())
                    ans = f"The average of '{target}' is {val:,.2f}."
                elif op == 'min':
                    val = float(working[target].min())
                    if entity_col and entity_col in df.columns:
                        rows = working[working[target] == val]
                        entities = rows[entity_col].dropna().astype(str).unique().tolist()
                        if entities:
                            ent_text = ', '.join(entities[:3])
                            ans = f"The minimum '{target}' is {val}, for {entity_col}: {ent_text}."
                        else:
                            ans = f"The minimum '{target}' is {val}."
                    else:
                        ans = f"The minimum '{target}' is {val}."
                else:  # max
                    val = float(working[target].max())
                    if entity_col and entity_col in df.columns:
                        rows = working[working[target] == val]
                        entities = rows[entity_col].dropna().astype(str).unique().tolist()
                        if entities:
                            ent_text = ', '.join(entities[:3])
                            ans = f"The maximum '{target}' is {val}, for {entity_col}: {ent_text}."
                        else:
                            ans = f"The maximum '{target}' is {val}."
                    else:
                        ans = f"The maximum '{target}' is {val}."

                answers.append({"question_text": qtext, "answer": ans, "data": [{"column": target, op: val}]})

            elif op == 'group_by' or op == 'comparison':
                # Determine group_by column
                gb = group_by
                if isinstance(gb, list):
                    gb_cols = gb
                else:
                    gb_cols = [gb] if gb else []

                if not gb_cols:
                    answers.append({"question_text": qtext, "answer": "No group_by column provided for grouping/comparison.", "data": []})
                    continue

                # Compute aggregation. Comparisons default to averages, while
                # group_by plans can explicitly request count/sum/min/max.
                agg_col = target
                aggregation = (q.get("aggregation") or "mean").lower()
                if aggregation not in {"count", "sum", "mean", "min", "max"}:
                    raise ValueError(f"Unsupported aggregation: {aggregation}")
                if aggregation != "count" and (not agg_col or not pd.api.types.is_numeric_dtype(working[agg_col])):
                    answers.append({"question_text": qtext, "answer": f"Column to aggregate '{agg_col}' is missing or not numeric.", "data": []})
                    continue

                if aggregation == "count":
                    grouped = working.groupby(gb_cols, dropna=False).size().reset_index(name="count")
                    value_name = "count"
                else:
                    grouped = working.groupby(gb_cols, dropna=False)[agg_col].agg(aggregation).reset_index(name=agg_col)
                    value_name = agg_col
                # Optionally filter to requested groups
                if groups:
                    grouped = grouped[grouped[gb_cols[0]].astype(str).isin(groups)]

                # Build natural language comparative answer
                rows = grouped.to_dict(orient='records')
                if not rows:
                    answers.append({"question_text": qtext, "answer": "No groups found for comparison.", "data": []})
                    continue

                parts = []
                for r in rows:
                    name = r[gb_cols[0]]
                    val = r[value_name]
                    parts.append(f"{name} has a {aggregation} of {val:,.2f}")

                # decide winner
                if len(rows) >= 2:
                    best = max(rows, key=lambda x: x[value_name])
                    winner = best[gb_cols[0]]
                    parts_text = ', while '.join(parts)
                    ans = f"{parts_text}. {winner} has the highest {aggregation}."
                else:
                    ans = parts[0]

                answers.append({"question_text": qtext, "answer": ans, "data": rows})

                # Chart only if requested and chart_type present
                if chart_type and plan.get('intent') in ('chart', 'both'):
                    labels = [r[gb_cols[0]] for r in rows]
                    values = [r[value_name] for r in rows]
                    charts.append({"title": qtext or f"{aggregation.title()} {agg_col or 'rows'} by {gb_cols[0]}", "labels": labels, "values": values, "type": chart_type})

            elif op == 'list':
                cols = q.get('select') or df.columns.tolist()
                # validate
                cols = [c for c in cols if c in df.columns]
                result_rows = working[cols].head(limit or 10).to_dict(orient='records')
                ans = f"Listing {len(result_rows)} rows."
                answers.append({"question_text": qtext, "answer": ans, "data": result_rows})

            else:
                answers.append({"question_text": qtext, "answer": f"Unsupported operation: {op}", "data": []})

        except Exception as exc:
            answers.append({"question_text": qtext, "answer": f"Error computing result: {str(exc)}", "data": []})

    return {"answers": answers, "charts": charts}


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError("GEMINI_API_KEY is missing or still set to the placeholder value. Add your real key to the backend .env file.")

    return genai.Client(api_key=api_key)


def build_dataset_info(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    column_stats = {}
    for col in df.columns:
        values = {
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique())
        }
        if col in numeric_cols:
            values["mean"] = round(float(df[col].mean()), 2) if len(df[col].dropna()) else None
            values["min"] = round(float(df[col].min()), 2) if len(df[col].dropna()) else None
            values["max"] = round(float(df[col].max()), 2) if len(df[col].dropna()) else None
            values["sum"] = round(float(df[col].sum()), 2) if len(df[col].dropna()) else 0
        column_stats[col] = values

    return {
        "columns": df.columns.tolist(),
        "total_rows": int(len(df)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_stats": column_stats
    }


def ask_llm(question: str, dataset_info: Dict[str, Any]):
    prompt = """
You are a data-analysis planner.
Your job is to decide which simple analytical operation should be used.
Return ONLY valid JSON.

Rules:
- Allowed operations: mean, sum, min, max, count, filter, unsupported.
- Use only the columns from the dataset metadata.
- Do not produce Python code.
- Do not use any columns that are not provided.
- Return a query plan only. Never calculate an answer yourself.
- Put every question condition in the "filters" array. Each filter needs "column", "operator", and "value".
- Supported filter operators are: ==, !=, >, <, >=, <=.
- Use "group_by" as a list of one or more grouping columns when grouping is needed.
- Use "order" as "asc" or "desc" when the results should be sorted.
- Use "limit" for top-N or a single highest/lowest result.
- Use "select" as a list of columns to return when the user requests particular output columns.
- For a simple total, return {"operation": "sum", "column": "revenue", "filters": []}
- For "What is the average Fare where Pclass = 1?", return:
{
  "operation": "mean",
  "column": "Fare",
  "filters": [{"column": "Pclass", "operator": "==", "value": 1}]
}
- For count, return {"operation": "count", "column": "customer_id"}
- For minimum or maximum, return {"operation": "min" or "max", "column": "revenue"}
- For a row filter, return {"operation": "filter", "filters": [{"column": "region", "operator": "==", "value": "north"}], "select": ["region"]}
- For "Which city has the highest number of matches played?", return:
{"operation": "count", "column": null, "group_by": ["city"], "order": "desc", "limit": 1, "filters": [], "select": ["city", "count"]}
- For the lowest count, use the same plan with "order": "asc" and "limit": 1.
- If the question is not supported, return {"operation": "unsupported"}

Return JSON only, no markdown.
"""

    user_input = json.dumps({"question": question, "dataset_info": dataset_info}) + "\n\n" + prompt
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction="You are a careful data-analysis planner. Return only valid JSON with a supported analysis operation. No markdown. Do not generate Python code.",
                response_mime_type="application/json",
                response_json_schema=ANALYSIS_RESPONSE_SCHEMA,
            ),
        )
        content = response.text.strip()

        if not content:
            raise ValueError("Empty response from Gemini API.")

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini response was not a JSON object.")
        return parsed
    except Exception as exc:
        message = str(exc).lower()
        if "429" in message or "quota" in message or "rate limit" in message or "billing" in message:
            raise RuntimeError("Gemini API quota exceeded. Your free-tier limit has been reached. Please add billing in Google AI Studio or retry later.") from exc
        raise RuntimeError(f"LLM request failed: {str(exc)}") from exc


def validate_column(df, col_name, allow_missing=False):
    if col_name is None:
        if allow_missing:
            return None
        raise ValueError("The LLM response did not include a required column name.")
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' was not found in the uploaded dataset.")
    return col_name


def apply_filters(df, filters):
    """Apply every structured filter to a dataframe before an operation runs."""
    if not filters:
        return df
    if not isinstance(filters, list):
        raise ValueError("The LLM filters must be a list.")

    filtered = df
    supported_operators = {"==", "!=", ">", "<", ">=", "<="}

    for filter_item in filters:
        if not isinstance(filter_item, dict):
            raise ValueError("Each filter must include a column, operator, and value.")

        column = validate_column(df, filter_item.get("column"))
        operator = filter_item.get("operator")
        value = filter_item.get("value")
        if operator not in supported_operators:
            raise ValueError(f"Unsupported filter operator: {operator}")
        if value is None:
            raise ValueError(f"The filter for '{column}' is missing a value.")

        series = filtered[column]
        if pd.api.types.is_numeric_dtype(series):
            try:
                comparison_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Filter value '{value}' is not numeric for column '{column}'.") from exc
        else:
            comparison_value = str(value).casefold()
            series = series.astype(str).str.casefold()

        if operator == "==":
            mask = series == comparison_value
        elif operator == "!=":
            mask = series != comparison_value
        elif operator == ">":
            mask = series > comparison_value
        elif operator == "<":
            mask = series < comparison_value
        elif operator == ">=":
            mask = series >= comparison_value
        else:  # operator == "<="
            mask = series <= comparison_value

        filtered = filtered[mask]

    return filtered


def build_fallback_instruction(question: str, df):
    q = question.lower().strip()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    if not numeric_cols:
        return {"operation": "unsupported"}

    top_match = re.search(r"top\s+(\d+)", q)
    if top_match:
        n = int(top_match.group(1))
        target_num_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        target_cat_col = next((c for c in categorical_cols if c.lower() in q), None)
        if target_cat_col:
            return {"operation": "top_n", "column": target_num_col, "group_by": target_cat_col, "n": n}
        return {"operation": "top_n", "column": target_num_col, "group_by": categorical_cols[0] if categorical_cols else df.columns[0], "n": n}

    if any(term in q for term in ["average", "mean", "avg"]):
        target_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        return {"operation": "mean", "column": target_col, "filters": []}

    if any(term in q for term in ["total", "sum of", "sum"]):
        target_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        return {"operation": "sum", "column": target_col, "filters": []}

    if "count" in q or "number of" in q:
        target_col = next((c for c in df.columns if c.lower() in q), df.columns[0])
        return {"operation": "count", "column": target_col, "filters": []}

    if any(term in q for term in ["minimum", "min", "lowest", "smallest"]):
        target_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        return {"operation": "min", "column": target_col, "filters": []}

    if any(term in q for term in ["maximum", "max", "highest", "largest"]):
        target_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        return {"operation": "max", "column": target_col, "filters": []}

    if any(term in q for term in ["group by", "by category", "by region", "by product"]):
        target_num_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        target_cat_col = next((c for c in categorical_cols if c.lower() in q), categorical_cols[0] if categorical_cols else df.columns[0])
        return {"operation": "group_by", "column": target_num_col, "group_by": target_cat_col}

    return {"operation": "unsupported"}


def execute_pandas_query(df, instruction):
    if not isinstance(instruction, dict):
        raise ValueError("LLM response was not in the expected format.")

    operation = str(instruction.get('operation', 'unsupported')).strip().lower()

    if operation == 'unsupported':
        raise ValueError("Unsupported analysis type. Please try another query.")

    # Filtering always happens before grouping, aggregation, or row preview.
    filters = instruction.get('filters', [])
    filtered_df = apply_filters(df, filters)
    if filtered_df.empty:
        return {"answer": "No rows matched the requested filters.", "data": []}

    if operation == 'filter':
        if not filters:
            raise ValueError("A filter operation requires at least one filter.")
        selected_columns = instruction.get('select') or filtered_df.columns.tolist()
        for column in selected_columns:
            validate_column(filtered_df, column)
        return {
            "answer": f"The filters returned {len(filtered_df):,} rows.",
            "data": filtered_df[selected_columns].head(10).to_dict(orient='records')
        }

    if operation not in {'mean', 'sum', 'min', 'max', 'count'}:
        raise ValueError(f"Unsupported operation: {operation}")

    column = instruction.get('column')
    if operation != 'count' or column is not None:
        column = validate_column(df, column)

    group_by = instruction.get('group_by') or []
    if isinstance(group_by, str):  # Accept older plans while Gemini now returns a list.
        group_by = [group_by]
    if not isinstance(group_by, list):
        raise ValueError("The group_by field must be a list of columns.")
    for group_column in group_by:
        validate_column(df, group_column)

    order = str(instruction.get('order') or 'desc').lower()
    if order not in {'asc', 'desc'}:
        raise ValueError("Order must be 'asc' or 'desc'.")
    limit = instruction.get('limit')
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("Limit must be a positive integer.")

    result_name = operation
    if group_by:
        if operation == 'count':
            if column is None:
                result_df = filtered_df.groupby(group_by, dropna=False).size().reset_index(name=result_name)
            else:
                result_df = filtered_df.groupby(group_by, dropna=False)[column].count().reset_index(name=result_name)
        else:
            result_df = filtered_df.groupby(group_by, dropna=False)[column].agg(operation).reset_index(name=result_name)
        result_df = result_df.sort_values(by=result_name, ascending=(order == 'asc'))
        if limit is not None:
            result_df = result_df.head(limit)

        selected_columns = instruction.get('select') or (group_by + [result_name])
        missing_columns = [name for name in selected_columns if name not in result_df.columns]
        if missing_columns:
            raise ValueError(f"Selected result columns were not found: {', '.join(missing_columns)}")
        return {
            "answer": f"Grouped {operation} result by {', '.join(group_by)}.",
            "data": result_df[selected_columns].to_dict(orient='records')
        }

    if operation == 'count':
        value = int(len(filtered_df) if column is None else filtered_df[column].count())
    else:
        value = float(getattr(filtered_df[column], operation)())

    return {
        "answer": f"The {operation} of '{column}' is {value:,.2f}." if operation != 'count' else f"The count is {value:,}.",
        "data": [{"column": column, result_name: value}]
    }


def generate_result_insight(question: str, result_data):
    if not result_data:
        return "No result data was produced for this question."

    try:
        small_payload = {
            "question": question,
            "result": result_data[:5]
        }

        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                "Write one short, human-readable sentence based only on this result data. Do not use markdown.\n\n"
                + json.dumps(small_payload)
            ),
        )
        if response.text:
            return response.text.strip()
    except Exception:
        pass

    return "The data was processed successfully and is ready for review."


# ==========================================
# 4. PANDAS QUERY PROCESSING ENGINE
# ==========================================
def process_query_with_pandas(question, df):
    """
    Safely performs data calculations using Pandas based on common data analysis intents:
    Top values, sums, averages, min/max, counts, and group-by aggregations.
    """
    q_lower = question.lower()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # Detect Top N requests (e.g. "top 5 products by revenue")
    top_match = re.search(r'top\s+(\d+)', q_lower)
    n = int(top_match.group(1)) if top_match else 5

    # Match target columns from question string
    target_num_col = next((c for c in numeric_cols if c.lower() in q_lower), numeric_cols[0] if numeric_cols else None)
    target_cat_col = next((c for c in categorical_cols if c.lower() in q_lower), categorical_cols[0] if categorical_cols else None)

    # Case A: Top N grouped query
    if "top" in q_lower and target_num_col:
        if target_cat_col:
            grouped = df.groupby(target_cat_col)[target_num_col].sum().reset_index()
            sorted_df = grouped.sort_values(by=target_num_col, ascending=False).head(n)
            results_str = ", ".join([f"{row[target_cat_col]}: {row[target_num_col]:,.2f}" for _, row in sorted_df.iterrows()])
            return f"Top {n} '{target_cat_col}' by total '{target_num_col}': {results_str}."
        else:
            sorted_df = df.sort_values(by=target_num_col, ascending=False).head(n)
            return f"Top {n} rows sorted by '{target_num_col}' with maximum recorded value of {sorted_df[target_num_col].max():,.2f}."

    # Case B: Average / Mean query
    if "average" in q_lower or "mean" in q_lower:
        if target_num_col:
            avg_val = df[target_num_col].mean()
            return f"The average '{target_num_col}' across all {len(df):,} rows is {avg_val:,.2f}."

    # Case C: Total / Sum query
    if "total" in q_lower or "sum" in q_lower:
        if target_num_col:
            sum_val = df[target_num_col].sum()
            return f"The total sum of '{target_num_col}' is {sum_val:,.2f}."

    # Case D: Highest / Maximum query
    if "highest" in q_lower or "maximum" in q_lower or "max" in q_lower:
        if target_num_col:
            max_idx = df[target_num_col].idxmax()
            max_row = df.loc[max_idx]
            cat_info = f" ({target_cat_col}: {max_row[target_cat_col]})" if target_cat_col else ""
            return f"The highest '{target_num_col}' is {max_row[target_num_col]:,.2f}{cat_info}."

    # Fallback to AI function wrapper
    return ask_ai(question, df)


# ==========================================
# 4. AI API INTEGRATION PLACEHOLDER
# ==========================================
def ask_ai(question, df):
    """
    Sends dataset schema metadata and user question to an LLM (Gemini API / OpenAI API).
    Note: Full CSV data is never sent to the LLM to preserve privacy and performance.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    schema_info = {
        "columns": df.columns.tolist(),
        "total_rows": len(df),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=['number']).columns.tolist()
    }

    if not api_key:
        return (
            f"[AI Engine Placeholder] Query '{question}' received. "
            f"Dataset loaded with {schema_info['total_rows']:,} rows and columns [{', '.join(schema_info['columns'][:5])}]. "
            "To enable live AI analysis, configure GEMINI_API_KEY in your environment."
        )

    # When API key is provided, Gemini / OpenAI call happens here:
    # prompt = f"Given schema {schema_info}, answer question: {question}"
    # response = ai_client.generate(prompt)
    return f"AI API Response for '{question}' based on dataset schema."


# ==========================================
# RUN FLASK APP
# ==========================================
if __name__ == '__main__':
    print("🚀 DataPilot AI Backend starting on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(debug=True, port=5000)
