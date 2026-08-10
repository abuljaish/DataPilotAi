import json
import os
import re
from typing import Dict, Any

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for local frontend communication

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Single-user active dataset storage in memory (for MVP development)
ACTIVE_DATASET = {
    "filename": None,
    "filepath": None,
    "df": None
}

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
            "categorical_columns": categorical_cols,
            "total_missing": total_missing,
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

        try:
            llm_instruction = ask_llm(question, dataset_info)
        except RuntimeError as llm_error:
            err_text = str(llm_error).lower()
            if "quota" in err_text or "429" in err_text or "rate limit" in err_text or "billing" in err_text:
                llm_instruction = build_fallback_instruction(question, df)
                fallback_used = True
            else:
                raise

        result = execute_pandas_query(df, llm_instruction)
        insight = generate_result_insight(question, result.get("data", []))

        return jsonify({
            "success": True,
            "question": question,
            "dataset": ACTIVE_DATASET["filename"],
            "answer": result["answer"],
            "data": result.get("data", []),
            "insight": insight,
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
def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError("GEMINI_API_KEY is missing or still set to the placeholder value. Add your real key to the backend .env file.")

    genai.configure(api_key=api_key)
    return [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest"
    ]


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
- Allowed operations: sum, average, count, minimum, maximum, top_n, group_by, filter, unsupported.
- Use only the columns from the dataset metadata.
- Do not produce Python code.
- Do not use any columns that are not provided.
- If the question asks for top 5 products by revenue, return:
{
  "operation": "top_n",
  "column": "revenue",
  "group_by": "product",
  "n": 5
}
- For a simple total, return {"operation": "sum", "column": "revenue"}
- For average, return {"operation": "average", "column": "revenue"}
- For count, return {"operation": "count", "column": "customer_id"}
- For minimum or maximum, return {"operation": "minimum" or "maximum", "column": "revenue"}
- For filter, return {"operation": "filter", "filter_column": "region", "filter_value": "north"}
- For group by totals, return {"operation": "group_by", "column": "revenue", "group_by": "product"}
- If the question is not supported, return {"operation": "unsupported"}

Return JSON only, no markdown.
"""

    user_input = json.dumps({"question": question, "dataset_info": dataset_info}) + "\n\n" + prompt
    last_error = None

    for model_name in get_gemini_model():
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction="You are a careful data-analysis planner. Return only valid JSON with a supported analysis operation. No markdown. Do not generate Python code."
            )
            response = model.generate_content(user_input)
            content = response.text.strip()

            if not content:
                raise ValueError("Empty response from Gemini API.")

            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL)

            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Gemini response was not a JSON object.")
            return parsed
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "404" in str(exc) or "not found" in message:
                continue
            if "429" in message or "quota" in message or "rate limit" in message or "billing" in message:
                raise RuntimeError("Gemini API quota exceeded. Your free-tier limit has been reached. Please add billing in Google AI Studio or retry later.") from exc
            raise RuntimeError(f"LLM request failed: {str(exc)}") from exc

    raise RuntimeError(f"LLM request failed: {str(last_error)}") from last_error


def validate_column(df, col_name, allow_missing=False):
    if col_name is None:
        if allow_missing:
            return None
        raise ValueError("The LLM response did not include a required column name.")
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' was not found in the uploaded dataset.")
    return col_name


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
        return {"operation": "average", "column": target_col}

    if any(term in q for term in ["total", "sum of", "sum"]):
        target_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        return {"operation": "sum", "column": target_col}

    if "count" in q or "number of" in q:
        target_col = next((c for c in df.columns if c.lower() in q), df.columns[0])
        return {"operation": "count", "column": target_col}

    if any(term in q for term in ["minimum", "min", "lowest", "smallest"]):
        target_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        return {"operation": "minimum", "column": target_col}

    if any(term in q for term in ["maximum", "max", "highest", "largest"]):
        target_col = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0])
        return {"operation": "maximum", "column": target_col}

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
        raise ValueError("This analysis type is not supported in the MVP yet.")

    if operation == 'top_n':
        column = validate_column(df, instruction.get('column'))
        group_by = validate_column(df, instruction.get('group_by'))
        n = int(instruction.get('n', 5))
        grouped = df.groupby(group_by, as_index=False)[column].sum().sort_values(by=column, ascending=False).head(n)
        answer = f"Top {n} values for '{group_by}' by '{column}': " + ", ".join(
            [f"{row[group_by]}: {row[column]:,.2f}" for _, row in grouped.iterrows()]
        )
        return {"answer": answer, "data": grouped.to_dict(orient='records')}

    if operation == 'group_by':
        column = validate_column(df, instruction.get('column'))
        group_by = validate_column(df, instruction.get('group_by'))
        grouped = df.groupby(group_by, as_index=False)[column].sum().sort_values(by=column, ascending=False)
        answer = f"Grouped total of '{column}' by '{group_by}'."
        return {"answer": answer, "data": grouped.to_dict(orient='records')}

    if operation == 'sum':
        column = validate_column(df, instruction.get('column'))
        total = float(df[column].sum())
        return {"answer": f"The total of '{column}' is {total:,.2f}.", "data": [{"column": column, "sum": total}]}

    if operation == 'average':
        column = validate_column(df, instruction.get('column'))
        avg = float(df[column].mean())
        return {"answer": f"The average of '{column}' is {avg:,.2f}.", "data": [{"column": column, "average": avg}]}

    if operation == 'count':
        column = validate_column(df, instruction.get('column'))
        count = int(df[column].count())
        return {"answer": f"There are {count:,} non-empty values in '{column}'.", "data": [{"column": column, "count": count}]}

    if operation == 'minimum':
        column = validate_column(df, instruction.get('column'))
        min_val = float(df[column].min())
        return {"answer": f"The minimum value in '{column}' is {min_val:,.2f}.", "data": [{"column": column, "minimum": min_val}]}

    if operation == 'maximum':
        column = validate_column(df, instruction.get('column'))
        max_val = float(df[column].max())
        return {"answer": f"The maximum value in '{column}' is {max_val:,.2f}.", "data": [{"column": column, "maximum": max_val}]}

    if operation == 'filter':
        filter_column = validate_column(df, instruction.get('filter_column'))
        filter_value = instruction.get('filter_value')
        if filter_value is None:
            raise ValueError("The LLM filter instruction is missing a filter value.")

        filtered = df[df[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False)]
        if filtered.empty:
            return {"answer": f"No rows matched the filter '{filter_value}' in '{filter_column}'.", "data": []}

        answer = f"Filtered rows for '{filter_column}' = '{filter_value}' returned {len(filtered):,} rows."
        return {"answer": answer, "data": filtered.head(10).to_dict(orient='records')}

    raise ValueError(f"Unsupported operation: {operation}")


def generate_result_insight(question: str, result_data):
    if not result_data:
        return "No result data was produced for this question."

    try:
        small_payload = {
            "question": question,
            "result": result_data[:5]
        }

        for model_name in get_gemini_model():
            try:
                model = genai.GenerativeModel(model_name=model_name)
                response = model.generate_content(
                    "Write one short, human-readable sentence based only on this result data. Do not use markdown.\n\n"
                    + json.dumps(small_payload)
                )
                return response.text.strip()
            except Exception:
                continue
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
