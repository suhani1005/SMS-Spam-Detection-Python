from flask import Flask, render_template, request
from pathlib import Path
from datetime import datetime
import sqlite3
import pickle
import time

app = Flask(__name__)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"

DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "predictions.db"

DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD TRAINED MACHINE LEARNING MODEL
# ==========================================

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ==========================================
# 3. DATABASE CONNECTION
# ==========================================

def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ==========================================
# 4. INITIALIZE DATABASE
# ==========================================

def initialize_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


initialize_database()


# ==========================================
# 5. SAVE PREDICTION
# ==========================================

def save_prediction(message, prediction):
    with get_db_connection() as connection:
        connection.execute("""
            INSERT INTO prediction_history
            (message, prediction, created_at)
            VALUES (?, ?, ?)
        """, (
            message,
            prediction,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


# ==========================================
# 6. DASHBOARD STATISTICS
# ==========================================

def get_statistics():
    with get_db_connection() as connection:
        records = connection.execute("""
            SELECT prediction
            FROM prediction_history
        """).fetchall()

    total = len(records)

    spam_count = sum(
        1 for row in records
        if row["prediction"] == "Spam Message"
    )

    safe_count = sum(
        1 for row in records
        if row["prediction"] == "Not Spam"
    )

    return total, spam_count, safe_count


# ==========================================
# 7. DAA: LINEAR SEARCH
# ==========================================

def linear_search(records, search_query):
    matching_records = []
    search_query = search_query.strip().lower()

    for record in records:
        if search_query in record["message"].lower():
            matching_records.append(record)

    return matching_records


# ==========================================
# 8. DAA: MERGE SORT
# ==========================================

def merge_sort(records):
    """
    Sort SMS history alphabetically by message.
    Uses Merge Sort.
    """

    if len(records) <= 1:
        return records

    middle = len(records) // 2

    left_half = merge_sort(records[:middle])
    right_half = merge_sort(records[middle:])

    return merge(left_half, right_half)


def merge(left, right):
    """Merge two sorted lists."""

    sorted_records = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):

        left_message = left[i]["message"].lower()
        right_message = right[j]["message"].lower()

        if left_message <= right_message:
            sorted_records.append(left[i])
            i += 1
        else:
            sorted_records.append(right[j])
            j += 1

    sorted_records.extend(left[i:])
    sorted_records.extend(right[j:])

    return sorted_records


# ==========================================
# 9. DAA: EXECUTION TIME MEASUREMENT
# ==========================================

def measure_execution_time(algorithm, records, *args):
    start_time = time.perf_counter()

    algorithm(records, *args)

    end_time = time.perf_counter()

    return end_time - start_time


# ==========================================
# 10. DAA: PERFORMANCE ANALYSIS
# ==========================================

def analyze_daa_algorithms():

    with get_db_connection() as connection:
        records = connection.execute("""
            SELECT id, message, prediction, created_at
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

    if not records:
        print("\nNo SMS records available for DAA analysis.")
        print("Check some SMS messages before testing performance.")
        return

    # Use a keyword from an existing SMS
    first_message = records[0]["message"].strip()

    if not first_message:
        print("\nNo searchable SMS message is available.")
        return

    search_keyword = first_message.split()[0]

    # Measure Linear Search
    linear_time = measure_execution_time(
        linear_search,
        records,
        search_keyword
    )

    # Measure Merge Sort
    merge_time = measure_execution_time(
        merge_sort,
        records
    )

    print("\n========== DAA PERFORMANCE ANALYSIS ==========")
    print("Total SMS records:", len(records))
    print("Search keyword:", search_keyword)
    print(f"Linear Search: {linear_time:.8f} seconds")
    print(f"Merge Sort:    {merge_time:.8f} seconds")
    print("==============================================\n")


# ==========================================
# 11. HOME PAGE AND SMS PREDICTION
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    message = ""

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if message:

            prediction = model.predict([message])[0]

            if prediction == "spam":
                result = "Spam Message"
            else:
                result = "Not Spam"

            save_prediction(message, result)

        else:
            result = "Please enter a message"

    total, spam_count, safe_count = get_statistics()

    return render_template(
        "index.html",
        result=result,
        message=message,
        total=total,
        spam_count=spam_count,
        safe_count=safe_count
    )


# ==========================================
# 12. PREDICTION HISTORY AND SEARCH
# ==========================================

@app.route("/history")
def history():

    search_query = request.args.get("search", "").strip()
    sort_order = request.args.get("sort", "")

    with get_db_connection() as connection:
        records = connection.execute("""
            SELECT id, message, prediction, created_at
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

    # Apply Linear Search
    if search_query:
        records = linear_search(records, search_query)

    # Apply Merge Sort
    if sort_order == "az":
        records = merge_sort(records)

    total, spam_count, safe_count = get_statistics()

    return render_template(
        "history.html",
        records=records,
        total=total,
        spam_count=spam_count,
        safe_count=safe_count,
        search_query=search_query,
        sort_order=sort_order,
        cleared=False
    )


# ==========================================
# DAA ANALYSIS PAGE
# ==========================================

@app.route("/daa-analysis")
def daa_analysis():

    with get_db_connection() as connection:
        records = connection.execute("""
            SELECT id, message, prediction, created_at
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

    total_records = len(records)

    if records:
        search_keyword = records[0]["message"].strip().split()[0]

        linear_time = measure_execution_time(
            linear_search, records, search_keyword
        )

        merge_time = measure_execution_time(
            merge_sort, records
        )
    else:
        search_keyword = ""
        linear_time = 0
        merge_time = 0

    return render_template(
        "daa_analysis.html",
        total_records=total_records,
        search_keyword=search_keyword,
        linear_time=linear_time,
        merge_time=merge_time
    )


# ==========================================
# 13. CLEAR PREDICTION HISTORY
# ==========================================

@app.route("/clear-history", methods=["POST"])
def clear_history():

    with get_db_connection() as connection:
        connection.execute("DELETE FROM prediction_history")

    return render_template(
        "history.html",
        records=[],
        total=0,
        spam_count=0,
        safe_count=0,
        search_query="",
        sort_order="",
        cleared=True
    )


# ==========================================
# 14. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    # Measure algorithm execution time at startup
    analyze_daa_algorithms()

    # Disable the automatic reloader to avoid duplicate analysis output
    app.run(debug=True, use_reloader=False)
