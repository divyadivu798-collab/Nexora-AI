import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from datetime import datetime
from huggingface_hub import InferenceClient
import json
import os
import threading
import pandas as pd
import numpy as np
from PIL import Image,ImageDraw, ImageTk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.nexora.orchestrator import NexoraOrchestrator


# ============================================================
# NEXORA AI
# Personalized AI Life & Intelligence Platform
# ============================================================

APP_NAME = "NEXORA AI"
MEMORY_FILE = "memory.json"

orchestrator = NexoraOrchestrator()

# ------------------------------------------------------------
# MEMORY
# ------------------------------------------------------------

if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except Exception:
        memory = []
else:
    memory = []


# ------------------------------------------------------------
# HUGGING FACE AI
# ------------------------------------------------------------

client = InferenceClient()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

messages = [
    {
        "role": "system",
        "content": (
            "You are NEXORA AI, a helpful and intelligent AI assistant. "
            "Give clear, simple and useful answers. "
            "Explain programming step by step for beginners. "
            "You are part of NEXORA, a personalized AI Life and "
            "Intelligence Platform."
        )
    }
]


# ============================================================
# COLORS
# ============================================================

BG = "#090D1A"
SIDEBAR = "#10172A"
CARD = "#151D35"
CARD2 = "#1B2542"

TEXT = "#FFFFFF"
MUTED = "#9AA4BF"

ACCENT = "#00D9FF"
GREEN = "#42F5A7"
YELLOW = "#FFD166"

USER_COLOR = "#243B80"
AI_COLOR = "#151D35"


# ============================================================
# GLOBAL DATA VARIABLES
# ============================================================

current_dataframe = None
learn_progress ={}
current_file_name = ""


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(APP_NAME)
root.geometry("1200x760")
root.minsize(1000, 650)
root.configure(bg=BG)


# ============================================================
# SIDEBAR
# ============================================================

sidebar = tk.Frame(
    root,
    bg=SIDEBAR,
    width=230
)

sidebar.pack(
    side="left",
    fill="y"
)

sidebar.pack_propagate(False)


logo = tk.Label(
    sidebar,
    text="NEXORA",
    font=("Segoe UI", 23, "bold"),
    fg=ACCENT,
    bg=SIDEBAR
)

logo.pack(
    pady=(30, 4)
)


subtitle = tk.Label(
    sidebar,
    text="AI LIFE & INTELLIGENCE",
    font=("Segoe UI", 8),
    fg=MUTED,
    bg=SIDEBAR
)

subtitle.pack(
    pady=(0, 30)
)


def sidebar_button(text):
    button = tk.Button(
        sidebar,
        text=text,
        font=("Segoe UI", 11),
        fg=TEXT,
        bg=SIDEBAR,
        activebackground=CARD2,
        activeforeground=TEXT,
        relief="flat",
        anchor="w",
        padx=22,
        pady=12,
        bd=0,
        cursor="hand2"
    )

    button.pack(
        fill="x",
        padx=10,
        pady=3
    )

    return button


home_btn = sidebar_button("⌂  Home")
assistant_btn = sidebar_button("✦  AI Assistant")
tools_btn = sidebar_button("⚙  AI Tools")
projects_btn = sidebar_button("▣  Projects")


# ============================================================
# MAIN AREA
# ============================================================

main = tk.Frame(
    root,
    bg=BG
)

main.pack(
    side="left",
    fill="both",
    expand=True
)


# ============================================================
# PAGE FRAME
# ============================================================

page_frame = tk.Frame(
    main,
    bg=BG
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    main,
    bg=BG,
    height=80
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


title = tk.Label(
    header,
    text="AI Assistant",
    font=("Segoe UI", 22, "bold"),
    fg=TEXT,
    bg=BG
)

title.pack(
    side="left",
    padx=30,
    pady=20
)


status = tk.Label(
    header,
    text="● Online",
    font=("Segoe UI", 10),
    fg=ACCENT,
    bg=BG
)

status.pack(
    side="right",
    padx=30
)


# ============================================================
# CHAT AREA
# ============================================================

chat_frame = tk.Frame(
    main,
    bg=BG
)

chat_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=(5, 10)
)


chat_box = scrolledtext.ScrolledText(
    chat_frame,
    bg=BG,
    fg=TEXT,
    insertbackground=TEXT,
    font=("Segoe UI", 11),
    wrap=tk.WORD,
    relief="flat",
    bd=0,
    padx=20,
    pady=20
)

chat_box.pack(
    fill="both",
    expand=True
)


chat_box.tag_config(
    "user",
    foreground="#FFFFFF",
    background=USER_COLOR,
    lmargin1=400,
    lmargin2=400,
    rmargin=10,
    spacing3=12
)

chat_box.tag_config(
    "ai",
    foreground="#FFFFFF",
    background=AI_COLOR,
    lmargin1=10,
    lmargin2=10,
    rmargin=400,
    spacing3=12
)

chat_box.tag_config(
    "system",
    foreground=MUTED,
    justify="center"
)


def add_message(sender, text, message_type):

    chat_box.config(state="normal")

    if message_type == "user":

        chat_box.insert(
            tk.END,
            f"\nYou\n{text}\n",
            "user"
        )

    elif message_type == "ai":

        chat_box.insert(
            tk.END,
            f"\nNEXORA AI\n{text}\n",
            "ai"
        )

    else:

        chat_box.insert(
            tk.END,
            f"\n{text}\n",
            "system"
        )

    chat_box.see(tk.END)
    chat_box.config(state="disabled")


add_message(
    "NEXORA AI",
    "Hello! 👋 I'm NEXORA AI.\nHow can I help you today?",
    "ai"
)


# ============================================================
# AI RESPONSE
# ============================================================

def get_ai_response(user_text):

    global messages

    try:

        # Request routing
        route_result = orchestrator.route_request(user_text)

        intent = route_result.get("intent", "general")
        module = route_result.get("module", "assistant")

        system_context = (
            f"\nCurrent request intent: {intent}"
            f"\nSuggested NEXORA module: {module}"
        )

        messages.append(
            {
                "role": "user",
                "content": user_text + system_context
            }
        )

        memory.append(
            {
                "role": "user",
                "content": user_text,
                "time": datetime.now().isoformat()
            }
        )

        with open(
            MEMORY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                memory,
                f,
                ensure_ascii=False,
                indent=2
            )

        response = client.chat_completion(
            messages=messages,
            model=MODEL,
            max_tokens=500,
            temperature=0.7
        )

        answer = response.choices[0].message.content

        messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        memory.append(
            {
                "role": "assistant",
                "content": answer,
                "time": datetime.now().isoformat()
            }
        )

        with open(
            MEMORY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                memory,
                f,
                ensure_ascii=False,
                indent=2
            )

        return answer

    except Exception as e:

        return (
            "Sorry, I couldn't connect to the AI right now.\n\n"
            f"Error: {str(e)}"
        )


# ============================================================
# SEND MESSAGE
# ============================================================

def send_message():

    user_text = input_box.get(
        "1.0",
        tk.END
    ).strip()

    if not user_text:
        return

    input_box.delete(
        "1.0",
        tk.END
    )

    add_message(
        "You",
        user_text,
        "user"
    )

    send_button.config(
        state="disabled",
        text="Thinking..."
    )

    status.config(
        text="● Thinking...",
        fg=YELLOW
    )

    thread = threading.Thread(
        target=process_ai,
        args=(user_text,),
        daemon=True
    )

    thread.start()


def process_ai(user_text):

    answer = get_ai_response(user_text)

    root.after(
        0,
        show_ai_response,
        answer
    )


def show_ai_response(answer):

    add_message(
        "NEXORA AI",
        answer,
        "ai"
    )

    send_button.config(
        state="normal",
        text="Send"
    )

    status.config(
        text="● Online",
        fg=ACCENT
    )

    input_box.focus()


# ============================================================
# INPUT AREA
# ============================================================

input_frame = tk.Frame(
    main,
    bg=CARD,
    height=90
)

input_frame.pack(
    fill="x",
    padx=25,
    pady=(0, 25)
)

input_frame.pack_propagate(False)


input_box = tk.Text(
    input_frame,
    bg=CARD,
    fg=TEXT,
    insertbackground=TEXT,
    font=("Segoe UI", 11),
    relief="flat",
    bd=0,
    wrap=tk.WORD
)

input_box.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(15, 5),
    pady=15
)


send_button = tk.Button(
    input_frame,
    text="Send",
    command=send_message,
    font=("Segoe UI", 11, "bold"),
    bg=ACCENT,
    fg="#07101F",
    activebackground=ACCENT,
    activeforeground="#07101F",
    relief="flat",
    bd=0,
    padx=25,
    cursor="hand2"
)

send_button.pack(
    side="right",
    padx=15,
    pady=20
)


def enter_pressed(event):

    if event.state & 0x0001:
        return

    send_message()

    return "break"


input_box.bind(
    "<Return>",
    enter_pressed
)


# ============================================================
# COMMON PAGE FUNCTIONS
# ============================================================

def hide_chat():

    header.pack_forget()
    chat_frame.pack_forget()
    input_frame.pack_forget()


def clear_page():

    for widget in page_frame.winfo_children():
        widget.destroy()


def create_page_title(
    parent,
    title_text,
    subtitle_text
):

    tk.Label(
        parent,
        text=title_text,
        font=("Segoe UI", 26, "bold"),
        fg=TEXT,
        bg=BG
    ).pack(
        anchor="w",
        padx=35,
        pady=(30, 5)
    )

    tk.Label(
        parent,
        text=subtitle_text,
        font=("Segoe UI", 11),
        fg=MUTED,
        bg=BG
    ).pack(
        anchor="w",
        padx=35,
        pady=(0, 25)
    )


def show_page():

    page_frame.pack(
        fill="both",
        expand=True
    )


# ============================================================
# HOME PAGE
# ============================================================

def show_home():

    hide_chat()
    clear_page()
    show_page()

    create_page_title(
        page_frame,
        "Welcome to NEXORA AI",
        "Your personalized AI Life & Intelligence Platform."
    )

    welcome = tk.Frame(
        page_frame,
        bg=CARD,
        padx=25,
        pady=25
    )

    welcome.pack(
        fill="x",
        padx=35,
        pady=10
    )

    tk.Label(
        welcome,
        text="NEXORA AI",
        font=("Segoe UI", 22, "bold"),
        fg=ACCENT,
        bg=CARD
    ).pack(anchor="w")

    tk.Label(
        welcome,
        text=(
            "One intelligent workspace to Learn, Work, Create, "
            "Analyze, Predict, Connect and Explore."
        ),
        font=("Segoe UI", 12),
        fg=TEXT,
        bg=CARD,
        wraplength=850,
        justify="left"
    ).pack(
        anchor="w",
        pady=(10, 0)
    )

    stats = tk.Frame(
        page_frame,
        bg=BG
    )

    stats.pack(
        fill="x",
        padx=35,
        pady=20
    )

    cards = [
        ("✦", "AI Assistant", "Conversational AI"),
        ("📊", "Data Intelligence", "Analyze datasets"),
        ("🔮", "Prediction Lab", "Machine learning"),
        ("🎯", "Personalization", "User-focused AI")
    ]

    for icon, name, desc in cards:

        card = tk.Frame(
            stats,
            bg=CARD2,
            padx=20,
            pady=20
        )

        card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5
        )

        tk.Label(
            card,
            text=icon,
            font=("Segoe UI", 22),
            fg=ACCENT,
            bg=CARD2
        ).pack(anchor="w")

        tk.Label(
            card,
            text=name,
            font=("Segoe UI", 12, "bold"),
            fg=TEXT,
            bg=CARD2
        ).pack(
            anchor="w",
            pady=(8, 3)
        )

        tk.Label(
            card,
            text=desc,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=CARD2
        ).pack(anchor="w")


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

def show_assistant():

    page_frame.pack_forget()

    header.pack(
        fill="x"
    )

    chat_frame.pack(
        fill="both",
        expand=True,
        padx=25,
        pady=(5, 10)
    )

    input_frame.pack(
        fill="x",
        padx=25,
        pady=(0, 25)
    )

    title.config(
        text="AI Assistant"
    )

    status.config(
        text="● Online",
        fg=ACCENT
    )

    input_box.focus()


# ============================================================
# AI TOOLS PAGE
# ============================================================
 
def show_tools():

    hide_chat()
    clear_page()
    show_page()

    create_page_title(
        page_frame,
        "AI Tools",
        "Intelligent tools inside the NEXORA ecosystem."
    )

    tools = [
        (
            "📊",
            "Data Intelligence",
            "Upload CSV or Excel files, clean data, analyze statistics and create charts.",
            show_data_intelligence
        ),
        (
            "🔮",
            "Prediction Lab",
            "Use machine learning for prediction and evaluation.",
            show_prediction
        ),
        (
            "📚",
            "NEXORA Learn",
            "Personalized learning, explanations, quizzes and skill development.",
            show_learn
        ),
        (
            "💼",
            "NEXORA Career",
            "Career paths, skill-gap analysis, resume and interview assistance.",
            show_career
        ),
        (
            "🎨",
            "Creative Studio",
            "Content ideas, scripts, captions and creative workflows.",
            show_creative
        ),
        (
            "🛍",
            "Smart Shopping",
            "AI-assisted product discovery and recommendations.",
            show_shopping
        )
    ]

    container = tk.Frame(
        page_frame,
        bg=BG
    )

    container.pack(
        fill="both",
        expand=True,
        padx=35
    )

    for i, (icon, name, desc, command) in enumerate(tools):

        row = i // 2
        col = i % 2

        card = tk.Frame(
            container,
            bg=CARD,
            padx=20,
            pady=20,
            cursor="hand2"
        )

        card.grid(
            row=row,
            column=col,
            sticky="nsew",
            padx=8,
            pady=8
        )

        button = tk.Button(
            card,
            text=f"{icon}  {name}",
            font=("Segoe UI", 14, "bold"),
            fg=ACCENT,
            bg=CARD,
            activeforeground=ACCENT,
            activebackground=CARD2,
            relief="flat",
            bd=0,
            cursor="hand2",
            anchor="w",
            command=command
        )

        button.pack(
            fill="x",
            anchor="w"
        )

        description = tk.Label(
            card,
            text=desc,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=CARD,
            wraplength=350,
            justify="left",
            cursor="hand2"
        )

        description.pack(
            fill="x",
            anchor="w",
            pady=(8, 0)
        )

        # Make the whole card clickable
        card.bind("<Button-1>", lambda event, cmd=command: cmd())
        description.bind("<Button-1>", lambda event, cmd=command: cmd())

    container.grid_columnconfigure(0, weight=1)
    container.grid_columnconfigure(1, weight=1)

# ============================================================
# NEXORA DATA INTELLIGENCE ENGINE
# ============================================================

current_dataframe = None
current_file_name = ""

rows_label = None
cols_label = None
missing_label = None
numeric_label = None
file_label = None
output = None


def write_output(title, text):
    if output is None:
        return

    output.insert(
        tk.END,
        "\n\n" + title + "\n"
    )

    output.insert(
        tk.END,
        "=" * 60 + "\n\n"
    )

    output.insert(
        tk.END,
        str(text)
    )


def load_dataset():

    global current_dataframe
    global current_file_name

    file_path = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:

        if file_path.lower().endswith(".csv"):

            try:
                df = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python"
                )
            except Exception:
                df = pd.read_csv(
                    file_path
                )

        else:

            df = pd.read_excel(
                file_path
            )

        if df.empty:

            messagebox.showwarning(
                "Empty Dataset",
                "The selected file does not contain data."
            )

            return

        current_dataframe = df
        current_file_name = os.path.basename(
            file_path
        )

        update_data_dashboard()

        analyze_dataset()

        messagebox.showinfo(
            "Dataset Loaded",
            f"{current_file_name}\n\n"
            f"{len(df)} rows loaded successfully."
        )

    except Exception as e:

        messagebox.showerror(
            "Dataset Error",
            f"Could not load dataset.\n\n{e}"
        )


def update_data_dashboard():

    if current_dataframe is None:
        return

    df = current_dataframe

    if file_label is not None:

        file_label.config(
            text=f"Loaded: {current_file_name}"
        )

    if rows_label is not None:

        rows_label.config(
            text=f"Rows\n{df.shape[0]}"
        )

    if cols_label is not None:

        cols_label.config(
            text=f"Columns\n{df.shape[1]}"
        )

    if missing_label is not None:

        missing_label.config(
            text=(
                f"Missing Values\n"
                f"{int(df.isnull().sum().sum())}"
            )
        )

    if numeric_label is not None:

        numeric_count = len(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        numeric_label.config(
            text=f"Numeric Columns\n{numeric_count}"
        )


def analyze_dataset():

    if current_dataframe is None:
        return

    df = current_dataframe

    if output is None:
        return

    output.delete(
        "1.0",
        tk.END
    )

    write_output(
        "NEXORA DATA INTELLIGENCE REPORT",
        f"Dataset: {current_file_name}\n"
        f"Rows: {len(df)}\n"
        f"Columns: {len(df.columns)}"
    )
def analyze_dataset():

    if current_dataframe is None:
        return

    df = current_dataframe

    if output is None:
        return

    output.delete(
        "1.0",
        tk.END
    )

    write_output(
        "NEXORA DATA INTELLIGENCE REPORT",
        f"Dataset: {current_file_name}\n"
        f"Rows: {len(df)}\n"
        f"Columns: {len(df.columns)}"
    )

    missing = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            exclude=np.number
        ).columns
    )

    write_output(
        "DATA QUALITY",
        f"Missing Values: {missing}\n"
        f"Duplicate Rows: {duplicates}\n"
        f"Numeric Columns: {len(numeric_columns)}\n"
        f"Categorical Columns: {len(categorical_columns)}"
    )

    if numeric_columns:

        summary = (
            df[numeric_columns]
            .describe()
            .round(2)
            .transpose()
        )

        write_output(
            "STATISTICAL SUMMARY",
            summary
        )

    # Automatic business-column detection

    keywords = [
        "sales",
        "sale",
        "revenue",
        "profit",
        "amount",
        "price",
        "quantity",
        "units",
        "income",
        "value"
    ]

    business_columns = []

    for column in df.columns:

        name = str(
            column
        ).lower()

        if any(
            word in name
            for word in keywords
        ):

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                business_columns.append(
                    column
                )

    if business_columns:

        report = []

        for column in business_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            total = values.sum()
            average = values.mean()
            highest = values.max()
            lowest = values.min()

            report.append(
                f"{column}\n"
                f"Total: {total:,.2f}\n"
                f"Average: {average:,.2f}\n"
                f"Highest: {highest:,.2f}\n"
                f"Lowest: {lowest:,.2f}"
            )

        write_output(
            "BUSINESS METRICS",
            "\n\n".join(report)
        )
        # --------------------------------------------------------
    # AUTOMATIC TREND ANALYSIS
    # --------------------------------------------------------

    trend_report = []

    for column in business_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) < 2:
            continue

        first = float(values.iloc[0])
        last = float(values.iloc[-1])

        if first != 0:

            change = (
                (last - first)
                / abs(first)
            ) * 100

        else:
            change = 0

        if change > 0:

            trend_report.append(
                f"📈 {column}: "
                f"increased by "
                f"{change:.2f}%"
            )

        elif change < 0:

            trend_report.append(
                f"📉 {column}: "
                f"decreased by "
                f"{abs(change):.2f}%"
            )

        else:

            trend_report.append(
                f"➡️ {column}: "
                f"no major change"
            )

    if trend_report:

        write_output(
            "AUTOMATIC TREND ANALYSIS",
            "\n".join(trend_report)
        )

    # --------------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------------

    category_report = []

    for column in categorical_columns:

        unique_values = (
            df[column]
            .dropna()
            .nunique()
        )

        if unique_values <= 30:

            top_items = (
                df[column]
                .value_counts()
                .head(10)
            )

            category_report.append(
                f"{column}\n"
                f"Unique Values: {unique_values}\n"
                f"{top_items.to_string()}"
            )

    if category_report:

        write_output(
            "TOP CATEGORIES / VALUES",
            "\n\n".join(
                category_report
            )
        )

    # --------------------------------------------------------
    # TOP AND LOWEST BUSINESS VALUES
    # --------------------------------------------------------

    ranking_report = []

    for column in business_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if values.notna().sum() == 0:
            continue

        highest = values.idxmax()
        lowest = values.idxmin()

        ranking_report.append(
            f"{column}\n"
            f"Highest Row: {highest}\n"
            f"Highest Value: "
            f"{values.loc[highest]:,.2f}\n"
            f"Lowest Row: {lowest}\n"
            f"Lowest Value: "
            f"{values.loc[lowest]:,.2f}"
        )

    if ranking_report:

        write_output(
            "HIGHEST / LOWEST PERFORMANCE",
            "\n\n".join(
                ranking_report
            )
        )

    write_output(
        "NEXORA INSIGHT",
        "Dataset analysis completed.\n\n"
        "NEXORA automatically detected "
        "business metrics, calculated "
        "statistics, analyzed trends and "
        "identified high/low values."
    )
 # ============================================================
# CLEAN DATA
# ============================================================

def clean_dataset():

    global current_dataframe

    if current_dataframe is None:

        messagebox.showwarning(
            "No Dataset",
            "Please upload a dataset first."
        )

        return

    df = current_dataframe.copy()

    before_rows = len(df)

    # Remove duplicate rows
    df = df.drop_duplicates()

    removed_duplicates = (
        before_rows - len(df)
    )

    # Numeric columns
    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    # Categorical columns
    categorical_columns = list(
        df.select_dtypes(
            exclude=np.number
        ).columns
    )

    # Fill numeric missing values
    numeric_filled = 0

    for column in numeric_columns:

        missing_before = int(
            df[column].isnull().sum()
        )

        if missing_before > 0:

            median_value = (
                df[column].median()
            )

            df[column] = (
                df[column]
                .fillna(median_value)
            )

            numeric_filled += (
                missing_before
            )

    # Fill categorical missing values
    categorical_filled = 0

    for column in categorical_columns:

        missing_before = int(
            df[column].isnull().sum()
        )

        if missing_before > 0:

            mode_values = (
                df[column].mode()
            )

            if not mode_values.empty:

                df[column] = (
                    df[column]
                    .fillna(mode_values.iloc[0])
                )

                categorical_filled += (
                    missing_before
                )

    current_dataframe = df

    update_data_dashboard()

    if output is not None:

        output.delete(
            "1.0",
            tk.END
        )

        write_output(
            "DATA CLEANING COMPLETED",
            "✓ Duplicate rows removed\n"
            f"✓ Duplicates removed: "
            f"{removed_duplicates}\n"
            "✓ Numeric missing values "
            "filled using median\n"
            f"✓ Numeric values filled: "
            f"{numeric_filled}\n"
            "✓ Categorical missing values "
            "filled using mode\n"
            f"✓ Categorical values filled: "
            f"{categorical_filled}\n"
            f"✓ Final rows: {len(df)}\n"
            f"✓ Final columns: "
            f"{len(df.columns)}"
        )

    messagebox.showinfo(
        "Data Cleaning",
        "Dataset cleaned successfully!"
    )


def create_data_chart():

    if current_dataframe is None:

        messagebox.showwarning(
            "No Dataset",
            "Please upload a dataset first."
        )

        return

    df = current_dataframe

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    if not numeric_columns:

        messagebox.showwarning(
            "Chart",
            "No numeric columns found."
        )

        return

    try:

        column = numeric_columns[0]

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            df[column].values
        )

        plt.title(
            f"NEXORA Trend Chart - {column}"
        )

        plt.xlabel(
            "Record Number"
        )

        plt.ylabel(
            column
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()

    except Exception as e:

        messagebox.showerror(
            "Chart Error",
            str(e)
        )
 # ============================================================
# AUTO BUSINESS DASHBOARD - FIXED
# ============================================================

def create_business_dashboard():

    try:

        if current_dataframe is None:

            messagebox.showwarning(
                "No Dataset",
                "Please upload a dataset first."
            )

            return

        df = current_dataframe.copy()

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        if not numeric_columns:

            messagebox.showwarning(
                "Dashboard",
                "No numeric columns found."
            )

            return

        keywords = [
            "sales",
            "sale",
            "revenue",
            "profit",
            "amount",
            "price",
            "quantity",
            "units"
        ]

        selected_column = None

        for column in numeric_columns:

            name = str(
                column
            ).lower()

            if any(
                word in name
                for word in keywords
            ):

                selected_column = column
                break

        if selected_column is None:

            selected_column = numeric_columns[0]

        values = pd.to_numeric(
            df[selected_column],
            errors="coerce"
        ).dropna()

        if values.empty:

            messagebox.showwarning(
                "Dashboard",
                "No usable numeric data found."
            )

            return

        total = values.sum()
        average = values.mean()
        highest = values.max()
        lowest = values.min()

        window = tk.Toplevel(
            root
        )

        window.title(
            "NEXORA AI - Auto Dashboard"
        )

        window.geometry(
            "1100x700"
        )

        window.configure(
            bg=BG
        )

        window.transient(
            root
        )

        window.lift()
        window.focus_force()

        tk.Label(
            window,
            text="NEXORA AI BUSINESS DASHBOARD",
            font=("Segoe UI", 20, "bold"),
            fg=ACCENT,
            bg=BG
        ).pack(
            pady=(20, 5)
        )

        tk.Label(
            window,
            text=f"Automatic analysis: {selected_column}",
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=BG
        ).pack(
            pady=(0, 15)
        )

        cards = tk.Frame(
            window,
            bg=BG
        )

        cards.pack(
            fill="x",
            padx=25
        )

        metrics = [
            ("TOTAL", total),
            ("AVERAGE", average),
            ("HIGHEST", highest),
            ("LOWEST", lowest)
        ]

        for name, value in metrics:

            card = tk.Frame(
                cards,
                bg=CARD2,
                padx=15,
                pady=12
            )

            card.pack(
                side="left",
                fill="both",
                expand=True,
                padx=5
            )

            tk.Label(
                card,
                text=name,
                font=("Segoe UI", 9, "bold"),
                fg=MUTED,
                bg=CARD2
            ).pack()

            tk.Label(
                card,
                text=f"{value:,.2f}",
                font=("Segoe UI", 14, "bold"),
                fg=TEXT,
                bg=CARD2
            ).pack(
                pady=5
            )

        figure = plt.Figure(
            figsize=(9, 4),
            dpi=100
        )

        axis = figure.add_subplot(
            111
        )

        axis.plot(
            values.values,
            marker="o",
            markersize=3
        )

        axis.set_title(
            f"{selected_column} Trend"
        )

        axis.set_xlabel(
            "Record Number"
        )

        axis.set_ylabel(
            selected_column
        )

        axis.grid(
            True,
            alpha=0.3
        )

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(
            figure,
            master=window
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=25,
            pady=20
        )

    except Exception as e:

        messagebox.showerror(
            "Auto Dashboard Error",
            f"Dashboard could not be created.\n\n{e}"
        )
 # ============================================================
# DATA INTELLIGENCE PAGE
# ============================================================

def show_data_intelligence():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=20
    )

    create_page_title(
        page_frame,
        "Data Intelligence Lab",
        "Upload → Clean → Analyze → Visualize"
    )

    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------

    button_frame = tk.Frame(
        page_frame,
        bg=BG
    )

    button_frame.pack(
        fill="x",
        padx=35,
        pady=10
    )

    tk.Button(
        button_frame,
        text="📂 Upload CSV / Excel",
        command=load_dataset,
        font=("Segoe UI", 10, "bold"),
        bg=ACCENT,
        fg="#07101F",
        relief="flat",
        padx=15,
        pady=9,
        cursor="hand2"
    ).pack(
        side="left",
        padx=4
    )

    tk.Button(
        button_frame,
        text="🧹 Clean Data",
        command=clean_dataset,
        font=("Segoe UI", 10, "bold"),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        padx=15,
        pady=9,
        cursor="hand2"
    ).pack(
        side="left",
        padx=4
    )

    tk.Button(
        button_frame,
        text="📊 Create Chart",
        command=create_data_chart,
        font=("Segoe UI", 10, "bold"),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        padx=15,
        pady=9,
        cursor="hand2"
    ).pack(
        side="left",
        padx=4
    )

    tk.Button(
        button_frame,
        text="🚀 Auto Dashboard",
        command=create_business_dashboard,
        font=("Segoe UI", 10, "bold"),
        bg=CARD2,
        fg=ACCENT,
        relief="flat",
        padx=15,
        pady=9,
        cursor="hand2"
    ).pack(
        side="left",
        padx=4
    )

    # --------------------------------------------------------
    # FILE STATUS
    # --------------------------------------------------------

    global file_label

    file_label = tk.Label(
        page_frame,
        text="No dataset loaded",
        font=("Segoe UI", 10),
        fg=MUTED,
        bg=BG
    )

    file_label.pack(
        anchor="w",
        padx=35,
        pady=(5, 10)
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    dashboard = tk.Frame(
        page_frame,
        bg=BG
    )

    dashboard.pack(
        fill="x",
        padx=35,
        pady=5
    )

    global rows_label
    global cols_label
    global missing_label
    global numeric_label

    rows_label = tk.Label(
        dashboard,
        text="Rows\n0",
        font=("Segoe UI", 11, "bold"),
        fg=TEXT,
        bg=CARD2,
        padx=20,
        pady=12
    )

    rows_label.pack(
        side="left",
        fill="both",
        expand=True,
        padx=4
    )

    cols_label = tk.Label(
        dashboard,
        text="Columns\n0",
        font=("Segoe UI", 11, "bold"),
        fg=TEXT,
        bg=CARD2,
        padx=20,
        pady=12
    )

    cols_label.pack(
        side="left",
        fill="both",
        expand=True,
        padx=4
    )

    missing_label = tk.Label(
        dashboard,
        text="Missing Values\n0",
        font=("Segoe UI", 11, "bold"),
        fg=TEXT,
        bg=CARD2,
        padx=20,
        pady=12
    )

    missing_label.pack(
        side="left",
        fill="both",
        expand=True,
        padx=4
    )

    numeric_label = tk.Label(
        dashboard,
        text="Numeric Columns\n0",
        font=("Segoe UI", 11, "bold"),
        fg=TEXT,
        bg=CARD2,
        padx=20,
        pady=12
    )

    numeric_label.pack(
        side="left",
        fill="both",
        expand=True,
        padx=4
    )

    # --------------------------------------------------------
    # OUTPUT AREA
    # --------------------------------------------------------

    global output

    output = scrolledtext.ScrolledText(
        page_frame,
        bg=CARD,
        fg=TEXT,
        insertbackground=TEXT,
        font=("Consolas", 9),
        relief="flat",
        wrap=tk.NONE
    )

    output.pack(
        fill="both",
        expand=True,
        padx=35,
        pady=15
    )

    output.insert(
        tk.END,
        "NEXORA DATA INTELLIGENCE LAB\n"
        "==============================\n\n"
        "Upload a CSV or Excel file to begin.\n\n"
        "Workflow:\n"
        "Upload → Clean → Analyze → "
        "Visualize → Dashboard\n"
    )             

# ============================================================
# PREDICTION LAB
# ============================================================

def run_prediction():

    if current_dataframe is None:

        messagebox.showwarning(
            "No Dataset",
            "Please upload a dataset first."
        )

        return

    df = current_dataframe.copy()

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_columns) < 2:

        messagebox.showwarning(
            "Prediction",
            "At least two numeric columns are required."
        )

        return

    target = prediction_target.get()

    if target not in numeric_columns:

        messagebox.showwarning(
            "Prediction",
            "Please select a valid target column."
        )

        return

    feature_columns = [
        column
        for column in numeric_columns
        if column != target
    ]

    data = df[
        feature_columns + [target]
    ].dropna()

    if len(data) < 10:

        messagebox.showwarning(
            "Prediction",
            "Not enough clean rows for training."
        )

        return

    X = data[
        feature_columns
    ]

    y = data[
        target
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    prediction_output.delete(
        "1.0",
        tk.END
    )

    prediction_output.insert(
        tk.END,
        "NEXORA PREDICTION LAB\n"
        "==============================\n\n"
    )

    prediction_output.insert(
        tk.END,
        f"Target Column : {target}\n"
    )

    prediction_output.insert(
        tk.END,
        f"Features      : {', '.join(feature_columns)}\n\n"
    )

    prediction_output.insert(
        tk.END,
        "MODEL\n"
        "------------------------------\n"
        "Linear Regression\n\n"
    )

    prediction_output.insert(
        tk.END,
        "EVALUATION METRICS\n"
        "------------------------------\n"
    )

    prediction_output.insert(
        tk.END,
        f"MAE  : {mae:.4f}\n"
    )

    prediction_output.insert(
        tk.END,
        f"MSE  : {mse:.4f}\n"
    )

    prediction_output.insert(
        tk.END,
        f"RMSE : {rmse:.4f}\n"
    )

    prediction_output.insert(
        tk.END,
        f"R²   : {r2:.4f}\n\n"
    )

    prediction_output.insert(
        tk.END,
        "Note: Predictions are estimates based on the supplied "
        "dataset and are not guaranteed future outcomes."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

def show_prediction():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=20
    )

    create_page_title(
        page_frame,
        "Prediction Lab",
        "Train and evaluate a basic machine-learning model."
    )

    controls = tk.Frame(
        page_frame,
        bg=CARD,
        padx=20,
        pady=20
    )

    controls.pack(
        fill="x",
        padx=35,
        pady=10
    )

    tk.Label(
        controls,
        text="Target Column:",
        font=("Segoe UI", 10, "bold"),
        fg=TEXT,
        bg=CARD
    ).pack(
        side="left"
    )

    global prediction_target

    prediction_target = tk.StringVar()

    target_menu = tk.OptionMenu(
        controls,
        prediction_target,
        ""
    )

    target_menu.config(
        bg=CARD2,
        fg=TEXT,
        activebackground=CARD2,
        activeforeground=TEXT,
        relief="flat"
    )

    target_menu["menu"].config(
        bg=CARD2,
        fg=TEXT
    )

    target_menu.pack(
        side="left",
        padx=10
    )

    def refresh_targets():

        menu = target_menu["menu"]

        menu.delete(
            0,
            "end"
        )

        if current_dataframe is not None:

            columns = list(
                current_dataframe.select_dtypes(
                    include=np.number
                ).columns
            )

            for column in columns:

                menu.add_command(
                    label=column,
                    command=lambda c=column:
                    prediction_target.set(c)
                )

            if columns:

                prediction_target.set(
                    columns[-1]
                )

    tk.Button(
        controls,
        text="Refresh",
        command=refresh_targets,
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        padx=12,
        pady=8
    ).pack(
        side="left",
        padx=5
    )

    tk.Button(
        controls,
        text="Run ML Prediction",
        command=run_prediction,
        bg=ACCENT,
        fg="#07101F",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        padx=15,
        pady=8
    ).pack(
        side="left",
        padx=5
    )

    global prediction_output

    prediction_output = scrolledtext.ScrolledText(
        page_frame,
        bg=CARD,
        fg=TEXT,
        insertbackground=TEXT,
        font=("Consolas", 10),
        relief="flat",
        wrap=tk.WORD
    )

    prediction_output.pack(
        fill="both",
        expand=True,
        padx=35,
        pady=10
    )

    prediction_output.insert(
        tk.END,
        "Upload a dataset and select a target column."
    )

    refresh_targets()


# ==========================================
# NEXORA LEARN
# ==========================================
# ============================================================
# NEXORA LEARN - AI COURSE DASHBOARD
# ============================================================

def show_learn():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=15
    )

    create_page_title(
        page_frame,
        "NEXORA Learn",
        "Your interactive learning ecosystem"
    )

    # Scrollable area
    outer = tk.Frame(page_frame, bg=BG)
    outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(
        outer,
        bg=BG,
        highlightthickness=0
    )
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(
        outer,
        orient="vertical",
        command=canvas.yview
    )
    scrollbar.pack(side="right", fill="y")

    canvas.configure(
        yscrollcommand=scrollbar.set
    )

    content = tk.Frame(
        canvas,
        bg=BG
    )

    canvas_window = canvas.create_window(
        (0, 0),
        window=content,
        anchor="nw"
    )

    content.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfig(
            canvas_window,
            width=e.width
        )
    )

    tk.Label(
        content,
        text="CHOOSE YOUR LEARNING TRACK",
        font=("Segoe UI", 18, "bold"),
        fg=ACCENT,
        bg=BG
    ).pack(
        anchor="w",
        padx=25,
        pady=(5, 3)
    )

    tk.Label(
        content,
        text="Learn from fundamentals to advanced level — visual concepts, coding, practice and quizzes.",
        font=("Segoe UI", 10),
        fg=MUTED,
        bg=BG
    ).pack(
        anchor="w",
        padx=25,
        pady=(0, 15)
    )

    tracks = [
        ("🧠", "Artificial Intelligence",
         "AI fundamentals, agents, search, reasoning and applications."),

        ("🤖", "Machine Learning",
         "Regression, classification, clustering and ML projects."),

        ("🐍", "Python Programming",
         "Python fundamentals, OOP, libraries and projects."),

        ("💻", "Data Structures & Algorithms",
         "Algorithms, complexity, problem solving and coding practice."),

        ("📊", "Data Analytics",
         "Excel, SQL, Python, statistics and dashboards."),

        ("✨", "Generative AI",
         "LLMs, prompting, RAG, agents and GenAI projects."),

        ("🧬", "Deep Learning",
         "Neural networks, CNN, RNN, transformers and projects."),

        ("🚀", "AI Projects",
         "End-to-end practical AI projects.")
    ]

    grid = tk.Frame(
        content,
        bg=BG
    )
    grid.pack(
        fill="both",
        expand=True,
        padx=18
    )

    for i, (icon, title, description) in enumerate(tracks):

        row = i // 2
        col = i % 2

        card = tk.Frame(
            grid,
            bg=CARD,
            padx=20,
            pady=18,
            cursor="hand2"
        )

        card.grid(
            row=row,
            column=col,
            sticky="nsew",
            padx=7,
            pady=7
        )

        tk.Button(
            card,
            text=f"{icon}  {title}",
            font=("Segoe UI", 13, "bold"),
            fg=ACCENT,
            bg=CARD,
            activebackground=CARD2,
            activeforeground=ACCENT,
            relief="flat",
            bd=0,
            cursor="hand2",
            anchor="w",
            command=lambda t=title: open_learning_track(t)
        ).pack(
            fill="x"
        )

        tk.Label(
            card,
            text=description,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=CARD,
            wraplength=350,
            justify="left"
        ).pack(
            anchor="w",
            pady=(8, 0)
        )

    grid.grid_columnconfigure(0, weight=1)
    grid.grid_columnconfigure(1, weight=1)

    # Mouse wheel scrolling
    canvas.bind_all(
        "<MouseWheel>",
        lambda event: canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )
    )

    # Navigation buttons
    nav = tk.Frame(
        page_frame,
        bg=BG
    )
    nav.pack(
        fill="x",
        pady=(5, 0)
    )

    tk.Button(
        nav,
        text="▲ UP",
        command=lambda: canvas.yview_scroll(-4, "units"),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=15
    ).pack(
        side="left",
        padx=5
    )

    tk.Button(
        nav,
        text="▼ DOWN",
        command=lambda: canvas.yview_scroll(4, "units"),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=15
    ).pack(
        side="left",
        padx=5
    )
def show_learn():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=25,
        pady=20
    )

    create_page_title(
        page_frame,
        "NEXORA Learn",
        "Learn from Basic → Intermediate → Advanced"
    )

    courses = [
        ("🤖", "Artificial Intelligence",
         "AI fundamentals, agents, search, reasoning and applications."),

        ("🧠", "Machine Learning",
         "Regression, classification, clustering and model building."),

        ("🐍", "Python Programming",
         "Python fundamentals, OOP, libraries and projects."),

        ("🧩", "Data Structures & Algorithms",
         "Algorithms, complexity, problem solving and coding."),

        ("📊", "Data Analytics",
         "Excel, SQL, Python, statistics and dashboards."),

        ("✨", "Generative AI",
         "LLMs, prompting, RAG, agents and GenAI projects."),

        ("🧠", "Deep Learning",
         "Neural networks, CNN, RNN, transformers and projects."),

        ("🚀", "AI Projects",
         "End-to-end practical AI projects.")
    ]

    container = tk.Frame(
        page_frame,
        bg=BG
    )

    container.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=10
    )

    for i, (icon, title, description) in enumerate(courses):

        row = i // 2
        col = i % 2

        card = tk.Frame(
            container,
            bg=CARD,
            padx=15,
            pady=15
        )

        card.grid(
            row=row,
            column=col,
            sticky="nsew",
            padx=8,
            pady=8
        )

        button = tk.Button(
            card,
            text=f"{icon}  {title}",
            font=("Segoe UI", 14, "bold"),
            fg=ACCENT,
            bg=CARD,
            activeforeground=ACCENT,
            activebackground=CARD2,
            relief="flat",
            bd=0,
            cursor="hand2",
            anchor="w",
            command=lambda t=title: open_learning_track(t)
        )

        button.pack(
            fill="x",
            anchor="w"
        )

        desc = tk.Label(
            card,
            text=description,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=CARD,
            wraplength=350,
            justify="left",
            cursor="hand2"
        )

        desc.pack(
            fill="x",
            anchor="w",
            pady=(8, 0)
        )

        desc.bind(
            "<Button-1>",
            lambda event, t=title: open_learning_track(t)
        )

    container.grid_columnconfigure(0, weight=1)
    container.grid_columnconfigure(1, weight=1)   

def open_learning_track(track_name):

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=25,
        pady=20
    )

    create_page_title(
        page_frame,
        track_name,
        "Choose your learning level"
    )

    tk.Button(
        page_frame,
        text="← Back to NEXORA Learn",
        command=show_learn,
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=15,
        pady=8
    ).pack(
        anchor="w",
        padx=25,
        pady=10
    )

    tk.Label(
        page_frame,
        text=f"📚 {track_name}",
        font=("Segoe UI", 20, "bold"),
        fg=ACCENT,
        bg=BG
    ).pack(
        anchor="w",
        padx=25,
        pady=(15, 5)
    )

    tk.Label(
        page_frame,
        text="Learn step by step: Basic → Intermediate → Advanced",
        font=("Segoe UI", 10),
        fg=MUTED,
        bg=BG
    ).pack(
        anchor="w",
        padx=25,
        pady=(0, 20)
    )

    levels = [
        (
            "🌱",
            "BASIC",
            "Foundation",
            "Learn the core concepts with simple explanations and visuals."
        ),
        (
            "⚡",
            "INTERMEDIATE",
            "Application",
            "Practice concepts, algorithms, coding and real-world examples."
        ),
        (
            "🚀",
            "ADVANCED",
            "Mastery",
            "Build advanced systems, projects and industry-level skills."
        )
    ]

    for icon, level, subtitle, description in levels:

        card = tk.Frame(
            page_frame,
            bg=CARD,
            padx=25,
            pady=20
        )

        card.pack(
            fill="x",
            padx=25,
            pady=8
        )

        tk.Label(
            card,
            text=f"{icon}  {level} • {subtitle}",
            font=("Segoe UI", 15, "bold"),
            fg=ACCENT,
            bg=CARD
        ).pack(
            anchor="w"
        )

        tk.Label(
            card,
            text=description,
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=CARD,
            wraplength=750,
            justify="left"
        ).pack(
            anchor="w",
            pady=(7, 12)
        )

        tk.Button(
            card,
            text=f"▶ START {level}",
            command=lambda l=level:
                open_course_level(track_name, l),
            bg=CARD2,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground="#07101F",
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=8
        ).pack(
            anchor="w"
        )    
# ============================================================
# NEXORA INTERACTIVE LEARNING CHAPTER
# ============================================================

def open_course_level(track_name, level):

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=15
    )

    create_page_title(
        page_frame,
        f"{track_name} • {level}",
        "Choose a chapter • Learn step by step"
    )

    # ---------------- TOP BAR ----------------

    top = tk.Frame(
        page_frame,
        bg=BG
    )
    top.pack(
        fill="x",
        padx=25,
        pady=8
    )

    tk.Button(
        top,
        text="← BACK TO LEVELS",
        command=lambda: open_learning_track(track_name),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=15,
        pady=8
    ).pack(side="left")

    tk.Label(
        top,
        text=f"{level} • {track_name}",
        font=("Segoe UI", 13, "bold"),
        fg=ACCENT,
        bg=BG
    ).pack(
        side="left",
        padx=20
    )

    # ---------------- PROGRESS ----------------

    modules = get_course_modules(
        track_name,
        level
    )

    total = len(modules)

    completed = 0

    for item in modules:

        title = item[1]

        if learn_progress.get(
            f"{track_name}|{level}|{title}",
            False
        ):
            completed += 1

    percentage = (
        int((completed / total) * 100)
        if total else 0
    )

    progress_frame = tk.Frame(
        page_frame,
        bg=CARD2,
        padx=18,
        pady=12
    )

    progress_frame.pack(
        fill="x",
        padx=25,
        pady=8
    )

    progress_text = tk.Label(
        progress_frame,
        text=(
            f"📈 Level Progress: "
            f"{completed}/{total} chapters "
            f"• {percentage}%"
        ),
        font=("Segoe UI", 10, "bold"),
        fg=TEXT,
        bg=CARD2
    )

    progress_text.pack(
        anchor="w"
    )

    progress_bar = tk.Canvas(
        progress_frame,
        height=8,
        bg="#202A45",
        highlightthickness=0
    )

    progress_bar.pack(
        fill="x",
        pady=(8, 0)
    )

    progress_bar.create_rectangle(
        0,
        0,
        percentage * 7,
        8,
        fill=ACCENT,
        outline=""
    )

    # ---------------- SCROLL AREA ----------------

    outer = tk.Frame(
        page_frame,
        bg=BG
    )

    outer.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=5
    )

    canvas = tk.Canvas(
        outer,
        bg=BG,
        highlightthickness=0
    )

    scrollbar = tk.Scrollbar(
        outer,
        orient="vertical",
        command=canvas.yview
    )

    canvas.configure(
        yscrollcommand=scrollbar.set
    )

    scrollbar.pack(
        side="right",
        fill="y"
    )

    canvas.pack(
        side="left",
        fill="both",
        expand=True
    )

    content = tk.Frame(
        canvas,
        bg=BG
    )

    window_id = canvas.create_window(
        (0, 0),
        window=content,
        anchor="nw"
    )

    def update_scroll(event=None):

        canvas.configure(
            scrollregion=canvas.bbox("all")
        )

    content.bind(
        "<Configure>",
        update_scroll
    )

    def resize_content(event):

        canvas.itemconfig(
            window_id,
            width=event.width
        )

    canvas.bind(
        "<Configure>",
        resize_content
    )

    # ---------------- CHAPTER CARDS ----------------

    for number, title, description in modules:

        chapter_key = (
            f"{track_name}|{level}|{title}"
        )

        done = learn_progress.get(
            chapter_key,
            False
        )

        card = tk.Frame(
            content,
            bg=CARD,
            padx=20,
            pady=18
        )

        card.pack(
            fill="x",
            padx=10,
            pady=7
        )

        # Chapter number

        tk.Label(
            card,
            text=f"{number:02d}",
            font=("Segoe UI", 18, "bold"),
            fg=ACCENT,
            bg=CARD
        ).pack(
            side="left",
            padx=(0, 18)
        )

        info = tk.Frame(
            card,
            bg=CARD
        )

        info.pack(
            side="left",
            fill="both",
            expand=True
        )

        status = (
            "✓ COMPLETED"
            if done
            else "NOT STARTED"
        )

        tk.Label(
            info,
            text=title,
            font=("Segoe UI", 13, "bold"),
            fg=TEXT,
            bg=CARD
        ).pack(
            anchor="w"
        )

        tk.Label(
            info,
            text=description,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=CARD,
            wraplength=650,
            justify="left"
        ).pack(
            anchor="w",
            pady=(5, 3)
        )

        tk.Label(
            info,
            text=status,
            font=("Segoe UI", 8, "bold"),
            fg=ACCENT if done else MUTED,
            bg=CARD
        ).pack(
            anchor="w"
        )

        tk.Button(
            card,
            text=(
                "✓ REVIEW"
                if done
                else "▶ START"
            ),
            command=lambda m=title:
                open_learning_chapter(
                    track_name,
                    level,
                    m
                ),
            bg=ACCENT if not done else CARD2,
            fg="#07101F" if not done else TEXT,
            activebackground=ACCENT,
            activeforeground="#07101F",
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            padx=18,
            pady=9
        ).pack(
            side="right",
            padx=(10, 0)
        )

    # ---------------- BOTTOM NAVIGATION ----------------

    navigation = tk.Frame(
        page_frame,
        bg=BG
    )

    navigation.pack(
        fill="x",
        padx=25,
        pady=7
    )

    tk.Button(
        navigation,
        text="▲ UP",
        command=lambda: canvas.yview_scroll(
            -5,
            "units"
        ),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=18,
        pady=7
    ).pack(
        side="left",
        padx=4
    )

    tk.Button(
        navigation,
        text="▼ DOWN",
        command=lambda: canvas.yview_scroll(
            5,
            "units"
        ),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=18,
        pady=7
    ).pack(
        side="left",
        padx=4
    )

    tk.Label(
        navigation,
        text=f"📚 {total} Chapters",
        font=("Segoe UI", 10, "bold"),
        fg=ACCENT,
        bg=BG
    ).pack(
        side="right"
    )

    # ---------------- MOUSE WHEEL ----------------

    def mouse_scroll(event):

        canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )

    canvas.bind(
        "<MouseWheel>",
        mouse_scroll
    )

    canvas.bind(
        "<Button-4>",
        lambda event: canvas.yview_scroll(-3, "units")
    )

    canvas.bind(
        "<Button-5>",
        lambda event: canvas.yview_scroll(3, "units")
    )

    canvas.yview_moveto(0)

def open_learning_chapter(track_name, level, module):

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=15
    )

    create_page_title(
        page_frame,
        module,
        f"{track_name} • {level}"
    )

    top = tk.Frame(page_frame, bg=BG)
    top.pack(fill="x", padx=20, pady=5)

    tk.Button(
        top,
        text="← Back to Modules",
        command=lambda: open_course_level(track_name, level),
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=15,
        pady=8
    ).pack(side="left")

    tk.Label(
        top,
        text="Chapter Progress: 0%",
        font=("Segoe UI", 11, "bold"),
        fg=ACCENT,
        bg=BG
    ).pack(side="left", padx=25)

    canvas = tk.Canvas(
        page_frame,
        bg=BG,
        highlightthickness=0
    )
    canvas.pack(
        side="left",
        fill="both",
        expand=True,
        padx=(20, 0)
    )

    scrollbar = tk.Scrollbar(
        page_frame,
        orient="vertical",
        command=canvas.yview
    )
    scrollbar.pack(
        side="right",
        fill="y",
        padx=(0, 20)
    )

    canvas.configure(
        yscrollcommand=scrollbar.set
    )

    content = tk.Frame(
        canvas,
        bg=BG
    )

    canvas_window = canvas.create_window(
        (0, 0),
        window=content,
        anchor="nw"
    )

    content.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfig(
            canvas_window,
            width=e.width
        )
    )

    def scroll(event):
        canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )

    canvas.bind_all(
        "<MouseWheel>",
        scroll
    )

    def section(title, text):

        card = tk.Frame(
            content,
            bg=CARD,
            padx=25,
            pady=20
        )
        card.pack(
            fill="x",
            padx=10,
            pady=8
        )

        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 15, "bold"),
            fg=ACCENT,
            bg=CARD
        ).pack(anchor="w")

        tk.Label(
            card,
            text=text,
            font=("Segoe UI", 11),
            fg=TEXT,
            bg=CARD,
            wraplength=850,
            justify="left"
        ).pack(
            anchor="w",
            pady=(12, 0)
        )

    if module == "What is Artificial Intelligence?":

        section(
            "📖 CONCEPT",
            "Artificial Intelligence (AI) is a branch of computer science "
            "that builds systems which can perform tasks that normally "
            "require human intelligence. AI can learn from data, identify "
            "patterns, understand information, solve problems and make decisions."
        )

        section(
            "💡 KEY IDEA",
            "Human Intelligence → Learn → Think → Decide\n\n"
            "AI System → Data → Algorithm → Learning/Reasoning → Output"
        )

        section(
            "🌍 REAL-WORLD EXAMPLES",
            "• Google Maps → route suggestions\n"
            "• Netflix → movie recommendations\n"
            "• Banking → fraud detection\n"
            "• Healthcare → medical image analysis\n"
            "• Voice assistants → speech understanding\n"
            "• NEXORA AI → learning, career and data assistance"
        )

        section(
            "🖼️ VISUAL EXPLANATION",
            "                 AI SYSTEM\n"
            "                      ↓\n"
            "              INPUT / DATA\n"
            "                      ↓\n"
            "             PROCESS / ALGORITHM\n"
            "                      ↓\n"
            "              MODEL / KNOWLEDGE\n"
            "                      ↓\n"
            "                 OUTPUT\n\n"
            "Example:\n"
            "Student Question → AI understands → searches knowledge → "
            "generates explanation → Student gets answer"
        )

        section(
            "💻 CODING LAB",
            "Simple Python example:\n\n"
            "name = input('What is your name? ')\n"
            "print('Hello', name)\n\n"
            "This is not full AI, but it demonstrates the basic idea "
            "of receiving input and producing output."
        )

        section(
            "🔍 LINE-BY-LINE",
            "1. input() receives information from the user.\n"
            "2. The information is stored in the variable name.\n"
            "3. print() produces an output.\n"
            "4. A real AI system adds data processing, learning, "
            "reasoning or model inference."
        )

        section(
            "📝 PRACTICE",
            "Try to answer:\n\n"
            "1. What is AI?\n"
            "2. Give three examples of AI.\n"
            "3. What is the difference between human intelligence and AI?\n"
            "4. What are input and output in an AI system?"
        )

        section(
            "🎯 CHAPTER QUIZ",
            "Q1. What does AI stand for?\n"
            "A) Automated Internet\n"
            "B) Artificial Intelligence\n"
            "C) Advanced Input\n\n"
            "Q2. Which is an AI application?\n"
            "A) Fraud detection\n"
            "B) Plain notebook\n"
            "C) Wooden table\n\n"
            "Q3. An AI system generally works with:\n"
            "A) Data and algorithms\n"
            "B) Only paper\n"
            "C) Only electricity"
        )

    else:

        section(
            "📖 CONCEPT",
            f"{module}\n\n"
            f"This chapter belongs to {track_name} at {level} level. "
            "The complete concept, examples, visual explanation, coding "
            "lab, practice and quiz will be provided here."
        )

        section(
            "🖼️ VISUAL EXPLANATION",
            f"{module}\n"
            "      ↓\n"
            "CONCEPT\n"
            "      ↓\n"
            "EXAMPLE\n"
            "      ↓\n"
            "CODE / PRACTICE\n"
            "      ↓\n"
            "QUIZ"
        )

        section(
            "📝 PRACTICE",
            "Practice questions for this chapter will appear here."
        )

        section(
            "🎯 CHAPTER QUIZ",
            "Chapter quiz will appear here."
        )

    canvas.update_idletasks()
    canvas.yview_moveto(0)
    ()

        
# ============================================================
# NEXORA COURSE MODULES
# ============================================================

# ============================================================
# NEXORA LEARN - COURSE SYLLABUS DATABASE
# ============================================================

COURSE_MODULES = {

    "Artificial Intelligence": {
        "BASIC": [
            (1, "What is Artificial Intelligence?", "Meaning, goals and real-world AI."),
            (2, "History & Evolution of AI", "Journey from early AI to modern systems."),
            (3, "Human Intelligence vs AI", "Learning, thinking and decision making."),
            (4, "Types of AI", "Narrow AI, General AI and Super AI."),
            (5, "Intelligent Agents", "Agents, sensors, actuators and environments."),
            (6, "AI Applications", "Healthcare, banking, education and daily life."),
            (7, "AI Problem Solving", "How AI solves problems."),
            (8, "AI Ethics", "Fairness, privacy, safety and responsible AI.")
        ],
        "INTERMEDIATE": [
            (1, "Problem Solving in AI", "Search-based problem solving."),
            (2, "Breadth First Search", "BFS with examples and coding."),
            (3, "Depth First Search", "DFS and traversal."),
            (4, "Informed Search", "Heuristics and intelligent search."),
            (5, "A* Algorithm", "Cost plus heuristic path finding."),
            (6, "Knowledge Representation", "Representing knowledge for AI."),
            (7, "Reasoning", "Logical and probabilistic reasoning."),
            (8, "Expert Systems", "Rule-based intelligent systems.")
        ],
        "ADVANCED": [
            (1, "AI Planning", "Planning and decision systems."),
            (2, "Multi-Agent Systems", "Multiple intelligent agents."),
            (3, "Advanced Reasoning", "Complex reasoning systems."),
            (4, "Responsible AI", "Trustworthy AI development."),
            (5, "AI Architecture", "Designing complete AI systems."),
            (6, "AI Deployment", "Taking AI systems into real use."),
            (7, "AI Project Design", "Designing an end-to-end AI project."),
            (8, "AI Capstone Project", "Build a complete practical AI system.")
        ]
    },

    "python Programming": {
        "BASIC": [
            (1, "Introduction to Python", "Python, features and uses."),
            (2, "Variables & Data Types", "Store and work with data."),
            (3, "Operators", "Arithmetic, comparison and logical operators."),
            (4, "Input & Output", "Interact with users."),
            (5, "Conditional Statements", "if, elif and else."),
            (6, "Loops", "for and while loops."),
            (7, "Strings", "Create and manipulate text."),
            (8, "Lists, Tuples & Sets", "Python collection types."),
            (9, "Dictionaries", "Key-value data structures."),
            (10, "Functions", "Reusable blocks of code.")
        ],
        "INTERMEDIATE": [
            (1, "Object Oriented Programming", "Classes and objects."),
            (2, "Inheritance", "Reuse code through inheritance."),
            (3, "Polymorphism", "Different behavior through one interface."),
            (4, "Encapsulation", "Organize and protect data."),
            (5, "Exception Handling", "Handle runtime errors."),
            (6, "File Handling", "Read and write files."),
            (7, "Modules & Packages", "Organize Python programs."),
            (8, "NumPy", "Numerical computing with arrays."),
            (9, "Pandas", "Data manipulation and analysis."),
            (10, "Matplotlib", "Create data visualizations.")
        ],
        "ADVANCED": [
            (1, "Advanced Python", "Advanced language features."),
            (2, "Decorators", "Modify function behavior."),
            (3, "Generators", "Memory-efficient iteration."),
            (4, "Lambda & Functional Programming", "Functional programming techniques."),
            (5, "APIs with Python", "Connect applications through APIs."),
            (6, "Database Programming", "Work with databases."),
            (7, "Python Automation", "Automate repetitive tasks."),
            (8, "Python AI Project", "Build a practical Python AI application.")
        ]
    },

    "Machine Learning": {
        "BASIC": [
            (1, "What is Machine Learning?", "Meaning and basic ML workflow."),
            (2, "Types of Machine Learning", "Supervised, unsupervised and reinforcement learning."),
            (3, "Dataset & Features", "Rows, columns, features and targets."),
            (4, "Training & Testing", "Train, validation and test data."),
            (5, "Data Preprocessing", "Cleaning and preparing data."),
            (6, "Regression", "Predict continuous values."),
            (7, "Classification", "Predict categories."),
            (8, "ML Evaluation", "Basic evaluation metrics.")
        ],
        "INTERMEDIATE": [
            (1, "Linear Regression", "Build a regression model."),
            (2, "Logistic Regression", "Classification using logistic regression."),
            (3, "Decision Trees", "Tree-based prediction."),
            (4, "Random Forest", "Ensemble tree learning."),
            (5, "K-Nearest Neighbors", "Similarity-based classification."),
            (6, "Support Vector Machine", "Hyperplanes and classification."),
            (7, "Clustering", "Group similar data."),
            (8, "K-Means", "Practical clustering algorithm.")
        ],
        "ADVANCED": [
            (1, "Feature Engineering", "Create useful model features."),
            (2, "Hyperparameter Tuning", "Improve model configuration."),
            (3, "Cross Validation", "Reliable model evaluation."),
            (4, "Ensemble Learning", "Combine multiple models."),
            (5, "Pipelines", "Build reusable ML workflows."),
            (6, "Model Explainability", "Understand model decisions."),
            (7, "ML Deployment", "Use trained models in applications."),
            (8, "End-to-End ML Project", "Build a complete ML system.")
        ]
    },

    "Data Structures & Algorithms": {
        "BASIC": [
            (1, "Introduction to DSA", "Data structures and algorithms."),
            (2, "Time & Space Complexity", "Big O basics."),
            (3, "Arrays", "Store ordered data."),
            (4, "Strings", "String algorithms and operations."),
            (5, "Linked Lists", "Nodes and linked structures."),
            (6, "Stacks", "LIFO data structure."),
            (7, "Queues", "FIFO data structure."),
            (8, "Searching & Sorting", "Basic searching and sorting.")
        ],
        "INTERMEDIATE": [
            (1, "Binary Search", "Fast searching in sorted data."),
            (2, "Recursion", "Functions calling themselves."),
            (3, "Trees", "Hierarchical data structures."),
            (4, "Binary Search Trees", "Ordered tree operations."),
            (5, "Heaps", "Priority-based data structures."),
            (6, "Hashing", "Fast key-value lookup."),
            (7, "Graphs", "Vertices, edges and networks."),
            (8, "BFS & DFS", "Graph traversal algorithms.")
        ],
        "ADVANCED": [
            (1, "Dynamic Programming", "Solve overlapping subproblems."),
            (2, "Greedy Algorithms", "Choose locally optimal solutions."),
            (3, "Backtracking", "Explore possible solutions."),
            (4, "Graph Algorithms", "Advanced graph problem solving."),
            (5, "Shortest Path", "Dijkstra and related methods."),
            (6, "Minimum Spanning Tree", "Kruskal and Prim."),
            (7, "Advanced Complexity", "Analyze difficult algorithms."),
            (8, "DSA Coding Project", "Solve a practical problem set.")
        ]
    },

    "Data Analytics": {
        "BASIC": [
            (1, "Introduction to Data Analytics", "What data analytics means."),
            (2, "Types of Data", "Structured and unstructured data."),
            (3, "Excel Basics", "Tables, formulas and basic analysis."),
            (4, "Data Cleaning", "Missing, duplicate and incorrect data."),
            (5, "Descriptive Statistics", "Mean, median, mode and range."),
            (6, "Data Visualization", "Charts and visual storytelling."),
            (7, "SQL Basics", "Query databases using SQL."),
            (8, "Analytics Workflow", "Collect, clean, analyze and report.")
        ],
        "INTERMEDIATE": [
            (1, "Advanced Excel", "Pivot tables, lookup and analysis."),
            (2, "SQL Queries", "Filtering, grouping and joins."),
            (3, "Python for Analytics", "Use Python for data analysis."),
            (4, "NumPy", "Numerical data processing."),
            (5, "Pandas", "Data cleaning and manipulation."),
            (6, "Matplotlib", "Create analytical charts."),
            (7, "Statistics for Analytics", "Important statistical methods."),
            (8, "Dashboard Development", "Build interactive dashboards.")
        ],
        "ADVANCED": [
            (1, "Advanced SQL", "Subqueries, CTEs and window functions."),
            (2, "Advanced Data Cleaning", "Real-world messy datasets."),
            (3, "Exploratory Data Analysis", "Find patterns and insights."),
            (4, "Business Intelligence", "Turn data into decisions."),
            (5, "Power BI", "Create professional dashboards."),
            (6, "KPI & Reporting", "Metrics and business reporting."),
            (7, "Predictive Analytics", "Use models for future estimates."),
            (8, "Data Analytics Project", "Build a complete analytics project.")
        ]
    },

    "Generative AI": {
        "BASIC": [
            (1, "What is Generative AI?", "Understand AI that creates content."),
            (2, "Generative AI vs Traditional AI", "Compare AI systems and outputs."),
            (3, "Large Language Models", "Understand LLM basics."),
            (4, "Tokens & Context", "How language models process text."),
            (5, "Prompt Engineering", "Write useful prompts."),
            (6, "Text Generation", "Generate useful text with AI."),
            (7, "Image Generation", "How AI creates images."),
            (8, "Generative AI Safety", "Privacy, bias and responsible use.")
        ],
        "INTERMEDIATE": [
            (1, "Advanced Prompting", "Improve AI responses."),
            (2, "Few-Shot Prompting", "Teach through examples."),
            (3, "Structured Output", "Generate JSON and formatted results."),
            (4, "Embeddings", "Represent text as vectors."),
            (5, "Vector Databases", "Store and search embeddings."),
            (6, "RAG", "Connect AI with external knowledge."),
            (7, "AI Agents", "AI systems that use tools."),
            (8, "GenAI Application", "Build a practical GenAI app.")
        ],
        "ADVANCED": [
            (1, "RAG Architecture", "Design retrieval-augmented systems."),
            (2, "Agentic AI", "Build tool-using AI agents."),
            (3, "Function Calling", "Connect AI with application functions."),
            (4, "Fine-Tuning Concepts", "Adapt models to specific tasks."),
            (5, "Multimodal AI", "Work with text, image and audio."),
            (6, "LLM Evaluation", "Measure AI response quality."),
            (7, "GenAI Deployment", "Deploy AI applications."),
            (8, "GenAI Capstone Project", "Build an end-to-end GenAI system.")
        ]
    },

    "Deep Learning": {

        "BASIC": [
            (1, "What is Deep Learning?",
             "Deep learning and its real-world uses."),
            (2, "Neural Networks",
             "Understand neurons and neural network basics."),
            (3, "Perceptron",
             "The basic building block of neural networks."),
            (4, "Layers & Weights",
             "Input, hidden and output layers."),
            (5, "Activation Functions",
             "ReLU, Sigmoid and Softmax."),
            (6, "Loss Functions",
             "Measure how wrong a model is."),
            (7, "Training Neural Networks",
             "Forward pass and learning."),
            (8, "Deep Learning Applications",
             "Vision, speech, text and recommendations.")
        ],

        "INTERMEDIATE": [
            (1, "Backpropagation",
             "How neural networks learn from errors."),
            (2, "Gradient Descent",
             "Optimize neural network parameters."),
            (3, "Convolutional Neural Networks",
             "CNNs for image processing."),
            (4, "Image Classification",
             "Classify images using deep learning."),
            (5, "Recurrent Neural Networks",
             "Neural networks for sequential data."),
            (6, "LSTM Networks",
             "Handle long-term sequence information."),
            (7, "Transfer Learning",
             "Reuse knowledge from trained models."),
            (8, "Deep Learning Project",
             "Build a practical neural network.")
        ],

        "ADVANCED": [
            (1, "Advanced CNN",
             "Modern image-processing architectures."),
            (2, "Advanced RNN",
             "Advanced sequence modelling techniques."),
            (3, "Transformers",
             "Attention-based neural networks."),
            (4, "Attention Mechanism",
             "How models focus on important information."),
            (5, "Vision Transformers",
             "Transformers for computer vision."),
            (6, "Model Optimization",
             "Improve model speed and efficiency."),
            (7, "Deep Learning Deployment",
             "Use trained models in applications."),
            (8, "Deep Learning Capstone",
             "Build an end-to-end deep learning system.")
        ]
    },
    "AI Projects": {

        "BASIC": [
            (1, "AI Project Fundamentals",
             "How to plan an AI project."),
            (2, "Problem Definition",
             "Convert a real problem into an AI problem."),
            (3, "Data Collection",
             "Find and prepare project data."),
            (4, "Data Cleaning",
             "Clean project datasets."),
            (5, "Exploratory Data Analysis",
             "Understand the project data."),
            (6, "Model Selection",
             "Choose a suitable AI or ML model."),
            (7, "Project Testing",
             "Test and evaluate the system."),
            (8, "Mini AI Project",
             "Build a complete beginner AI project.")
        ],

        "INTERMEDIATE": [
            (1, "End-to-End ML Project",
             "Build a complete machine learning workflow."),
            (2, "Data Analytics Project",
             "Analyze a real-world dataset."),
            (3, "Prediction Project",
             "Build a prediction system."),
            (4, "Computer Vision Project",
             "Create an image-based application."),
            (5, "NLP Project",
             "Build a text-processing application."),
            (6, "Recommendation System",
             "Recommend items using data."),
            (7, "AI Web Application",
             "Connect an AI model with an application."),
            (8, "Project Documentation",
             "Write complete technical documentation.")
        ],

        "ADVANCED": [
            (1, "AI System Architecture",
             "Design a complete AI platform."),
            (2, "AI and Data Pipeline",
             "Connect data processing with AI."),
            (3, "RAG Application",
             "Build a knowledge-based AI system."),
            (4, "AI Agent Project",
             "Build an AI system that uses tools."),
            (5, "Multimodal AI Project",
             "Combine text, image and other inputs."),
            (6, "AI Deployment",
             "Deploy an AI application."),
            (7, "Testing and Evaluation",
             "Measure quality and reliability."),
            (8, "NEXORA AI Capstone",
             "Design and build a complete AI platform.")
        ]
    }
}

def get_course_modules(track_name, level):

    return COURSE_MODULES.get(
        track_name,
        {}
    ).get(
        level,
        []
    )


def get_ai_ml_chapter_content(track_name, module):

    content = {

        "What is Artificial Intelligence?": {
            "concept": """
WHAT IS ARTIFICIAL INTELLIGENCE?

Artificial Intelligence (AI) is a branch of Computer Science
that focuses on creating systems that can perform tasks
that normally require human intelligence.

These tasks may include:
• Learning
• Reasoning
• Problem solving
• Understanding language
• Recognising images
• Making decisions
• Finding patterns

The main idea is to make a computer system capable of
performing an intelligent task.

HUMAN INTELLIGENCE

Human
  ↓
Observe
  ↓
Learn from experience
  ↓
Think / Reason
  ↓
Make a decision
  ↓
Take action

AI SYSTEM

Data
  ↓
Algorithm
  ↓
Training / Processing
  ↓
Model
  ↓
Prediction / Decision
  ↓
Output
""",

            "visual": """
AI WORKFLOW

        ┌──────────────┐
        │     INPUT    │
        │ Data / Image │
        │ Text / Sound │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  PROCESSING  │
        │ Algorithms   │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ AI MODEL     │
        │ Learn/Reason │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │   OUTPUT     │
        │ Decision /   │
        │ Prediction   │
        └──────────────┘

HUMAN VS AI

Human:
Experience → Learning → Thinking → Decision

AI:
Data → Algorithm → Training → Inference → Output
""",

            "example": """
REAL-WORLD EXAMPLES

1. Google Maps
AI can analyse roads, traffic and locations to suggest
a suitable route.

2. Netflix
AI can study viewing behaviour and recommend movies.

3. Banking
AI can help detect unusual or suspicious transactions.

4. Healthcare
AI can assist with analysing medical images and information.

5. Online Shopping
AI can recommend products based on previous activity.

6. NEXORA AI
NEXORA can use AI for learning assistance, data analysis,
career guidance and intelligent user interaction.
""",

            "code": """user_age = 20

if user_age >= 18:
    print("Adult")
else:
    print("Minor")
""",

            "lines": """
LINE-BY-LINE EXPLANATION

Line 1:
user_age = 20

A variable named user_age is created and the value 20
is stored inside it.

Line 3:
if user_age >= 18:

The program checks whether the age is greater than
or equal to 18.

Line 4:
print("Adult")

If the condition is True, the program displays Adult.

Line 5:
else:

This block runs when the condition is False.

Line 6:
print("Minor")

The program displays Minor.
""",

            "practice": """
PRACTICE QUESTIONS

1. Define Artificial Intelligence in your own words.

2. List five tasks that AI systems can perform.

3. Explain the basic AI workflow.

4. Give three real-world examples of AI.

5. Write two differences between human intelligence
   and artificial intelligence.

6. Why is data important for modern AI systems?
""",

            "qa": """
QUESTIONS & ANSWERS

Q1. What is Artificial Intelligence?

Answer:
AI is a branch of computer science that develops systems
capable of performing tasks that normally require
human intelligence.

Q2. What are some common AI tasks?

Answer:
Learning, reasoning, pattern recognition, prediction,
language understanding and decision making.

Q3. Give three examples of AI.

Answer:
Recommendation systems, navigation systems and
fraud detection systems.

Q4. What is an AI model?

Answer:
An AI model is a computational system that uses
learned patterns or rules to produce an output.

Q5. What is the difference between AI and normal
programming?

Answer:
Traditional programs usually follow explicitly written
rules. Many modern AI systems learn patterns from data.
""",

            "quiz": """
CHAPTER QUIZ

Q1. What does AI mainly try to achieve?

A. Make computers intelligent at specific tasks
B. Only store files
C. Only create websites
D. Only perform calculations

Answer: A


Q2. Which is an AI application?

A. Recommendation system
B. USB cable
C. Keyboard
D. Monitor

Answer: A


Q3. Which comes first in a typical AI workflow?

A. Output
B. Input / Data
C. Decision
D. Result

Answer: B


Q4. Which is an example of AI in banking?

A. Fraud detection
B. Printing documents
C. Connecting a charger
D. Changing wallpaper

Answer: A


Q5. AI systems can work with:

A. Only numbers
B. Only text
C. Data, images, text and other inputs
D. Only keyboards

Answer: C
"""
        }
    }

    content["What is Machine Learning?"] = {

            "concept": (
                "Machine Learning (ML) is a part of AI where computers "
                "learn patterns from data and use those patterns to make "
                "predictions or decisions.\n\n"
                "Instead of writing every rule manually, we give the "
                "system examples and allow a model to learn from them."
            ),

            "visual": (
                "DATA\n"
                "  ↓\n"
                "PREPARE DATA\n"
                "  ↓\n"
                "TRAIN MODEL\n"
                "  ↓\n"
                "EVALUATE\n"
                "  ↓\n"
                "PREDICT"
            ),

            "example": (
                "🏠 House prices → predict price\n"
                "📧 Email → spam detection\n"
                "🏦 Bank → fraud prediction\n"
                "🛒 Store → demand prediction\n"
                "🎬 Movies → recommendation"
            ),

            "code": (
                "from sklearn.linear_model import LinearRegression\n\n"
                "model = LinearRegression()\n\n"
                "model.fit(X_train, y_train)\n\n"
                "prediction = model.predict(X_test)\n\n"
                "print(prediction)"
            ),

            "lines": (
                "Line 1:\n"
                "Imports the Linear Regression algorithm.\n\n"
                "Line 3:\n"
                "Creates the machine-learning model.\n\n"
                "Line 5:\n"
                "Trains the model using training data.\n\n"
                "Line 7:\n"
                "Uses the trained model to make predictions."
            ),

            "practice": (
                "1. Define Machine Learning.\n"
                "2. What is training data?\n"
                "3. Give two ML applications.\n"
                "4. Explain the ML workflow."
            ),

            "qa": (
                "Q1. What is ML?\n"
                "Answer: ML allows computers to learn patterns "
                "from data and make predictions.\n\n"
                "Q2. What is model training?\n"
                "Answer: Training means learning patterns from "
                "provided data.\n\n"
                "Q3. What comes after training?\n"
                "Answer: Evaluation and prediction."
            ),

            "quiz": (
                "1. Machine Learning is a part of:\n"
                "A) AI\n"
                "B) HTML\n"
                "C) Operating System\n"
                "D) Networking\n\n"
                "Correct Answer: A\n\n"
                "2. ML learns mainly from:\n"
                "A) Data\n"
                "B) Keyboard\n"
                "C) Monitor\n"
                "D) Printer\n\n"
                "Correct Answer: A"
            )
        }

    return content.get(
        module,
        {
            "concept": (
                f"{module} is an important topic in "
                f"{track_name}.\n\n"
                "This chapter explains the concept, its purpose, "
                "working process and practical use."
            ),
            "visual": (
                "INPUT\n"
                "  ↓\n"
                "PROCESS\n"
                "  ↓\n"
                "MODEL / ALGORITHM\n"
                "  ↓\n"
                "OUTPUT"
            ),
            "example": (
                "This concept is used in real-world AI systems "
                "for solving practical problems."
            ),
            "code": (
                f"# Practical example\n"
                f"print(\"Learning \" + {module!r})"
            ),
            "lines": (
                "The first line represents the input or setup.\n"
                "The next line performs the main operation.\n"
                "The final line produces the output."
            ),
            "practice": (
                f"1. Define {module}.\n"
                "2. Explain its working.\n"
                "3. Give one real-world example."
            ),
            "qa": (
                f"Q1. What is {module}?\n"
                "Answer: It is an important concept in "
                f"{track_name}.\n\n"
                "Q2. Where can it be used?\n"
                "Answer: It can be used in practical AI systems."
            ),
            "quiz": (
                f"1. {module} belongs to which learning track?\n"
                f"A) {track_name}\n"
                "B) Cooking\n"
                "C) Music\n"
                "D) Sports\n\n"
                "Correct Answer: A"
            )
        }
    )
def create_learning_visual(heading, body):

    width = 420
    height = 220

    image = Image.new(
        "RGB",
        (width, height),
        "#10172A"
    )

    draw = ImageDraw.Draw(image)

    cyan = "#00D9FF"
    white = "#FFFFFF"
    muted = "#9AA4BF"
    blue = "#243B80"

    draw.rounded_rectangle(
        (5, 5, width - 5, height - 5),
        radius=18,
        outline=cyan,
        width=2
    )

    text = body.lower()

    # AI / ML concept visual
    if "artificial intelligence" in text:

        draw.ellipse(
            (35, 55, 145, 165),
            outline=cyan,
            width=4
        )

        draw.ellipse(
            (55, 75, 125, 145),
            outline=white,
            width=2
        )

        draw.text(
            (48, 25),
            "HUMAN BRAIN",
            fill=white
        )

        draw.text(
            (180, 65),
            "AI SYSTEM",
            fill=cyan
        )

        draw.rectangle(
            (175, 95, 350, 145),
            outline=cyan,
            width=3
        )

        draw.text(
            (200, 110),
            "LEARN • THINK • DECIDE",
            fill=white
        )

        draw.line(
            (145, 110, 175, 110),
            fill=cyan,
            width=3
        )

    elif "machine learning" in text:

        boxes = [
            ("DATA", 25),
            ("TRAIN", 130),
            ("MODEL", 235),
            ("PREDICT", 340)
        ]

        for label, x in boxes:

            draw.rounded_rectangle(
                (x, 85, x + 70, 135),
                radius=8,
                fill=blue,
                outline=cyan,
                width=2
            )

            draw.text(
                (x + 10, 102),
                label,
                fill=white
            )

        for x in [95, 200, 305]:

            draw.line(
                (x, 110, x + 30, 110),
                fill=cyan,
                width=3
            )

    elif "visual" in heading.lower():

        labels = [
            ("INPUT", 25),
            ("PROCESS", 130),
            ("MODEL", 235),
            ("OUTPUT", 340)
        ]

        for label, x in labels:

            draw.rounded_rectangle(
                (x, 80, x + 70, 135),
                radius=8,
                outline=cyan,
                width=3
            )

            draw.text(
                (x + 8, 100),
                label,
                fill=white
            )

        for x in [95, 200, 305]:

            draw.line(
                (x, 108, x + 30, 108),
                fill=cyan,
                width=3
            )

    elif "coding" in heading.lower():

        draw.rounded_rectangle(
            (30, 35, 390, 185),
            radius=12,
            fill="#080C18",
            outline=cyan,
            width=3
        )

        draw.ellipse(
            (48, 50, 58, 60),
            fill="#FF5555"
        )

        draw.ellipse(
            (65, 50, 75, 60),
            fill="#FFD166"
        )

        draw.ellipse(
            (82, 50, 92, 60),
            fill="#55DD88"
        )

        code_lines = [
            "model = AI()",
            "data = load_data()",
            "model.train(data)",
            "result = model.predict()"
        ]

        y = 80

        for line in code_lines:

            draw.text(
                (50, y),
                line,
                fill=cyan
            )

            y += 23

    elif "practice" in heading.lower():

        draw.rounded_rectangle(
            (45, 45, 375, 175),
            radius=15,
            fill=blue,
            outline=cyan,
            width=3
        )

        draw.text(
            (70, 65),
            "PRACTICE",
            fill=cyan
        )

        draw.text(
            (70, 100),
            "?  Solve the problem",
            fill=white
        )

        draw.text(
            (70, 130),
            "✓ Apply what you learned",
            fill=white
        )

    elif "quiz" in heading.lower():

        draw.rounded_rectangle(
            (55, 40, 365, 180),
            radius=15,
            fill=blue,
            outline=cyan,
            width=3
        )

        draw.text(
            (85, 60),
            "CHAPTER QUIZ",
            fill=cyan
        )

        draw.text(
            (85, 100),
            "Q1   A    B    C    D",
            fill=white
        )

        draw.text(
            (85, 135),
            "✓ Check your answer",
            fill=white
        )

    else:

        draw.rounded_rectangle(
            (60, 45, 360, 175),
            radius=15,
            fill=blue,
            outline=cyan,
            width=3
        )

        draw.text(
            (100, 75),
            "LEARN",
            fill=cyan
        )

        draw.text(
            (100, 110),
            "UNDERSTAND",
            fill=white
        )

        draw.text(
            (100, 145),
            "PRACTICE → MASTER",
            fill=white
        )

    return image
# ============================================================
# NEXORA LEARNING CHAPTER
# ============================================================

def create_learning_section(parent, heading, body):

    section = tk.Frame(
        parent,
        bg=CARD,
        padx=18,
        pady=14
    )

    section.pack(
        fill="x",
        padx=25,
        pady=6
    )

    content = tk.Frame(
        section,
        bg=CARD
    )

    content.pack(
        fill="x"
    )

    left = tk.Frame(
        content,
        bg=CARD
    )

    left.pack(
        side="left",
        fill="both",
        expand=True,
        padx=(0, 20)
    )

    tk.Label(
        left,
        text=heading,
        font=("Segoe UI", 13, "bold"),
        fg=ACCENT,
        bg=CARD
    ).pack(
        anchor="w",
        pady=(0, 8)
    )

    tk.Label(
        left,
        text=body,
        font=("Segoe UI", 10),
        fg=TEXT,
        bg=CARD,
        justify="left",
        anchor="nw",
        wraplength=650
    ).pack(
        fill="both",
        expand=True,
        anchor="w"
    )

    visual_frame = tk.Frame(
        content,
        bg="#10172A",
        width=420,
        height=220
    )

    visual_frame.pack(
        side="right",
        fill="y",
        padx=(10, 0)
    )

    visual_frame.pack_propagate(False)

    image = create_learning_visual(
        heading,
        body
    )

    photo = ImageTk.PhotoImage(image)

    visual_label = tk.Label(
        visual_frame,
        image=photo,
        bg="#10172A"
    )

    visual_label.image = photo

    visual_label.pack(
        fill="both",
        expand=True,
        padx=5,
        pady=5
    )
# ============================================================
# NEXORA LEARNING STEP CONTENT
# ============================================================

def create_learning_step_content(
    parent,
    heading,
    key,
    content,
    module,
    state
):

    zoom = state.get("zoom", 100)

    size = max(
        9,
        int(11 * zoom / 100)
    )

    title_size = max(
        12,
        int(16 * zoom / 100)
    )

    card = tk.Frame(
        parent,
        bg=CARD,
        padx=25,
        pady=20
    )

    card.pack(
        fill="both",
        expand=True,
        padx=25,
        pady=15
    )

    tk.Label(
        card,
        text=heading,
        font=("Segoe UI", title_size, "bold"),
        fg=ACCENT,
        bg=CARD
    ).pack(
        anchor="w",
        pady=(0, 15)
    )

    # ---------- CONTENT ----------

    text = content.get(
        key,
        ""
    )

    if not text:
        text = (
            f"Detailed learning material for {module} "
            f"will appear here."
        )

    # ---------- VISUAL DIAGRAM ----------

    if key == "visual":

        diagram = tk.Frame(
            card,
            bg="#10172A",
            padx=20,
            pady=20
        )

        diagram.pack(
            fill="x",
            pady=(0, 20)
        )

        flow = [
            "INPUT / DATA",
            "↓",
            "PROCESS / ALGORITHM",
            "↓",
            "MODEL / KNOWLEDGE",
            "↓",
            "OUTPUT / DECISION"
        ]

        for item in flow:

            tk.Label(
                diagram,
                text=item,
                font=(
                    "Segoe UI",
                    max(10, int(12 * zoom / 100)),
                    "bold"
                ),
                fg=TEXT if item == "↓" else ACCENT,
                bg="#10172A"
            ).pack(
                pady=3
            )

    # ---------- NOTES ----------

    text_box = tk.Text(
        card,
        height=18,
        bg="#10172A",
        fg=TEXT,
        insertbackground=TEXT,
        font=(
            "Segoe UI",
            size
        ),
        relief="flat",
        wrap="word",
        padx=15,
        pady=15
    )

    text_box.pack(
        fill="both",
        expand=True,
        pady=5
    )

    text_box.insert(
        "1.0",
        str(text)
    )

    text_box.config(
        state="disabled"
    )

    # ---------- EXTRA LEARNING NOTE ----------

    if key == "concept":

        note = (
            "\n\nKEY IDEA\n"
            "──────────────\n"
            f"{module} should be understood step by step. "
            "Focus on meaning, working, examples and practical use "
            "before moving to the next section."
        )

        text_box.config(
            state="normal"
        )

        text_box.insert(
            "end",
            note
        )

        text_box.config(
            state="disabled"
        )

    # ---------- PRACTICE ----------

    if key == "practice":

        tk.Label(
            card,
            text=(
                "💡 Practice Tip: First understand the concept, "
                "then try the problem without looking at the answer."
            ),
            font=(
                "Segoe UI",
                size,
                "bold"
            ),
            fg=ACCENT,
            bg=CARD,
            wraplength=900,
            justify="left"
        ).pack(
            anchor="w",
            pady=12
        )

    # ---------- QUIZ ----------

    if key == "quiz":

        tk.Label(
            card,
            text=(
                "🎯 CHAPTER CHECK\n\n"
                "Read the complete chapter before attempting "
                "the quiz. Your next goal is to answer without "
                "looking at the notes."
            ),
            font=(
                "Segoe UI",
                size,
                "bold"
            ),
            fg=TEXT,
            bg=CARD,
            wraplength=900,
            justify="left"
        ).pack(
            anchor="w",
            pady=15
        )  
def open_learning_track(track_name):

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=25,
        pady=20
    )

    create_page_title(
        page_frame,
        track_name,
        "Choose your learning level"
    )

    tk.Button(
        page_frame,
        text="← Back to NEXORA Learn",
        command=show_learn,
        bg=CARD2,
        fg=TEXT,
        relief="flat",
        cursor="hand2",
        padx=15,
        pady=8
    ).pack(
        anchor="w",
        padx=25,
        pady=10
    )

    tk.Label(
        page_frame,
        text="LEARNING JOURNEY",
        font=("Segoe UI", 15, "bold"),
        fg=ACCENT,
        bg=BG
    ).pack(
        anchor="w",
        padx=25,
        pady=(20, 10)
    )

    levels = [
        (
            "🌱 BASIC",
            "Foundation",
            "Start from the basics and build strong concepts.",
            "BASIC"
        ),
        (
            "⚡ INTERMEDIATE",
            "Application",
            "Use concepts to solve practical problems.",
            "INTERMEDIATE"
        ),
        (
            "🚀 ADVANCED",
            "Mastery",
            "Advanced concepts, projects and real-world skills.",
            "ADVANCED"
        )
    ]

    for title, subtitle, description, level in levels:

        card = tk.Frame(
            page_frame,
            bg=CARD,
            padx=20,
            pady=18
        )

        card.pack(
            fill="x",
            padx=25,
            pady=8
        )

        tk.Button(
            card,
            text=title,
            font=("Segoe UI", 14, "bold"),
            fg=ACCENT,
            bg=CARD,
            activeforeground=ACCENT,
            activebackground=CARD2,
            relief="flat",
            bd=0,
            cursor="hand2",
            anchor="w",
            command=lambda l=level: open_course_level(
                track_name,
                l
            )
        ).pack(
            fill="x",
            anchor="w"
        )

        tk.Label(
            card,
            text=subtitle,
            font=("Segoe UI", 10, "bold"),
            fg=TEXT,
            bg=CARD
        ).pack(
            anchor="w",
            pady=(5, 0)
        )

        tk.Label(
            card,
            text=description,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=CARD,
            wraplength=850,
            justify="left"
        ).pack(
            anchor="w",
            pady=(5, 0)
        )          
# ==========================================
# CAREER PAGE
# ==========================================

def show_career():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=20
    )

    create_page_title(
        page_frame,
        "NEXORA Career",
        "Explore roles and the skills commonly used in each path."
    )

    career_paths = [
        (
            "Data Analyst",
            "SQL, spreadsheets, statistics, Python and data visualization."
        ),
        (
            "Machine Learning Engineer",
            "Python, machine learning, model evaluation and deployment."
        ),
        (
            "AI Engineer",
            "Software engineering, AI fundamentals, APIs and model integration."
        ),
        (
            "Data Scientist",
            "Statistics, experimentation, data analysis and predictive modeling."
        )
    ]

    for role, skills in career_paths:
        row = tk.Frame(
            page_frame,
            bg=CARD,
            padx=18,
            pady=14
        )
        row.pack(
            fill="x",
            padx=35,
            pady=5
        )

        tk.Label(
            row,
            text=role,
            font=("Segoe UI", 12, "bold"),
            fg=ACCENT,
            bg=CARD,
            anchor="w"
        ).pack(
            fill="x"
        )

        tk.Label(
            row,
            text=skills,
            font=("Segoe UI", 10),
            fg=TEXT,
            bg=CARD,
            anchor="w",
            justify="left",
            wraplength=850
        ).pack(
            fill="x",
            pady=(5, 0)
        )


# ==========================================
# CREATIVE STUDIO
# ==========================================

def show_creative():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=20
    )

    create_page_title(
        page_frame,
        "Creative Studio",
        "AI-assisted content and creative workflow."
    )

    creative_items = [
        "🎨 Poster & Design Ideas",
        "🎬 Video & Reel Concepts",
        "✍️ Captions & Scripts",
        "📱 Social Media Content",
        "📊 Presentation Planning",
        "🎵 Music & Creative Concepts"
    ]

    for item in creative_items:

        tk.Label(
            page_frame,
            text=item,
            font=("Segoe UI", 12, "bold"),
            fg=TEXT,
            bg=CARD,
            padx=20,
            pady=15
        ).pack(
            fill="x",
            padx=35,
            pady=4
        )


# ==========================================
# SMART SHOPPING
# ==========================================

def show_shopping():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=20
    )

    create_page_title(
        page_frame,
        "NEXORA Smart Shopping",
        "Future AI-powered product discovery and local commerce."
    )

    tk.Label(
        page_frame,
        text=(
            "USER NEED + BUDGET\n"
            "        ↓\n"
            "SEARCH PRODUCTS / LOCAL SHOPS\n"
            "        ↓\n"
            "PRICE + AVAILABILITY + REVIEWS\n"
            "        ↓\n"
            "AI COMPARISON\n"
            "        ↓\n"
            "PERSONALIZED RECOMMENDATION"
        ),
        font=("Segoe UI", 12, "bold"),
        fg=TEXT,
        bg=CARD,
        padx=30,
        pady=30,
        justify="left"
    ).pack(
        fill="x",
        padx=35,
        pady=20
    )


# ==========================================
# PROJECTS PAGE
# ==========================================

def show_projects():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=20
    )

    create_page_title(
        page_frame,
        "NEXORA Projects",
        "Current implementation and future intelligent modules."
    )

    projects = [
        ("✓", "AI Assistant", "Working"),
        ("✓", "Hugging Face Integration", "Working"),
        ("✓", "Conversation Memory", "Working"),
        ("✓", "Request Orchestrator", "Working"),
        ("✓", "Background Threading", "Working"),
        ("✓", "Data Intelligence Lab", "Working"),
        ("✓", "Basic ML Prediction", "Working"),
        ("→", "Advanced AutoML", "Future"),
        ("→", "Voice & Image AI", "Future"),
        ("→", "Community & Local Commerce", "Future")
    ]

    for icon, name, state in projects:

        row = tk.Frame(
            page_frame,
            bg=CARD,
            padx=20,
            pady=12
        )

        row.pack(
            fill="x",
            padx=35,
            pady=4
        )

        tk.Label(
            row,
            text=icon,
            font=("Segoe UI", 13, "bold"),
            fg=ACCENT,
            bg=CARD
        ).pack(
            side="left",
            padx=(0, 12)
        )

        tk.Label(
            row,
            text=name,
            font=("Segoe UI", 11, "bold"),
            fg=TEXT,
            bg=CARD
        ).pack(
            side="left"
        )

        tk.Label(
            row,
            text=state,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=CARD
        ).pack(
            side="right"
        )


# ==========================================
# FUTURE SCOPE
# ==========================================

def show_future():

    hide_chat()
    clear_page()

    page_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=20
    )

    create_page_title(
        page_frame,
        "NEXORA Future Scope",
        "Planned expansion of the NEXORA AI ecosystem."
    )

    future_items = [
        "🎙 Voice Assistant",
        "🖼 Image Understanding",
        "🧠 Advanced Generative AI",
        "🤖 AutoML Model Selection",
        "📈 Advanced Forecasting",
        "🔍 Explainable AI — WHY / HOW / WHAT-IF",
        "👥 Community & Social Features",
        "📹 Voice & Video Communication",
        "🛍 Local Business Marketplace",
        "🎵 Music & Media Intelligence"
    ]

    for item in future_items:

        tk.Label(
            page_frame,
            text=item,
            font=("Segoe UI", 11, "bold"),
            fg=TEXT,
            bg=CARD,
            padx=20,
            pady=13
        ).pack(
            fill="x",
            padx=35,
            pady=3
        )


# ==========================================
# SIDEBAR BUTTON CONNECTION
# ==========================================

home_btn.config(
    command=show_home
)

assistant_btn.config(
    command=show_assistant
)

tools_btn.config(
    command=show_tools
)

projects_btn.config(
    command=show_projects
)


# ==========================================
# START APPLICATION
# ==========================================

show_assistant()

input_box.focus()

root.mainloop()