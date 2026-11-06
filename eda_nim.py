
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent
START = (3, 4, 5)
nodes = 0

def moves(piles, order="small_first"):
    result = []
    for i, n in enumerate(piles):
        takes = range(1, n + 1) if order == "small_first" else range(n, 0, -1)
        for take in takes:
            new = list(piles)
            new[i] = n - take
            result.append(tuple(new))
    return result

def minimax(piles, player):
    global nodes
    nodes += 1
    if sum(piles) == 0:
        return -1 if player == "A" else 1
    values = [minimax(nxt, "B" if player == "A" else "A") for nxt in moves(piles)]
    return max(values) if player == "A" else min(values)

def alphabeta(piles, player, alpha, beta, order):
    global nodes
    nodes += 1
    if sum(piles) == 0:
        return -1 if player == "A" else 1
    if player == "A":
        value = -2
        for nxt in moves(piles, order):
            value = max(value, alphabeta(nxt, "B", alpha, beta, order))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = 2
    for nxt in moves(piles, order):
        value = min(value, alphabeta(nxt, "A", alpha, beta, order))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value

def best_move(piles, player):
    best_nxt, best_value = None, None
    for nxt in moves(piles, "big_first"):
        value = alphabeta(nxt, "B" if player == "A" else "A", -2, 2, "big_first")
        better = best_value is None or (value > best_value if player == "A" else value < best_value)
        if better:
            best_nxt, best_value = nxt, value
    return best_nxt

# ساخت دیتاست مطابق اسلایدهای پروژه: ۲۰۰ بازی با seed ثابت
rng = np.random.default_rng(0)
rows = []
for game in range(200):
    piles = tuple(int(x) for x in rng.integers(1, 5, size=3))
    nodes = 0
    start_value = alphabeta(piles, "A", -2, 2, "big_first")
    nodes_ab = nodes
    nodes = 0
    minimax(piles, "A")
    nodes_mm = nodes
    board, player, n_moves = piles, "A", 0
    while sum(board) > 0:
        if player == "A":
            board = best_move(board, "A")
        else:
            options = moves(board)
            board = options[rng.integers(len(options))]
        player = "B" if player == "A" else "A"
        n_moves += 1
    rows.append({
        "p1": piles[0], "p2": piles[1], "p3": piles[2],
        "total": sum(piles),
        "nim_sum": piles[0] ^ piles[1] ^ piles[2],
        "first_wins": start_value > 0,
        "nodes_minimax": nodes_mm,
        "nodes_alphabeta": nodes_ab,
        "moves": n_moves,
        "winner": "B" if player == "A" else "A"
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "nim_games.csv", index=False)

# 1) کیفیت داده و ساختار
print("شکل دیتاست:", df.shape)
print("\nمقادیر گمشده:\n", df.isna().sum())
print("\nتعداد ردیف‌های تکراری:", df.duplicated().sum())
print("\nنوع داده ستون‌ها:\n", df.dtypes)
print("\nآمار توصیفی:\n", df.describe(include="all").T)

# شاخص‌های تکمیلی عددی
num = df.select_dtypes(include=np.number)
stats = num.agg(["mean", "median", "var", "std", "min", "max"]).T
stats["skew"] = num.skew()
stats["q1"] = num.quantile(.25)
stats["q3"] = num.quantile(.75)
stats.to_csv(OUT / "numeric_statistics.csv")
print("\nشاخص‌های تکمیلی:\n", stats.round(2))

# 2) بررسی دسته‌ها و توازن
print("\nفراوانی برنده:\n", df["winner"].value_counts())
print("\nفراوانی برد شروع‌کننده:\n", df["first_wins"].value_counts())
print("\nتعداد شروع‌های یکتا:", df[["p1","p2","p3"]].drop_duplicates().shape[0])

# 3) نمودارهای تحلیلی
plt.rcParams["figure.dpi"] = 140
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df["nodes_minimax"].plot(kind="hist", bins=20, ax=axes[0,0], edgecolor="black")
axes[0,0].set_title("Distribution of Minimax nodes")
axes[0,0].set_xlabel("Number of visited nodes")
axes[0,0].set_ylabel("Frequency")

(df.groupby("total")["first_wins"].mean()*100).plot(kind="bar", ax=axes[0,1])
axes[0,1].set_title("First-player win rate by total stones")
axes[0,1].set_xlabel("Total stones at start")
axes[0,1].set_ylabel("Win rate (%)")
axes[0,1].set_ylim(0, 105)

df[["nodes_minimax","nodes_alphabeta"]].plot(kind="box", ax=axes[1,0])
axes[1,0].set_yscale("log")
axes[1,0].set_title("Search nodes: Minimax vs Alpha-Beta (log scale)")
axes[1,0].set_ylabel("Visited nodes (log scale)")

axes[1,1].scatter(df["total"], df["nodes_minimax"], alpha=.65)
axes[1,1].set_yscale("log")
axes[1,1].set_title("Total stones vs Minimax nodes")
axes[1,1].set_xlabel("Total stones")
axes[1,1].set_ylabel("Minimax nodes (log scale)")
fig.tight_layout()
fig.savefig(OUT / "eda_charts.png", bbox_inches="tight")
plt.close(fig)

# نمودار اضافی برای فراوانی بردها
fig, ax = plt.subplots(figsize=(6,4))
(df["winner"].value_counts(normalize=True).sort_index()*100).plot(kind="bar", ax=ax)
ax.set_title("Winner distribution")
ax.set_xlabel("Winner")
ax.set_ylabel("Share of games (%)")
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(OUT / "winner_distribution.png", bbox_inches="tight")
plt.close(fig)

# 4) روابط و خروجی‌های تحلیلی
corr = df[["total","nodes_minimax","nodes_alphabeta","moves"]].corr()
corr.to_csv(OUT / "correlation_matrix.csv")
print("\nماتریس همبستگی:\n", corr.round(2))
print("\nدرصد برد A در موقعیت‌های شروع:", round(df["first_wins"].mean()*100,1))
print("توافق nim_sum غیرصفر با first_wins:", round(((df["nim_sum"] != 0) == df["first_wins"]).mean()*100,1), "%")
print("نسبت مجموع گره‌های Alpha-Beta به Minimax:", round(df["nodes_alphabeta"].sum()/df["nodes_minimax"].sum()*100,2), "%")
print("درصد برد بازیکن هوشمند A مقابل B تصادفی:", round((df["winner"]=="A").mean()*100,1), "%")

# گزارش متنی کوتاه
report = f"""
گزارش EDA دیتاست Nim
====================
ابعاد: {df.shape[0]} ردیف و {df.shape[1]} ستون
مقادیر گمشده: {int(df.isna().sum().sum())}
ردیف‌های تکراری کامل: {int(df.duplicated().sum())}
شروع‌های یکتای سه پشته: {df[['p1','p2','p3']].drop_duplicates().shape[0]}
میانگین nodes_minimax: {df.nodes_minimax.mean():.1f}
میانه nodes_minimax: {df.nodes_minimax.median():.1f}
انحراف معیار nodes_minimax: {df.nodes_minimax.std():.1f}
کمینه/بیشینه nodes_minimax: {df.nodes_minimax.min()} / {df.nodes_minimax.max()}
درصد برد A در وضعیت شروع: {df.first_wins.mean()*100:.1f}%
سهم مجموع گره‌های Alpha-Beta از Minimax: {df.nodes_alphabeta.sum()/df.nodes_minimax.sum()*100:.2f}%
درصد برد A در بازی‌های شبیه‌سازی‌شده: {(df.winner=='A').mean()*100:.1f}%
همبستگی total و nodes_minimax: {df[['total','nodes_minimax']].corr().iloc[0,1]:.2f}

نکته: تکراری‌ها به دلیل نمونه‌گیری تصادفی ۲۰۰ بازی از تنها ۴^3=۶۴ ترکیب شروع ممکن ایجاد شده‌اند؛
پس تکرار یک ردیف الزاماً خطا نیست و در این آزمایش می‌تواند طبیعی باشد. حذف خودکار آن‌ها توصیه نمی‌شود.
"""
(OUT / "EDA_report.txt").write_text(report, encoding="utf-8")
print(report)
