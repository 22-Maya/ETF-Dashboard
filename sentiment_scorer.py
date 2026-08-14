from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3
import pandas as pd

analyzer = SentimentIntensityAnalyzer()
print("Sentiment analyzer ready.")

def score_text(text):
    if not text or len(text.strip()) < 5:
        return 0.0
    scores = analyzer.polarity_scores(text)
    return round(scores["compound"], 4)

def score_all_unscored():
    conn = sqlite3.connect("etf_news.db")
    
    unscored = pd.read_sql("""
        SELECT id, ticker, headline, body 
        FROM articles 
        WHERE sentiment IS NULL
    """, conn)
    
    if unscored.empty:
        print("No unscored articles found.")
        conn.close()
        return

    print(f"Scoring {len(unscored)} articles...")

    for _, row in unscored.iterrows():
        body = row["body"] if isinstance(row["body"], str) else ""
        text = row["headline"] + " " + body
        score = score_text(text)
        conn.execute(
            "UPDATE articles SET sentiment = ? WHERE id = ?",
            (score, row["id"])
        )
        print(f"  {row['ticker']} | {score:+.4f} | {row['headline'][:60]}")

    conn.commit()
    conn.close()
    print("\nDone scoring.")

def get_summary():
    conn = sqlite3.connect("etf_news.db")
    summary = pd.read_sql("""
        SELECT 
            ticker,
            ROUND(AVG(sentiment), 4)  AS avg_sentiment,
            COUNT(*)                  AS article_count,
            ROUND(AVG(CASE WHEN date = DATE('now') 
                      THEN sentiment END), 4) AS today_sentiment,
            ROUND(AVG(CASE WHEN date >= DATE('now', '-7 days') 
                      THEN sentiment END), 4) AS week_sentiment
        FROM articles
        WHERE sentiment IS NOT NULL
        GROUP BY ticker
        ORDER BY avg_sentiment DESC
    """, conn)
    conn.close()
    return summary

def get_signal(score):
    if score is None:
        return "NO DATA"
    if score > 0.5:
        return "STRONG BUY"
    elif score > 0.2:
        return "BUY"
    elif score > -0.2:
        return "HOLD"
    elif score > -0.5:
        return "SELL"
    else:
        return "STRONG SELL"

if __name__ == "__main__":
    score_all_unscored()

    print("\n--- ETF Sentiment Summary ---")
    summary = get_summary()

    for _, row in summary.iterrows():
        momentum = (
            round(row["today_sentiment"] - row["week_sentiment"], 4)
            if row["today_sentiment"] and row["week_sentiment"]
            else None
        )
        signal = get_signal(row["avg_sentiment"])
        momentum_str = f"{momentum:+.4f}" if momentum else "N/A"

        print(f"""
Ticker:         {row['ticker']}
Avg sentiment:  {row['avg_sentiment']:+.4f}
Articles:       {row['article_count']}
Signal:         {signal}
7-day momentum: {momentum_str}
        """)