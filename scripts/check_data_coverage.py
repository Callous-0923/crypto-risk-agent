import sqlite3, os
db = 'etc_agent.db'
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT asset, market_type, COUNT(*), MIN(open_time), MAX(open_time) FROM historical_market_bar GROUP BY asset, market_type ORDER BY asset, market_type")
rows = cur.fetchall()
sz = os.path.getsize(db) / 1024 / 1024
print(f"DB: {sz:.0f} MB")
print(f"{'Asset':6s} {'Market':12s} {'Bars':>10s} {'From':22s} {'To':22s}")
print("-" * 76)
for r in rows:
    print(f"{r[0]:6s} {r[1]:12s} {r[2]:>10,} {r[3]:22s} {r[4]:22s}")
conn.close()
