import sqlite3, os
conn = sqlite3.connect('etc_agent.db')
cur = conn.cursor()
cur.execute("SELECT label, COUNT(*) FROM risk_model_label GROUP BY label")
rows = cur.fetchall()
print("DB: %.0f MB" % (os.path.getsize('etc_agent.db') / 1024 / 1024))
print("Labels:")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")
conn.close()
