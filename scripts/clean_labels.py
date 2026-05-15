import sqlite3
conn = sqlite3.connect('etc_agent.db')
cur = conn.cursor()
cur.execute("DELETE FROM risk_model_label WHERE labeling_method LIKE 'historical%'")
conn.commit()
print(f"Deleted {cur.rowcount} old labels for re-labeling")
conn.close()
