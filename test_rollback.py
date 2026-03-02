import sys
from sqlalchemy import create_engine
engine = create_engine("postgresql://ziko:1234@localhost:5432/job1")
conn = engine.connect()
print(conn.connection)
print(conn.connection.cursor())
try:
    conn.connection.rollback()
    print("Success conn.connection.rollback()")
except Exception as e:
    print("Failed: ", e)

