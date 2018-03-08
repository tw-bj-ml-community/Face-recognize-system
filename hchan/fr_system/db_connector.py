import sqlite3
import io
import numpy as np

# def adapt_array(arr):
#     out = io.BytesIO()
#     np.save(out, arr)
#     out.seek(0)
#     return sqlite3.Binary(out.read())
#
#
# def convert_array(text):
#     out = io.BytesIO(text)
#     out.seek(0)
#     return np.load(out)
#
#
# # Converts np.array to TEXT when inserting
# sqlite3.register_adapter(np.ndarray, adapt_array)
#
# # Converts TEXT to np.array when selecting
# sqlite3.register_converter("ndarray", convert_array)


class DBManager:
    def __init__(self):
        self.conn = sqlite3.connect('dev.db')
        self.cursor = self.conn.cursor()

    def execute_insert(self, sql_str, *args):
        self.cursor.execute(sql_str, args)
        self.conn.commit()
        return self.cursor.lastrowid

    def execute_select(self,sql_str):
        result = self.cursor.execute(sql_str)
        return result.fetchall()
    def __del__(self):
        self.conn.close()

# c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")
