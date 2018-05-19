from pyhive import hive
conn = hive.Connection(host='localhost', port=3306)
cursor = conn.cursor()
cursor.execute('select * from phy')
for result in cursor.fetchall():
    print(result)
