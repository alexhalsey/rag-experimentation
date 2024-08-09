import sqlite3
import csv

conn = sqlite3.connect("path to tracking database")
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS llama_results (
                id integer PRIMARY KEY,
                user_prompt text,
                hyde_prompt text,
                hyde_response text,
                final_prompt text,
                response text,
                retrieved_nodes text,
                source_nodes text,
                temperature float,
                similarity_top_k integer,
                collection text,
                metadata_filter text,
                rerank integer default 0,
                top_n integer default 0,
                sentence_window default 0,
                user text,
                created_at timestamp default current_timestamp              
)""")

c.execute("DROP TABLE llama_results")


# dump to csv
c.execute('SELECT * FROM llama_results') 
rows = c.fetchall()
column_names = [description[0] for description in c.description]

with open('path to csv', 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(column_names)
    csv_writer.writerows(rows)

conn.close()

# add a column
alter_command = '''
                ALTER TABLE us_fed_fsr
                ADD COLUMN metadata_filter TEXT;
                '''
c.execute(alter_command)

conn.commit()
conn.close()


# stuff for demo
sources = rows[23][6].split('text')
for i in range(len(sources)):
    if i % 2 ==1:
        print (sources[i])
