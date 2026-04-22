import os
import pandas as pd
import psycopg as pg
from dotenv import load_dotenv

def load_interactions(table_name):
    load_dotenv()

    configs = {"sslmode": "require", "target_session_attrs": "read-write"}
    db_credits = {
        "host": os.getenv("DB_DESTINATION_HOST"),
        "port": os.getenv("DB_DESTINATION_PORT"),
        "dbname": os.getenv("DB_DESTINATION_NAME"),
        "user": os.getenv("DB_DESTINATION_USER"),
        "password": os.getenv("DB_DESTINATION_PASSWORD")
    }

    configs.update(db_credits)

    # with pg.connect(**configs) as conn:
    #     with conn.cursor() as cur:
    #         cur.execute(f"SELECT * FROM {table_name}")

    #         data = cur.fetchall()

    #         columns = [col.name for col in cur.description]

    # df = pd.DataFrame(data=data, columns=columns)
    df = pd.read_csv("data/events.csv")
    return df