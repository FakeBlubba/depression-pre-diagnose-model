import pandas as pd
import os

def get_DB_path():
    DB_PATH = "\\".join(os.path.dirname(os.path.abspath(__file__)).split("\\")[:-1])
    return  os.path.join(DB_PATH, "dbs")

def get_data_from_SAaDd():
    try:
        path = os.path.join(get_DB_path(), "Students Anxiety and Depression Dataset.xlsx")
        data = pd.read_excel(path)

        return data.values.tolist() # [['oh my gosh', 1.0],...]
    
    except FileNotFoundError:
        return -1
    

#   Returns a list of data
def get_data_from_ddrc():
    try:
        path = os.path.join(get_DB_path(), "depression_dataset_reddit_cleaned.csv")
        dataframe = pd.read_csv(path)
        return dataframe.values.tolist() # [['oh my gosh', 1],...]
    except FileNotFoundError:
        return -1

#   Create a database with index column, text column, value column 
def create_dataset(data, file_name = "composite_db.csv"):
    path = os.path.join(get_DB_path(), file_name)

    df = pd.DataFrame(data, columns=["Stringa", "Intero"])
    df.to_csv(file_name, index=True)




db = get_data_from_SAaDd()
print(db[0])
db2 = get_data_from_ddrc()
print(db2[0])