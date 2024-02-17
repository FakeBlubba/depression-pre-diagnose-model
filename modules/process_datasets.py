

def format_datasets(datasets_list):
    formatted_db = []
    print(len(datasets_list))
    for d, dataset in enumerate(datasets_list):
        for index, row in enumerate(datasets_list):
            if d == 0:  # format saand
                if row[1] == 1:
                    row[1] = 0
                else:
                    row[1] = 1
        formatted_db.append(row)

    return formatted_db