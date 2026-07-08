def normalize_rows(rows):
    rows = list(rows or [])
    if not rows:
        return rows

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    return [{column: row.get(column) for column in columns} for row in rows]
