class MyNode:
    def __init__(self, df, col_name, parent=None):
        self.parent = None
        self.probability_table = {}
        table_cols = set(df[col_name])
        for col in table_cols:
            self.probability_table[col] = []
        
        print(self.probability_table)


    def set_parent(self, parent):
        self.parent = parent


    def __str__(self):
        return str(self.value)