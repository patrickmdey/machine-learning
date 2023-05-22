from utils import *
import sys

x_min, x_max, y_min, y_max = 0, 1, 0, 1
file_name = "TP3-1"

point_amount = 30

if len(sys.argv) > 1:
    print("TRUE", str(sys.argv[1]))
    point_amount = int(sys.argv[1])

data, line = random_points_within_range(x_min, x_max, y_min, y_max, point_amount)
df = pd.DataFrame(data, columns=["x", "y", "class"])
df.to_csv(file_name+".csv", index=False)
line_df = pd.DataFrame([line], columns=["m", "b"])
line_df.to_csv(file_name+"-line.csv", index=False)