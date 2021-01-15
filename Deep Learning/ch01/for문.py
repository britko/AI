raw_data = [[1, 10], [2, 15], [3, 30], [4,55]]

all_data = [ x for x in raw_data]
x_data = [x[0] for x in raw_data]
y_data = [x[1] for x in raw_data]

print("all_data ==", all_data)
print("x_data ==", x_data)
print("y_data ==", y_data)

even_number = [ n for n in range(0, 10, 2)]
print("even_number ==", even_number)