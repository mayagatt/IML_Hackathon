import random
from sklearn.model_selection import train_test_split


def split_train_test(path):
    f = open(path)
    lines = f.readlines()
    line_groups = []
    while len(lines) > 0 :
        rand = random.randint(1, 5)
        if len(lines) >= rand:
            line_groups.append(" ".join(lines[0:rand]))
            del lines[0:rand]
    train, test = train_test_split(line_groups, test_size=0.2)
    return train, test

train, test = split_train_test("building_tool_all_data.txt")
print(train)
print(len(train))
print(test)
print(len(test))

