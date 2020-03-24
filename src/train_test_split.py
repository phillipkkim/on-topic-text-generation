import csv
import numpy as np
from tqdm import tqdm

np.random.seed(0)

def write_to_individual_files():
    train_count = 0
    dev_count = 0
    test_count = 0
    with open("../data/cnn_articles.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row.keys())
            random_number = np.random.randint(100)
            if random_number == 99:
                with open("../data/dev/highlights/" + str(dev_count) + ".txt", "w") as h:
                    h.write(row["highlights"])
                    h.close()
                with open("../data/dev/stories/" + str(dev_count) + ".txt", "w") as t:
                    t.write(row["story"])
                    t.close()
                dev_count += 1
            elif random_number == 98:
                with open("../data/test/highlights/" + str(test_count) + ".txt", "w") as h:
                    h.write(row["highlights"])
                    h.close()
                with open("../data/test/stories/" + str(test_count) + ".txt", "w") as t:
                    t.write(row["story"])
                    t.close()
                test_count += 1
            else:
                with open("../data/train/highlights/" + str(train_count) + ".txt", "w") as h:
                    h.write(row["highlights"])
                    h.close()
                with open("../data/train/stories/" + str(train_count) + ".txt", "w") as t:
                    t.write(row["story"])
                    t.close()
                train_count += 1
                if train_count % 100 == 0:
                    print(train_count, 92578)

def write_to_csv():
    data_file = open("../data/cnn_articles.csv", "r")
    train_file = open("../data/train.csv", "w")
    dev_file = open("../data/dev.csv", "w")
    test_file = open("../data/test.csv", "w")
    reader = csv.DictReader(data_file)
    fieldnames = ["story", "highlights"]
    train_writer = csv.DictWriter(train_file, fieldnames = fieldnames)
    dev_writer = csv.DictWriter(dev_file, fieldnames = fieldnames)
    test_writer = csv.DictWriter(test_file, fieldnames = fieldnames)
    train_writer.writeheader()
    dev_writer.writeheader()
    test_writer.writeheader()
    # row_count = sum(1 for row in reader)
    # for row in tqdm(reader, total = row_count):
    for row in reader:
        if len(row["story"].split(" ")) <= 128:
            continue
        random_number = np.random.randint(100)
        if random_number >= 95:
            dev_writer.writerow({"story": row["story"], "highlights": row["highlights"]})
        elif random_number < 95 and random_number >= 90:
            test_writer.writerow({"story": row["story"], "highlights": row["highlights"]})
        else:
            train_writer.writerow({"story": row["story"], "highlights": row["highlights"]})
    data_file.close()
    train_file.close()
    dev_file.close()
    test_file.close()

if __name__ == '__main__':
    # write_to_csv()
    train_file = open("../data/train.csv", "r")
    dev_file = open("../data/dev.csv", "r")
    test_file = open("../data/test.csv", "r")
    fieldnames = ["story", "highlights"]
    train_reader = csv.DictReader(train_file, fieldnames = fieldnames)
    dev_reader = csv.DictReader(dev_file, fieldnames = fieldnames)
    test_reader = csv.DictReader(test_file, fieldnames = fieldnames)
    print(sum(1 for row in train_reader)-1) #89446
    print(sum(1 for row in dev_reader)-1) #946
    print(sum(1 for row in test_reader)-1) #926
    train_file.close()
    dev_file.close()
    test_file.close()
