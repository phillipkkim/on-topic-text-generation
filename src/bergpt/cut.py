import csv

filename = "../../outputs/bergpt_init_embed/newest_bergpt_test_prompt.csv"
new_filename = "../../outputs/bergpt_init_embed/newest_bergpt_test_prompt_new.csv"

with open(filename, "r") as f, open(new_filename, "w") as n_f:
    fieldnames = ["highlight", "reference", "generated"]
    writer = csv.DictWriter(n_f, fieldnames = fieldnames)
    reader = csv.DictReader(f, fieldnames = fieldnames)
    for row in reader:
        highlight = row["highlight"]
        generated = row["generated"]
        if highlight[1:] in generated:
            new_generated = generated.replace(highlight[1:], '')
        # elif " ".join(highlight.split(" ")[1:]) in generated:
        #     print("hahahahahahahahaha")
        #     new_generated = generated.replace(" ".join(highlight.split(" ")[1:]), '')
        else:
            new_generated = " ".join(generated.replace(" ".join(highlight.split(" ")[1:]), '').split(" ")[1:])
            print(new_generated)
        writer.writerow({"highlight": highlight, "generated" : new_generated, "reference": row["reference"]})
