

import os, json

from benchmark.exp_recall.embedding_search_benchmarking import \
    check_code, experiment_prompt


def process(file):
    data = json.load(open(file))
    fixed = False
    for sub_data in data.values():
        for exp_name, exp_contents in sub_data["results"].items():
            exp_cls = experiment_prompt[exp_name][0]
            for ins_res in exp_contents:
                success = ins_res[1]
                code = ins_res[2][0]
                new_success = check_code(code, exp_cls)
                if new_success != success:
                    ins_res[1] = new_success
                    fixed = True
    if fixed:
        new_file = file.replace(".json", "-fixed.json")
        with open(new_file, "w") as f:
            json.dump(data, f, indent=4)



if __name__ == '__main__':
    this_path = os.path.dirname(os.path.abspath(__file__))
    # iterate over json files in the directory
    for file in os.listdir(this_path):
        if file.endswith(".json"):
            process(file)