import numpy as np
import collections
import argparse
import json
import os
import re


def main(input_file_path, output_dir_name):
    input_file = open(input_file_path, 'r', encoding='utf-8')
    json_text = input_file.read()
    json_obj = json.loads(json_text, object_pairs_hook=collections.OrderedDict)

    try:
        os.mkdir(output_dir_name)
    except FileExistsError:
        pass

    tables = json_obj
    for table_name, columns in tables.items():
        is_unique = False
        is_first_col = True
        result_matrix = []

        for col_name, col_conf in columns.items():
            if 'unique' in col_conf:
                if col_conf['unique'] == 'y':
                    is_unique = True

            if is_unique:
                if 'var' in col_conf:
                    args = col_conf['var']
                    args_len = len(args)
                    if is_first_col or dim == args_len:
                        var = args
                    else:
                        var = [args[i % args_len] for i in np.arange(dim)]
                elif 'range' in col_conf:
                    args = col_conf['range']
                    var = np.arange(args[0], args[1])
                elif 'dependency' in col_conf:
                    args = col_conf['dependency']
                    dep_table, dep_column = re.split("@", args)
                    if 'var' in tables[dep_table][dep_column]:
                        dep_vars = tables[dep_table][dep_column]['var']
                    elif 'range' in tables[dep_table][dep_column]:
                        dep_vars = tables[dep_table][dep_column]['range']
                    var = []
                    dep_var_len = len(dep_vars)
                    for i in np.arange(dim):
                        var.append(dep_vars[i % dep_var_len])
            else:
                if 'range' in col_conf:
                    arg_limit = col_conf['limit']
                    args = col_conf['range']
                    var = np.random.randint(args[0], args[1], arg_limit)
                elif 'dependency' in col_conf:
                    args = col_conf['dependency']
                    dep_table, dep_column = re.split("@", args)
                    dep_vars = tables[dep_table][dep_column]['var']
                    dep_vars_len = len(dep_vars)
                    var = [dep_vars[i % dep_vars_len] for i in np.arange(dim)]
                elif 'var' in col_conf:
                    args = col_conf['var']
                    indices = np.random.randint(0, len(args), dim)
                    var = [args[i] for i in indices]

            if is_first_col:
                is_first_col = False

            result_matrix.append(np.array(var))
            dim = len(var)

        result_matrix = np.array(result_matrix, dtype=object).T
        print(table_name, result_matrix.shape)
        print(result_matrix)

        output_path_prefix = os.path.join(output_dir_name, 'table_')
        output_file = open(output_path_prefix + table_name + '.csv', 'w', encoding="utf-8")
        for i, row in enumerate(result_matrix):
           row = str(row).replace(' ', ',').replace('[', '').replace(']', '')
           output_file.write(row + '\n')
        output_file.close()

    input_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transactional Data Generator')
    parser.add_argument(type=str, dest='input_file_path', help='Input File Path')
    parser.add_argument(type=str, dest='output_dir_path', help='Output Directory Path')
    args = parser.parse_args()

    main(args.input_file_path, args.output_dir_path)