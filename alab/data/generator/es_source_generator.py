import json
import os
import re

basic_json_form = "{\"index\": {\"_index\": \"$index_name\", \"_type\": " \
                  "\"$type_name\", \"_id\": \"$id\"}}"


def generate(index_name_prefix, type_name, input_dir_path, output_dir_path, use_field_name_index=False):
    basic_json_form2 = basic_json_form.replace("$type_name", type_name)

    # Gather each files by date
    files = os.listdir(input_dir_path)
    date_files = {}
    fixed_files = []

    for filename in files:
        m = re.search('.*?_(\d+).*?', filename)
        if os.path.isdir(os.path.join(input_dir_path, filename)):
            continue
        elif filename.startswith('.'):
            continue
        elif not filename.endswith('.txt'):
            continue
        elif m is not None:
            date = m.group(1)
            if date not in date_files:
                date_files[date] = []
            date_files[date].append(os.path.join(input_dir_path, filename))
        else:
            fixed_files.append(filename)

    for key in date_files:
        for file in fixed_files:
            date_files[key].append(os.path.join(input_dir_path, file))

    # Create and merge files
    for date, paths in date_files.items():
        output_path = os.path.join(output_dir_path, date[0:6])
        os.makedirs(output_path, exist_ok=True);

        index_name = index_name_prefix + '.' + date[0:6]
        output_path = os.path.join(output_path, 'index.txt')

        output_file = open(output_path, 'w')
        for input_path in paths:
            if os.path.isdir(input_path):
                continue

            input_file = open(input_path, 'r')
            print(input_path)

            while True:
                line = input_file.readline()
                if not line:
                    break

                if len(line.strip()) == 0:
                    continue

                json_string = line
                json_dict = json.loads(json_string)

                obj_id = ""
                tmp_dict = {}
                rem_dict = {}
                term_dict = {}
                for key, val in json_dict.items():
                    if not key.startswith("#"):
                        if type(val) is list:
                            tmp_dict[key] = ",".join(val)
                    else:
                        rem_dict[key] = True
                        key = key[1:]
                        tmp_dict[key] = val
                    if key == "@id":
                        rem_dict[key] = True
                        obj_id = json_dict[key]
                        key = key[1:]
                        tmp_dict[key] = val

                for key, val in rem_dict.items():
                    json_dict.pop(key)

                for key, val in tmp_dict.items():
                    json_dict[key] = val

                if use_field_name_index:
                    for key in json_dict:
                        if key.startswith('@'):
                            continue
                        term_dict[key] = True
                    json_dict['ptag_terms'] = ' '.join(term_dict.keys())

                # Replace index name and type name
                basic_json = basic_json_form2.replace("$index_name", index_name)
                basic_json = basic_json.replace("$id", obj_id)
                batch_string = basic_json + "\n"
                batch_string += str(json_dict).replace('\'', '\"') + "\n"
                output_file.write(batch_string)

            input_file.close()
        output_file.close()


def main():
    generate(index_name_prefix="ptag_theme", type_name="object", input_dir_path="./output",
             output_dir_path="./output")


if __name__ == '__main__':
    main()


