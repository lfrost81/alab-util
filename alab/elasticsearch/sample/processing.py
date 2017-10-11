import json


def make_doc_as_aline():
    ofp = open('tag.knowledge.data', 'w')

    with open('data.jsonld', 'r') as fp:
        for doc in json.load(fp):
            ofp.write(json.dumps(doc, ensure_ascii=False) + '\n')

    ofp.close()


def remove_keyword_mappings(default_analyzer='keyword'):
    with open('mapping.json', 'r') as fp:
        mapping_dict = json.load(fp)

        traverse_and_edit_mapping(mapping_dict, default_analyzer)

    print(json.dumps(mapping_dict, indent=2, ensure_ascii=False))


def traverse_and_edit_mapping(element, default_analyzer_dict=None):
    if 'fields' in element:
        element.pop('fields')
        if default_analyzer_dict == 'keyword':
            element['type'] = 'keyword'
        elif default_analyzer_dict is not None:
            element['analyzer'] = default_analyzer_dict

    for k in element.keys():
        if type(element[k]) is not dict:
            continue
        traverse_and_edit_mapping(element[k], default_analyzer_dict)


if __name__ == '__main__':
    make_doc_as_aline()
    #remove_keyword_mappings()


