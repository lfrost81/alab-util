import requests
import json


def request(query, ip, port, index_name, type_name='', method='post', operation='_search',
            row_str_query=False, ignore_error=0):
    address = ip + ':' + str(port)

    # Make URI
    uri = '/'.join(['http:/', address])
    if len(index_name) > 0:
        uri = '/'.join([uri, index_name])
        if len(type_name) > 0:
            uri = '/'.join([uri, type_name])
    if len(operation) > 0:
        uri = '/'.join([uri, operation])

    # Convert Query to bytes
    bytes_query = ''
    if len(query) > 0:
        if type(query) is dict:
            bytes_query = json.dumps(query)
        elif row_str_query:
            bytes_query = bytes(query, encoding='utf-8')
        else:
            bytes_query = json.dumps(json.loads(query))

    # Send
    if method == 'post':
        response = requests.post(uri, data=bytes_query)
    elif method == 'get':
        response = requests.get(uri, data=bytes_query)
    elif method == 'delete':
        response = requests.delete(uri, data=bytes_query)
    elif method == 'put':
        response = requests.put(uri, data=bytes_query)

    if response.status_code != ignore_error:
        if response.status_code != 200:
            print(response.reason)
            raise

    print(response.text)
    return response.text
