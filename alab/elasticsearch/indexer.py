from alab.elasticsearch import agent
import codecs
import path
import json
import os


class ElasticIndexer:
    def __init__(self, config_path, source_path, block_size=3000, debug=False, recreate_index=True):
        self.config_path = config_path
        with open(self.config_path) as fp:
            self.conf = json.loads(fp.read())

        self.source_path = source_path

        self.block_size = block_size
        self.debug = debug
        self.recreate_index = recreate_index

    def run(self):
        # Get configurations
        ip = self.conf['es']['ip']
        port = self.conf['es']['port']
        index = self.conf['es']['index']

        meta_template = {'index': {'_index': index, "_type": 'object', '_id': 0}}

        # Delete and create index
        if self.recreate_index:
            query = self.conf['create_index_query']
            agent.request({}, ip, port, index, method='delete', operation='', ignore_error=404)
            agent.request(query, ip, port, index, method='put', operation='', ignore_error=400)

        # Index
        source_file_path = os.path.join(self.source_path)
        with codecs.open(source_file_path, 'r') as fp:
            tmp_list = []
            for i, line in enumerate(fp):
                if not line:
                    break

                meta_template['index']['_id'] = json.loads(line)['id']

                tmp_list.append(json.dumps(meta_template))
                tmp_list.append(line.replace('\n', ''))

                if self.debug:
                    print(json.loads(line))

                if i % self.block_size == self.block_size-1:
                    bulk = '\n'.join(tmp_list) + '\n'
                    agent.request(bulk, ip, port, index, method='put', operation='_bulk',
                                  row_str_query=True)
                    tmp_list.clear()
                    if self.debug:
                        print(bulk)

                    print(i+1, 'lines processed')

            if len(tmp_list) > 0:
                bulk = '\n'.join(tmp_list) + '\n'
                agent.request(bulk, ip, port, index, method='put', operation='_bulk',
                              row_str_query=True)

            print(i+1, 'lines processed')


def main():
    indexer = ElasticIndexer('conf/server_config.json',
                             os.path.join(path.ROOT, 'output', 'customer_201701.txt'))
    indexer.run()


if __name__ == '__main__':
    main()