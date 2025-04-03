# python logparser.py --model deepseek-chat --dataset adservice

import argparse
import json
from logbatcher.parsing_base import single_dataset_paring
from logbatcher.parser import Parser
from logbatcher.util import data_loader

# model: gpt-4o-mini, gpt-3.5-turbo, deepseek-chat
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='the Large Lauguage model used in LogBatcher, default to be gpt-4o-mini.')
    parser.add_argument('--dataset', type=str, default='Proxifier')
    args = parser.parse_args()
    return args

# load api key, dataset format and parser
if __name__ == '__main__':
    args = set_args()
    model, dataset, folder_name = args.model, args.dataset, args.dataset
    config = json.load(open('logparser/config.json', 'r'))
    # if config['api_key_from_openai'] == '<OpenAI_API_KEY>' and config['api_key_from_together'] == '<Together_API_KEY>':
    if config['api_key_from_openai'] == '<OpenAI_API_KEY>':
        print("Please provide your OpenAI API key and Together API key in the config.json file.")
        exit(0)

    parser = Parser(model, folder_name, config)

    # load contents from raw log file
    print(f"Loading {dataset} dataset")
    contents = data_loader(
        file_name=f"data/all-logs-per-service/{dataset}.log",
        dataset_format= config['datasets_format'][dataset],
        file_format ='raw'
    )

    # parse logs
    single_dataset_paring(
        dataset=dataset,
        contents=contents,
        output_dir= f'data/all-log-templates/{folder_name}/',
        parser=parser,
        debug=True,
        chunk_size=10000,
    )
