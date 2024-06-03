import time
import yaml
from yaml_format import YAMLize, represent_ordereddict
from collections import OrderedDict
import dotenv
from argparse import ArgumentParser
from tqdm import tqdm
import os, textwrap
import LLAMA
from llama_index.llms.openai import OpenAI
import requests, json


dotenv.load_dotenv()
DEBUG = True
def main(args):
    # Setup ############################################################
    print("Getting Retriever")
    # document_path = "/home/chris/chatbot-rag-server/textbooks/Chapter_15.pdf"
    # index = LLAMA.get_index(args.doc_path)

    retriever = LLAMA.get_retriever(
        args.doc_path,
        args.collection_name,
        k=args.k,
        force_reload=args.REFRESH_DOC_EMBEDDING
        )

    for model_name in tqdm(args.models):
        print(f"Getting Model: {model_name}")
        model = LLAMA.get_model(model_name, args.temperature)

        prompt_template = LLAMA.get_sme_template()

        expanded_retriever = LLAMA.get_expanded_retriever(retriever, model)

        query_engine = LLAMA.get_query_engine(expanded_retriever, model, prompt_template)

        outputs = []
        supports = []
        with open(args.prompts) as f:
            prompts = f.readlines()
            
        for prompt in tqdm(prompts):
            outputs_this_prompt = []
            supports_this_prompt = []

            if args.SHOW_OUTPUTS:
                print("="*args.viewport_width)
                pretty = textwrap.wrap(f"Prompt: {prompt}", args.viewport_width)
                [print(line) for line in pretty]
                print("-"*args.viewport_width)

            for i in range(args.n_multishot):
                raw_response = query_engine.query(prompt)
                response = str(raw_response)
                support = raw_response.source_nodes[0].node.metadata["window"]

                if args.SHOW_OUTPUTS:
                    pretty = textwrap.wrap(f"Response: {response}", args.viewport_width)
                    [print(line) for line in pretty]
                    if i < args.n_multishot-1:
                        print("-"*args.viewport_width)
                    
                outputs_this_prompt.append(response)
                supports_this_prompt.append(support)
                
            outputs.append(outputs_this_prompt)
            supports.append(supports_this_prompt)

            if args.SHOW_OUTPUTS:
                print("="*args.viewport_width + "\n"*3)
            
            # save responses to file
        date = time.strftime("%m%d%Y")
        exp = "T" if not args.no_expand else "F"

        i = 0
        filename = f"results_{model_name}_{exp}_{date}_{i}.yml"
        while(os.path.isfile(os.path.join(args.results_dir, filename))):
            i += 1
            filename = f"results_{model_name}_{exp}_{date}_{i}.yml"

            
        if DEBUG: print(f"Writing to {filename}")

        prompts = [prompt.replace("\n", " ").strip() for prompt in prompts]
        outputs = [[output.replace("\n", " ").strip() for output in output_list] for output_list in outputs]
        supports = [[support.replace("\n", " ").strip() for support in support_list] for support_list in supports]
        # YAMLizes test resutls
        test_results = YAMLize(model_name, date, exp, prompts, outputs, supports, args)
        # write YAMLized results to YAML file  
        with open(os.path.join(args.results_dir, filename), 'w') as file:
            yaml.dump(test_results, file, allow_unicode=True, default_flow_style=False)
yaml.add_representer(OrderedDict, represent_ordereddict)          


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--doc_path', type=str, default=os.path.join("documents", "Chapter_15.pdf"))
    parser.add_argument('--models', type=str, default=None, nargs='+', help="Model names")
    parser.add_argument('--prompts', type=str, default=os.path.join("prompts", "theory_of_computing_prompts.txt"))
    parser.add_argument('--results_dir', type=str, default=os.path.join("responses"))
    parser.add_argument('--client_path', type=str, default=os.path.join("clients", "client.db"))
    parser.add_argument('--collection_name', type=str, default="textbooks")
    parser.add_argument('--n_multishot', type=int, default=1)
    parser.add_argument('--viewport_width', type=int, default=80)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--no_expand', action='store_true')
    parser.add_argument('--pool_responses', action='store_true')

    parser.add_argument('--REFRESH_DOC_EMBEDDING', action='store_true')
    parser.add_argument('--SHOW_OUTPUTS', action='store_true')
    args = parser.parse_args()
    main(args)
