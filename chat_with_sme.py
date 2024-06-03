from argparse import ArgumentParser
import os
import LLAMA


def main(args):
    # Setup ############################################################
    print(f"Getting Model: {args.model}")
    model = LLAMA.get_model(args.model, args.temperature)

    # index = LLAMA.get_index(args.doc_path)
    retriever = LLAMA.get_retriever(
        args.doc_path,
        k=2,
        force_reload=args.force_reload
        )

    print("Getting Retriever")
    

    prompt_template = LLAMA.get_sme_template() # Formats the prompt for the model

    if not args.expand:
        query_engine = LLAMA.get_query_engine(retriever, model, prompt_template)
    else:
        expanded_retriever = LLAMA.get_expanded_retriever(retriever, model)
        query_engine = LLAMA.get_query_engine(expanded_retriever, model, prompt_template)

   

    # Prompt loop #####################################################
    while(True):
        print("="*args.viewport_width)

        user_prompt = input("Prompt: ")

        if user_prompt.strip() == "": break
        print("-"*args.viewport_width)

        print("Response: \n")
        response =  query_engine.query(user_prompt)
        print(str(response)) # Actual call to chain
    
        print()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--doc_path', type=str, default=os.path.join("documents", "Chapter_15.pdf"))
    parser.add_argument('--client_path', type=str, default=os.path.join("clients", "client.db"))
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--viewport_width", type=int, default=80)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--pool_responses", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()

    main(args)
