import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    # both
    parser.add_argument('--batch_size', type=int, default='1')
    parser.add_argument('--db_path', type=str, default="refine_results/0722-233911_E1.jsonl")

    ## gritlm
    # train
    parser.add_argument('--debug_mode', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='embedding')
    parser.add_argument('--prompt', type=str, default='inspired2')
    parser.add_argument('--query_instr', type=str, default='Retrieve relevant items based on user conversation history:')
    parser.add_argument('--doc_instr', type=str, default='Represent the item description for retrieval:')
    parser.add_argument('--data_json', type=str, default='inspired2/test_processed_title.jsonl')
    parser.add_argument('--db_json', type=str, default='inspired2/inspired2_item_db_title.jsonl')
    parser.add_argument('--embeddings_path', type=str, default='')
    parser.add_argument('--base_model_path', type=str, default='"GritLM/GritLM-7B"')

    # inference
    parser.add_argument('--target_model_path', type=str, default='')
    parser.add_argument('--to_json', type=str, default='log_name')
    parser.add_argument('--query_max_len', type=int, default='128')
    parser.add_argument('--passage_max_len', type=int, default='1024')
    parser.add_argument("--store_results", action='store_true', help="store or not")
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--data_path', type=str, default="")

    parser.add_argument('--output_file', type=str, default="")

    ## refinement
    parser.add_argument('--gpt_api', type=str,
                        default="")
    parser.add_argument('--gpt_model', type=str, default="gpt-4.1")  # gpt-4.1-mini
    parser.add_argument('--epochs', type=int, default=1)


    args = parser.parse_args()

    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(__file__)
    elif sysChecker() == "Windows":
        args.home = ''
    print(args.home)

    return args
