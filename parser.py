import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default='16')
    parser.add_argument('--mode', type=str, default='embedding')
    parser.add_argument('--prompt', type=str, default='inspired2')
    parser.add_argument('--query_instr', type=str, default='Retrieve relevant items based on user conversation history:')
    parser.add_argument('--doc_instr', type=str, default='Represent the item description for retrieval:')
    parser.add_argument('--data_json', type=str, default='inspired2/test_processed_title.jsonl')
    parser.add_argument('--db_json', type=str, default='inspired2/inspired2_item_db_title.jsonl')
    parser.add_argument('--embeddings_path', type=str, default='') # inspired2/inspired2_item_embeddings2.pt
    parser.add_argument('--base_model_path', type=str, default='"GritLM/GritLM-7B"')
    parser.add_argument('--target_model_path', type=str, default='')
    parser.add_argument('--to_json', type=str, default='log_name')
    parser.add_argument('--query_max_len', type=int, default='128')
    parser.add_argument('--passage_max_len', type=int, default='1024')
    parser.add_argument("--store_results", action='store_true', help="store or not")
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--instruction_aug', action='store_true')
    parser.add_argument('--pooling', type=str, default=' ')
    parser.add_argument('--passage_num', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--debug_mode', type=bool, default=False)

    # /home/user/junpyo/gritlm/model_weights/test1/0413164554/E5
    args = parser.parse_args()

    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(__file__)
    elif sysChecker() == "Windows":
        args.home = ''
    print(args.home)

    return args
