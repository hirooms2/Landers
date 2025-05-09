import os
import json


class Prompter():
    def __init__(self, args):
        self.args = args
        self.template = json.load(open(os.path.join(args.home, 'template', f"{args.prompt}.json"), 'r', encoding='utf-8'))
        self.max_char_len=max(self.args.query_max_len, self.args.passage_max_len) * 10

    def get_instruction(self):
        return self.template['query_instr'], self.template['doc_instr']


    def generate_prompt(self, data):
        if 'inspired2' in self.args.prompt:
            dialog = data['context'][-self.max_char_len:]
            return self.template['template'].format(dialog=dialog)
        elif self.args.prompt == 'durecdial2':
            dialog = data['context'][-self.max_char_len:]
            profile = ' '.join([i.strip() for i in data['profile'].split('|') if 'accept' in i.lower() or 'reject' in i.lower()])
            return self.template['template'].format(dialog=dialog, profile=profile, goal=data['goal'])