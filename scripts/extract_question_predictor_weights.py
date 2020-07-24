import torch
import os
import shutil


os.makedirs("./data/qasrl_question_predictor", exist_ok=True)

general_weights = "./data/qasrl_parser_elmo/weights.th"
all_weights = torch.load(general_weights, map_location="cpu")

all_prefixes = set(key.split(".")[0] for key in all_weights.keys())
print(all_prefixes)

QUESTION_PREFIX = "question_predictor."
question_keys = [key for key in all_weights.keys()
                 if key.startswith(QUESTION_PREFIX)]
question_weights = {}
for k in question_keys:
    new_key = k.replace(QUESTION_PREFIX, "")
    question_weights[new_key] = all_weights[k]

question_weights_path = "./data/question_predictor/weights.th"
torch.save(question_weights, question_weights_path)

# Now, extract the config and the vocab
shutil.copy2("./configs/qasrl_question_predictor.json", "./data/qasrl_question_predictor/config.json")
dst_vocab = "./data/qasrl_quetion_predictor/vocabulary"
if not os.path.exists(dst_vocab):
    shutil.copytree("./data/qasrl_parser_elmo/vocabulary", dst_vocab)

