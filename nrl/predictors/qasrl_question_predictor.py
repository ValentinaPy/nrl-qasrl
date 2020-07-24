from copy import deepcopy

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField, SpanField, ListField
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides

from nrl.data.util import cleanse_sentence_text
from nrl.predictors.qasrl_parser import read_verb_file, QaSrlParserPredictor

INFLECTIONS = "data/wiktionary/en_verb_inflections.txt"


@Predictor.register("qasrl_question_predictor")
class QaSrlQuestionPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)
        self._model_vocab = model.vocab
        self._slot_labels = self._model.slot_labels
        self._verb_map = read_verb_file(INFLECTIONS)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text'].split()
        text = cleanse_sentence_text(text)
        tokens = [Token(t) for t in text]
        text_field = TextField(tokens, self._dataset_reader._token_indexers)
        pred_span = json_dict['predicate']['span']
        pred_idx = int(pred_span.split(":")[0])
        predicate_indicator = SequenceLabelField([1 if i == pred_idx else 0
                                                  for i in range(len(text))], text_field)
        arg_spans = [arg['span'].split(":") for arg in json_dict['arguments']]
        arg_spans = [(int(begin), int(end)) for begin, end in arg_spans]
        # end index is inclusive in allennlp
        arg_spans = [SpanField(begin, end - 1, text_field)
                     for begin, end in arg_spans]
        arg_field = ListField(arg_spans)
        instance = Instance({
            'text': text_field,
            'predicate_indicator': predicate_indicator,
            'labeled_spans': arg_field
        })
        return instance

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        outputs = super().predict_json(inputs)
        question_slots = outputs['questions']
        # question_slots contains outputs for all input spans
        the_verb = inputs['predicate']['text']
        outputs = deepcopy(inputs)
        for argument, slots in zip(outputs['arguments'], question_slots):
            question_text = QaSrlParserPredictor.make_question_text(self, slots, the_verb)
            argument['question'] = question_text
            slots_as_dict = dict(zip(self._slot_labels, slots))
            argument.update(slots_as_dict)
        return outputs