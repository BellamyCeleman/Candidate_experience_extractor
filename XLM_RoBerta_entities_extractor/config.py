"""
Конфигурация для NER экстрактора
"""

import os


class XLM_RoBerta_entities_extractor_config:
    def __init__(self):
        self.MODEL_PATH = os.getenv(
            "MODEL_PATH",
            "XML_Roberta_neural_network_Anonimizator_finetune/ner_model_output"
        )