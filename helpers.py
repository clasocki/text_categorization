from enum import Enum


class Document(object):
    def __init__(self, id, tokens):
        self.id = id
        self.tokens = tokens


class DataSourceType(Enum):
    local_nasa_elections = 1
    local_explicit_groups = 2
    local_hci_graphs = 3
    db_explicit_groups = 4


class FactorizationAlgorithm(Enum):
    gradient_descent = 1
    linear_svd = 2
    randomized_svd = 3
    gradient_descent_engine = 4