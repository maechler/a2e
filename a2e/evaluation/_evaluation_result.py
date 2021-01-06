class EvaluationResult:
    def __init__(self, cost: float, config: dict = None, info: dict = None):
        self.id = None
        self.cost = cost
        self.config = config if config is not None else {}
        self.info = info if info is not None else {}

    def set_id(self, id: int):
        if self.id is None:
            self.id = id

    def add_info(self, key: str, value: any):
        self.info[key] = value

    def __float__(self):
        return self.cost

    def __str__(self):
        return '{:.4f}'.format(self.cost)

    def to_dict(self):
        return {
            'id': self.id,
            'cost': self.cost,
            **self.info,
            **{f'config_{k}': v for k, v in self.config.items()},
        }
