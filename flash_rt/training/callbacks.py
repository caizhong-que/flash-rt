"""Training callbacks for Word2Vec."""

from pathlib import Path
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    def __init__(self, log_interval: int = 1):
        super().__init__()
        self.epoch = 0
        self.log_interval = log_interval

    def on_epoch_begin(self, model):
        if self.epoch % self.log_interval == 0:
            print(f"Epoch #{self.epoch} start")
        self.epoch += 1

    def on_epoch_end(self, model):
        if (self.epoch - 1) % self.log_interval == 0:
            print(f"Epoch #{self.epoch - 1} end")


class EpochSaver(CallbackAny2Vec):
    def __init__(self, save_dir: str = "checkpoints", model_name: str = "word2vec"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def on_epoch_end(self, model):
        model.save(str(self.save_dir / f"{self.model_name}.model"))
