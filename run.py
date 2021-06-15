from helpers.parser import parse
from model.trainers import get_trainer
import time

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name
    n_epochs = args.n_epochs
    trainer = get_trainer(model_name)(**vars(args))

    trainer.initialize()

    total_start = time.time()
    for epoch in range(n_epochs):
        trainer.step(epoch)

    seconds = time.time() - total_start
    trainer.finalize(seconds)
