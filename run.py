from helpers.parser import parse
from model.trainers import get_trainer
import time

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name
    n_epochs = args.n_epochs
    inference_mode = args.inference_mode
    
    trainer = get_trainer(model_name)(**vars(args))
    trainer.initialize()
    total_start = time.time()

    if inference_mode:
        # Inference
        trainer.infer()
    else:
        # Training 
        for epoch in range(n_epochs):
            trainer.step(epoch)

    seconds = time.time() - total_start
    trainer.finalize(seconds)
