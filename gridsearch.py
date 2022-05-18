
from train import *
import pickle


def gridsearch():
    import itertools
    import pandas as pd

    grid_search = dict(
        dim_feedforward=[2048, 2048*2],
        lr=[0.0001],
        batch_size=[100],
        weight_decay=[1e-05],
        clip=[0.25],
        n_layers=[3],
        epochs=[8],
        n_head=[12, 10],
        dropout=[0.2, 0.3],
        model=['TFM'],
    )

    models_ = {
        'TFM': TransformerModel,
    }

    history = pd.DataFrame(columns=list(grid_search.keys()) + ['test_accuracy', 'train_accuracy', 'val_accuracy', 'evaluator'])

    print(f"{len([i for i in itertools.product(*(grid_search.values()))])} train to do")

    for param_values in itertools.product(*(grid_search.values())):
        params = {key: value for key, value in zip(grid_search.keys(), param_values)}
        print("Train with", params)

        model = models_[params['model']](
            device=device,
            input_size=data.embeddings.shape[1],
            w2v_vectors=data.embeddings,
            output_size=data.output_size,
            dim_feedforward=params['dim_feedforward'],
            n_layers=params['n_layers'],
            dropout=params['dropout'],
            n_head=params['n_head']
        ).to(device)

        evaluator = Evaluator(model=model, data=data)

        trainer = Trainer(
            model, data, device, evaluator=evaluator,
            batch_size=params['batch_size'],
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            clip=params['clip'],
            verbose=True
        )

        torch.cuda.empty_cache()
        trainer.fit(params['epochs'])
        test_acc, train_acc, val_acc = evaluator.final_eval()

        hist = {**params,
                'test_accuracy': test_acc, 'train_accuracy': train_acc,
                'val_accuracy': val_acc, 'evaluator': evaluator}

        del model
        del trainer
        del evaluator.model
        del evaluator.data
        torch.cuda.empty_cache()

        pickle.dump(history, open('history_auto_train_Transformer_2', 'wb'))

        history = pd.concat([history, pd.DataFrame([hist])])
        print(history.loc[:, list(grid_search.keys()) + ['test_accuracy', 'train_accuracy', 'val_accuracy']])
        print(history.sort_values(['val_accuracy'], ascending=False))


if __name__ == '__main__':
    use_cuda = True
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print("Use device", device)

    data: TextDataset = TextDataset(include_subtrees=False, mode='one_hot', device=device)
    print('TEXT DATASET CREATED')

    gridsearch()
