# Saver

This saver will automatically save your experiments in a folder where we store all the features, increasing the knowledge base with your research.

It works with CV, NLP and RecSys experiments based on Pytorch Lightning.

All you have to do is:

- If working with NLP:
```
saver = NLPSaver(train_dataset,
                 test_dataset,
                 model,
                 label_column='text')
```

The `train_dataset` and `test_dataset` fields are simply HF datasets. The model is a `LightningModule` object containing the model.
The `label_column` is optional and represents the name of the index of the label on the HF dataset.

- If working with CV:
```
saver = CVSaver(train_dataset,
                 test_dataset,
                 model)
```

The `train_dataset` and `test_dataset` fields are simply `torchvision` datasets. The model is a `LightningModule` object containing the model.

Then you have to do before you start your **Pytorch Lightning** training:
```
logger, emissions_callback = saver.start_and_get_csv_logger()
```

Insert `logger` and `emission_callback` in your `Trainer`:
```
trainer = pl.Trainer(
        max_epochs= ...,
        accelerator=...,
        devices=...,
        logger=logger,
        callbacks=[emissions_callback],
    )
test_result = trainer.test(model, test_loader)
```

Lastly, save some test metrics. Simply call:
```
saver.stop_and_save(accuracy=test_result[0]['test_acc'])
```

## Warning

The `model` field, which must be a `LightningModule` object must contain:
- `self.learning_rate`: learning rate value;
- `self.batch_size`: batch size value;
- `self.number_of_epochs`: number of epochs;
- `self.model`: the Pytorch or HF model depending on the task.