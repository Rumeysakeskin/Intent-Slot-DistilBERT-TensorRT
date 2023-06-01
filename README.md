## Joint Intent/Slot Classification for Jetson Nano, TX1/TX2, Xavier NX, and AGX Xavier with Onnxruntime and Onnx-TensorRT

The Joint Intent/Slot Classification was trained with NeMo and deployed with both onnxruntime and TensorRT for optimized performance.
All computation is performed using the onboard GPU.

### *Benchmark Performance Between Execution Providers*

| *Execution Provider* |*Inference Time* |
|:----------------:|:----------------:|
| onnxruntime      | 0.63685 sec      |
| onnx-tensorrt    | 0.00375 sec      |

*Performances are evaluated on NVIDIA A100 80 GB. Speeds vary by device.*


## Table Of Contents
- [Data Preperation](#Data-Preperation)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Export Onnx Model](#Export-Onnx-Model)
- [Onnxruntime Inference](#Onnxruntime-Inference)
- [TensorRT Inference](#TensorRT-Inference)
---
## Data Preperation
- To convert to the format of the model data, use the `import_datasets.py` utility, which implements the conversion for the Assistant dataset `assistant_utils.py`. 

For a dataset that follows your own annotation format, we recommend using one text file for all samples of the same intent, with the name of the file as the name of the intent. 
Use one line per query, with brackets to define slot names. This is very similar to the assistant format, and you can adapt this converter utility or your own format with small changes:

```python
answerid;scenario;intent;answer_annotation;answer_from_anno;answer_from_user
1001;coffee_machine;brewing_query;"[amount_per_person : dört] kişilik [coffee_type : filtre kahve] demle";"dört kişilik filtre kahve demle";"4 kişilik filtre kahve demle"
```

- Open the `prepare_dataset.ipynb` and run the following command for dataset conversion.
```python
# convert the dataset to the NeMo format
python import_datasets.py --dataset_name=assistant --source_data_dir=dataset --target_data_dir=nemo_format
```

- After conversion, target data directory should contain the following files:
```python
|--nemo_format/
  |-- dict.intents.csv
  |-- dict.slots.csv
  |-- train.tsv
  |-- train_slots.tsv
  |-- test.tsv
  |-- test_slots.tsv
```

---
Open the `nemo-trainer-onnx-tensorrt.ipynb` for the next steps below.
## Training
```python
nemo_model = NemoTrainer()
nemo_model.train()
```
---
## Evaluation
```python
nemo_model.test()
```
---
## Export Onnx Model
```python
nemo_checkpoint_path = "/nemo_experiments/IntentSlot/2023-05-29_10-51-01/checkpoints/"
onnx_filename = "turkish_isc.onnx"
nemo_model.export_model(nemo_checkpoint_path, onnx_filename)
```
---
## Onnxruntime Inference
```python
intent_slot_label_path = "/nemo_format/"
nemo_checkpoint_path= "/nemo_experiments/IntentSlot/2023-05-29_10-51-01/checkpoints/"
onnx_model = "turkish_isc.onnx"
query = ['iki kişilik türk kahvesi yap']

nemo_model.inference(query, nemo_checkpoint_path, onnx_model, intent_slot_label_path)
```
---
## TensorRT Inference
```python
trt_nemo = IntentSlotClassificationTRTInference()
query = ['iki kişilik türk kahvesi yap']
trt_nemo.inference(query)
```



