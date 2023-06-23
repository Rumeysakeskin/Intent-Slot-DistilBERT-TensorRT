## Joint Intent/Slot Classification for Jetson Nano, TX1/TX2, Xavier NX, and AGX Xavier with Onnxruntime and Onnx-TensorRT

The Joint Intent/Slot Classification was trained with NeMo and deployed with TensorRT for optimized performance.
All computation is performed using the onboard GPU.

## Table Of Contents
- [Data Preperation](#Data-Preperation)
- [Training](#Training)
- [Export Onnx Model](#Export-Onnx-Model)
- [TensorRT Inference](#TensorRT-Inference)
- [References](#References)
---
## Data Preperation
- To convert to the format of the model data, use the `import_datasets.py` utility, which implements the conversion for the Assistant dataset `assistant_utils.py`. 

For a dataset that follows your own annotation format, we recommend using one text file for all samples of the same intent, with the name of the file as the name of the intent. 
Use one line per query, with brackets to define slot names. This is very similar to the assistant format, and you can adapt this converter utility or your own format with small changes:

```python
answerid;scenario;intent;answer_annotation;answer_from_anno;answer_from_user
1001;coffee_machine;brewing_query;"I will take [amount_per_person : two] [content : slightly sweet] [coffee_type : turkish coffee]";"I will take two slightly sweet Turkish coffee";"I will take 2 slightly sweet Turkish coffee"
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
Hyper-parameters | Values | 
 ------- | ------- | 
Pretrained BERT Model| distilbert-base-uncased |
Optimizer| Adam |
LR Schedule| WarmupAnnealing |
Precision| 16 |
AMP Level| O1 |
Batch Size| 32 |
Max seq lenght| 50 |
#of train/test samples| 5000/1600 |

---

## Training
Open the `distelbert_train_export_onnx.ipynb` run the following for training.
```python
!(python intent_slot_classification_jetson.py \
  --dataset='./nemo_format/' \
  --config='confing/intent_slot_jetson.yaml' \
  --exp-dir='./nemo_experiments_jetson'\
  --model='distilbert-base-uncased' \
  --epochs=10)
```
---
## Export Onnx Model
```python
nemo_checkpoint_path = "./nemo_experiments_jetson/IntentSlot/2023-06-21_10-57-00/checkpoints/epoch=9-step=2930.ckpt" #ENGLISH
onnx_filename = "ENGLISH_nemo_model_DISTEL_BERT.onnx"
export_model(nemo_checkpoint_path, onnx_filename)
```
---
## TensorRT Inference
Open the `dintent_slot_distelbert_inference_tensorrt.ipynb` and run the following for inference.

```python
en_inference_model = ENNemoDialogueInferencerTRT()
query = "Please prepare two sweet filter coffee with milk"

intent, intent_score, merged_data, inference_time = en_inference_model.inference(query)
```

```python
Query: Please prepare two sweet filter coffee with milk

Predicted Intent: coffee_machine_brewing_query (Score: 0.9904)

Predicted Slot(s):
amount_per_person --> two (Score: 0.6788)
content --> sweet with milk (Score: 0.9168)
coffee_type --> filter coffee (Score: 0.9348)

Inference time: 0.0100 sec
```
---
## References
- [nemo_train_intent.py](https://github.com/dusty-nv/jetson-voice/blob/master/scripts/nemo_train_intent.py)
- [jetson-voice intent_slot.py](https://github.com/dusty-nv/jetson-voice/blob/master/jetson_voice/models/nlp/intent_slot.py)



