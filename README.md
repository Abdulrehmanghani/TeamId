# TeamId
TeamID is a model for the team players classification. The over all data preparation, traning, evaluation and deployment follows the following steps,
1. Data Preparation 
2. Model Training 
3. Model Evaluation
4. Model Conversion 

## Data Prepration 
Data prepration is important step before traning the model. Please follow the following steps for data prepration 

1. Copy the new dataset zip folder to the Download images folder and unzip the json file to the path `</Download_images/>`

2. The second step is to downloads the images corresponding to the json file downloaded in the steps 1. To download the images run following set of commands. 

```
cd Download_images/
python data_download.py --csv <Name of the jason file >
```
3. Third step is to prepare the players croops and csv according to the teams . Therefore, activate the detr environment using 
```
conda activate detr
```
and then run the following command 
```
cd ../TeamID_pre-processing
python teamid_model_pre_pros.py --csv < Path of the Json file >

```
4. The fourth step for data prepration is to copy and sprate the both team players croops images. Go to the folder Copy_image_crops  and open the file **copy_images.py**. In this code add all the exsisiting csv files one by one save and run the code. Now copy images by  using the following command 

```
cd ../Copy_image_crops/
python copy_images.py
```
5. The fifth step is to split data in to traning and validation sets. For this we have to split both team players croop images into train and val. Copy images from Copy_image_crops folder to ```Val_train_split_images/train/``` and ```Val_train_split_images/val/```and this can be done by using the following command, 
```
cd ../Val_train_split_images/
python val_train_split_images.py
```
## Model Training 

In this step we will train the TeamID model on the newly prepared dataset. First, we will check the available GPU by using the following command  and check the availble GPU.

```
nvidia-smi 
```

For traning the model go to the ```/Train``` and run the file **resnet_classifier.py**  using the following command
```
cd ../train
CUDA_VISIBLE_DEVICES= < Int corresponding to the available GPU like 0 or 1 >  python resnet_classifier.py --data_dir custom_dataset --output_dir TeamID_model_output --epoch < Enter number of Epocs> 
```

## Model Evaluation 

The evaluation of this model will done in the training code. The classification report will shown at the end of the training.

## Model Conversion
In this step we will convert the **pytorch model** to **ONNX** so that we can deploy it inside the **Triton Server** . We will add the pytorch model path and the output model path in the code file named as **convert_pth_to_onnx.py** and run the following commands
```
cd ../Model_conversion
python convert_pth_to_onnx.py
```


