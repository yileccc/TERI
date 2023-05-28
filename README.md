# TERI:  An Effective Framework for Trajectory Recovery with Irregular Time Intervals

## Dependencies
- Python>=3.7
- torch>=1.13.0



## Datasets

Please download the T-drive dataset [here](https://drive.google.com/drive/folders/1XHcrLSUpH9xSi2P92hZWn74vtw3JZ_gj?usp=sharing), and then extract it in `data` folder.



## Evaluate Model

We provide the trained models for the two stages in `model` folder. You can directly evaluate the trained models on test set by running:
```
cd recovery_stage
python test_TERI.py
```



## Train Model

To train TERI, change to the `recovery_stage` or `detection_stage` folder for the stage you are interested and run following command: 

```
python train.py --batch_size=128 --num_epochs=150
```