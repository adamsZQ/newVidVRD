### Download Extracted features or generate by yourself

```bash
cd newVidVRD/utils && 
python extract_features.py -i [Videos_dir] \
    -o [Output_dir] \
    -m [model_name] \
    -b [batch_size]
```
```
# The input_dir and output_dir are your own identified directory, former of which includes raw videos to be extracted and the latter is to store results.

# And we provide several CNN networks to be chosen:
    InceptionV3, Xception, Resnet50, VGG16, VGG19
# You can simply name one of these and run the python script "extract_features.py".

# The default size of Batch_Size is 32, but apparently you can set it yourself.
```

We extracted frames features from several types of neural networks:

| network | download_link |
| ------- | ------------- |
| inceptionv3 | --- |
| xception | --- |
| resnet50 | --- |
| vgg16 | --- |
| vgg19 | --- |

