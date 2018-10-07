# sketcher

start trainning:
```
python train.py
```

convert model
```
tensorflowjs_converter --input_format=tf_saved_model --output_node_names='Softmax' tfmodel/ jsmodel/
```

test:
```
firefox index.html
```
