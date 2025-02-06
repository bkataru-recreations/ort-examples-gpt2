# ort-examples-gpt2

following [github.com/pykeio/ort/tree/main/examples/gpt2](https://github.com/pykeio/ort/tree/main/examples/gpt2)

## setup

download the tokenizer & model into `data/`

```bash
$ mkdir -p data
$ curl -o data/tokenizer.json https://raw.githubusercontent.com/pykeio/ort/refs/heads/main/examples/gpt2/data/tokenizer.json   
$ curl -o data/gpt2.onnx https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx  
```
