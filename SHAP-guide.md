# How to SHAP
## The MODULAR Way!

### What, Why?

A non-unified project without an agreed upon requirements.txt with specific versions works fine when everyone works individually. But once all the data needs to be fed into one explainer, it results in total **chaos**.

So the solution is to have a modular SHAP app that can be gracefully connected to everyone's code and spit out the results. This way everyone can use their own pipelines, their own versions of packages, their preferred notebook environment, and then the results can be gathered into SHAP and then we can all enjoy a happy new year.

### App structure

    shap_app/
        run_shap.py
        explainers.py
        adapters.py
        plotting.py
        config.py
        README.md

## How to run

**Step 1:** make sure your model is being output to a file, and that you have the `shap_app/` directory.

**Step 2:** pop this code into your notebook:

```
from shap_app.run_shap import run_shap

run_shap(
    model_path="model.pkl",
    data_path="X_test.npy",
    model_type="sklearn"
)
```

replacing the name of the models, data, etc with your own variables.

model_type currently accepts:

    sklearn
    xgboost
    tensorflow
    pytorch

This functionality can be found in `shap_app/adapters.py`

**(Optional) Step 3:** adjust shap background sample, it's currently set to 50 samples. It can be found in `shap_app/explainers.py`



### Fingers crossed that this resolves our versions issues!

- You run SHAP in your environment, so no more sklearn version issues.

- Since you load your own models, no more pickle incompatibility!

- I only receive SHAP outputs, SHAP values are just numpy arrays which are version‑agnostic

- The app works in marimo, jupyter, vscode, colab, whatever. It’s just Python.

- It works for sklearn, xgboost, tensorflow, pytorch, mytorch, etc because adapters isolate framework differences.

**Step 4:** send me the following output files:

    shap_values.npy
    feature_names.json
    summary_plot.png
    force_plot.html


**Final step (very important):** have a great new year!

### I'm livid that I need to dumb down my own written language so no one thinks I used a chatbot to write this

## I used to be paid to write documentation