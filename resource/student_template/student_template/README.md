
# CSI 5155 Final Project Template.

This project has three phases, with a top level directory containing each.
They should each be entirely self-contained sharing the same top level requirements.txt.

Parts 1, 2, and 3 must all remain entirely separate and independent. Each should have their own set of source code, or notebook that is runnable.
Part 3 in particular must be submitted in the provided format so that it can be run on the heldout set. 
There are a set of tests to ensure the code remains compatible.

If you are developing your code in a notebook to begin with, please follow the conventions outlined in the provided source files.
Once you are done developing in the notebook, copy the relevant inference code into inference.py and check that everything runs using the provided tests.

Part3/inference.py is built around four functions:

- `make_model()` builds your model architecture. The placeholder is just a single linear layer, replace it with your own.
- `transform(image)` takes one PIL image and turns it into the tensor your model expects. Any resizing or normalising you do during training should happen here as well, so it matches at test time.
- `load_model(device)` builds the model with `make_model()` and loads the weights from Part3/weights.pt. You should not need to change this one.
- `predict(model, source, split, device)` loads the images with `load_dataset(source, split=split)` and returns a list of labels, 0 for real and 1 for AI-generated, one per image and in the same order as the dataset.
  When we run your submission, source will be a local folder of unlabelled images and split will be "train". You can also point it at the Hugging Face dataset to try it out.
  Do not change the signature or the assertions at the bottom.

Part3/train.py is a starting point for your training code. It imports `make_model` from inference.py so training and inference use the same architecture, and it saves your weights to Part3/weights.pt.
Make sure requirements.txt lists everything Part 3 needs to run, since that is what we install when evaluating.

When you are ready to submit, run this from the top level folder:

```
python test_submission.py --student-number 12345
```

It runs your predict function on 20 images from the test set and, if everything passes, creates submission_12345.zip for you to submit.

If you have any questions please refer to the assignment outline, documentation here, or email adefu020@uottawa.ca.

