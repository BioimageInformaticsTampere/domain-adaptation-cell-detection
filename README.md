# domain-adaptation-cell-detection

Code for article "Iterative unsupervised domain adaptation for generalized cell detection from brightfield z-stacks"

To run the code, change datapath in config.json and then run the script do_everything.py. 
Change parameters in do_everything.py to train, adapt or only test the models. Default is testing only (requires weights).

We recommend using GPU for faster execution. For training the models, Geforce GTX 1060 (6GB) was used. 

In addition to basic python (version 3.5 or higher) packages, keras with Tensorflow backend is required. 

Data and annotations can temporarily be found at http://compbio.uta.fi/ruusuvuori/focus_data/ under licence CC BY-NC-SA.

