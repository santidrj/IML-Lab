.ONESHELL:

kmeans:

optics: validation/heart-c_optics_val.txt validation/penbased_optics_val.txt validation/connect4_optics_val.txt

validation/heart-c_optics_val.txt: datasets/processed/processed_heart-c.pkl
	cd w1/models
	python heart-c-optics.py

validation/penbased_optics_val.txt: datasets/processed/processed_pen-based.pkl
	cd w1/models
	python pen-based_optics.py

validation/connect4_optics_val.txt: datasets/processed/encoded_subset_connect4.pkl
	cd w1/models
	python connect4_optics.py

figures/heart-c_optics_val.txt: datasets/processed/processed_heart-c.pkl
	cd w1/models
	python heart-c-optics.py

validation/penbased_optics_val.txt: datasets/processed/processed_pen-based.pkl
	cd w1/models
	python pen-based_optics.py

validation/connect4_optics_val.txt: datasets/processed/encoded_subset_connect4.pkl
	cd w1/models
	python connect4_optics.py
