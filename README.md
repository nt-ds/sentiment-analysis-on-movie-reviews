# [sentiment-analysis-on-movie-reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/)

### Directory structure
--- main folders and files only ---

		sentiment-analysis-on-movie-reviews
			data (raw and processed)
				lemmed							(processed during modeling step - containing training data only)
					124,844 npy files, each with input dim of (1 x) 13916
				norm							(processed during modeling step - containing training data only)
					124,844 npy files, each with input dim of (1 x) 15226
				stemmed							(processed during modeling step - containing training data only)
					124,844 npy files, each with input dim of (1 x) 10989
				processed_movie_reviews.csv		(processed during processing step)
				train.tsv.zip					(retrieved)

			models (saved models and vectorizers)
				lemmed_ann_model.hdf5
				lemmed_ann_model.png
				lemmed_decision_tree_model.pickle
				lemmed_gru_model.hdf5
				lemmed_gru_model.png
				lemmed_logistic_model.pickle
				lemmed_lstm_model.hdf5
				lemmed_lstm_model.png
				lemmed_random_forest_model.pickle
				lemmed_rnn_model.hdf5
				lemmed_rnn_model.png
				lemmed_vectorizer.pickle
				norm_ann_model.hdf5
				norm_ann_model.png
				norm_decision_tree_model.pickle
				norm_gru_model.hdf5
				norm_gru_model.png
				norm_logistic_model.pickle
				norm_lstm_model.hdf5
				norm_lstm_model.png
				norm_random_forest_model.pickle
				norm_rnn_model.hdf5
				norm_rnn_model.png
				norm_vectorizer.pickle
				stemmed_ann_model.hdf5
				stemmed_ann_model.png
				stemmed_decision_tree_model.pickle
				stemmed_gru_model.hdf5
				stemmed_gru_model.png
				stemmed_logistic_model.pickle
				stemmed_lstm_model.hdf5
				stemmed_lstm_model.png
				stemmed_random_forest_model.pickle
				stemmed_rnn_model.hdf5
				stemmed_rnn_model.png
				stemmed_vectorizer.pickle

			notebooks (executable code)
				01_EDA.ipynb
				02_process_data.ipynb
				03_models_normal_data.ipynb
				04_models_stemmed_data.ipynb
				05_models_lemmed_data.ipynb
				06_evaluate_models_normal_data.ipynb
				07_evaluate_models_stemmed_data.ipynb
				08_evaluate_models_lemmed_data.ipynb

			src (source code)
				clean_data.py
				model.py
				my_class.py