amazon = pd.read_csv("amazon_cleaned.csv")
amazon = amazon.dropna(axis=0,how="any")
amazon = amazon.reset_index(drop=True)
amazon_text = amazon.loc[:,"reviewText"]

max_words = 50000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(amazon_text.tolist())
X_train = tokenizer.texts_to_sequences(amazon_text.tolist())
