VENV = titanic

activate:
	$(VENV)\Scripts\activate

init:
	pip install -r requirements.txt

format: activate
	@black .

test: activate
	pytest

word-embedding:
	python -m spacy download en_core_web_sm
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip
	unzip glove.840B.300d.zip

build-image: init
	docker build --tag $(VENV) --no-cache .

run-container: build-image
	docker run -d -it -v $(CURDIR)/Docker:/Docker/mount --name $(VENV) --rm $(VENV)

run-script: run-container
	docker exec -it $(VENV) bash -c "python imdb_sentiment_analysis.py; mv submission.csv ./mount"
	docker stop $(VENV)