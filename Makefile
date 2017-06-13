clean:
	rm -rf log/ summaries/ model/

run:
	python train.py $$(git rev-parse --short HEAD)
