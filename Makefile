run:
	@echo "Running model $$(git rev-parse --short HEAD)"
	@echo "====================="
	python train.py $$(git rev-parse --short HEAD)

clean:
	rm -rf log/ summaries/ model/
