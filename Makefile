.PHONY: all build clean

IMAGENAME=lftjl

all: build

build:
	-rm -rf .venv Manifest.toml
	docker build -t ${IMAGENAME} .
	docker-compose build
	docker-compose run --rm julia julia --project=/work -e 'using Pkg; Pkg.instantiate()'

test: build
	docker-compose run --rm julia julia --project=/work -e 'using Pkg; Pkg.test()'

clean:
	rm -f Manifest.toml
	rm -f playground/notebook/julia/*.ipynb
	rm -f playground/notebook/python/*.ipynb
	docker-compose down
