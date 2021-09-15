.PHONY: all build clean

IMAGENAME=lftjl

all: build

build:
	-rm -rf .venv Manifest.toml
	docker build -t ${IMAGENAME} .
	docker-compose build
	docker-compose run --rm julia julia --project=/work -e 'using Pkg; Pkg.instantiate()'

clean:
	rm -f Manifest.toml
	docker-compose down
