.PHONY: all build clean

DOCKER_IMAGE=gomalizingflowjl
# You can also set 11.7.0. For example
# Run "$ CUDA_VERSION=11.7.0 make" in your terminal
CUDA_VERSION?=12.0.0

all: build

build:
	-rm -rf .venv Manifest.toml
	#DOCKER_BUILDKIT=0 docker build -t ${IMAGENAME} . --build-arg NB_UID=`id -u`
	docker build -t ${DOCKER_IMAGE} . --build-arg NB_UID=`id -u` --build-arg CUDA_VERSION=${CUDA_VERSION}
	docker compose build
	docker compose run --rm shell julia --project=/work -e 'using Pkg; Pkg.instantiate()'
	docker compose run --rm shell rye sync

test: build
	docker compose run --rm shell julia --project=/work -e 'using Pkg; Pkg.test()'

clean:
	-rm -rf .venv Manifest.toml
	rm -f playground/notebook/julia/*.ipynb
	rm -f playground/notebook/python/*.ipynb
	docker compose down
