.PHONY: all build clean

IMAGENAME=gomalizingflowjl

all: build

# Currently (march 2023), if you have docker compose installed, just configuring the default runtime may still not be enough.
# In addition to configuring the default runtime, you have to disable the default docker build kit, with:
# DOCKER_BUILDKIT=0 docker build <blah>
# See: https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime
build:
	-rm -rf .venv Manifest.toml
	DOCKER_BUILDKIT=0 docker build -t ${IMAGENAME} . --build-arg NB_UID=`id -u`
	docker-compose build
	docker-compose run --rm julia julia --project=/work -e 'using Pkg; Pkg.instantiate()'

test: build
	docker-compose run --rm julia julia --project=/work -e 'using Pkg; Pkg.test()'

clean:
	rm -f Manifest.toml
	rm -f playground/notebook/julia/*.ipynb
	rm -f playground/notebook/python/*.ipynb
	docker-compose down
