FROM python:3.8-slim-buster AS base
ARG TARGET=dev

LABEL author="LINKS Foundation"
LABEL maintainer="LINKS Foundation"

WORKDIR /$PROJECT_NAME

# copy and install requirements
COPY requirements/ requirements/
RUN pip install --upgrade pip
RUN cd requirements && python3 -m pip install -r requirements.$TARGET.txt --no-cache-dir

# copy main library files
COPY setup.py config.json ./
COPY LICENSE README.md ./
COPY settings/ settings/
COPY phenoai/ phenoai/
RUN pip install -e .

# copy the .env file
COPY env/$TARGET.env ./.env
COPY model_weights/ /mnt/model_weights

RUN mkdir /mnt/data
RUN mkdir /mnt/data/pheno_phases
RUN mkdir /mnt/data/input_data

# copy the main entrypoint and make it executable
COPY containers/webserver/docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]