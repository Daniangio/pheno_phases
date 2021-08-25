# README #

### Deployment ###

The repository is production-ready and can be deployed following these easy steps:

1. [Install Docker Compose](https://docs.docker.com/compose/install/)
2. Clone this repository
3. Create `env/prod.env` file copying from [this template file](env/dev.env) and filling fields with proper values
4. Run command `sh manage.sh build prod` to build docker images
5. Run command `sh manage.sh run prod -d` to run containers and detach them from console
6. Done! You can visit the running webserver at http://localhost:7500/docs

Any customization may slightly change the deployment, we suggest to keep the [manage.sh](manage.sh) script updated to make the deployment as easy as possible.

### API Description ###

APIs are contained in the `phenoai/api` folder, here's a brief description of their purpose:

* `GET /data/places` endpoint returns the information about the existent places, weather stations and varieties , as they are configured in the [configuraiton](config.json) file
* `GET /data/phases/{place}/{variety}` endpoint returns a dataframe with the registered phenological phase changes of the specified place and variety.
  !!!ATTENTION!!! The APIs that create this dataframe have not yet been implemented!!!
* `GET /data/input/{place}/{variety}/{year}` endpoint returns dataframe with the data to be given in input to the DL model(s) for performing the inference. If the dataframe does
  not exist, a 404 error is raised. The dataframes are saved as csv files on a shared volume, one for each combination of place, variety and year (e.g. a single dataframe contains
  the input data for the place "mastroberardino", variety "aglianico" and year "2020")
* `PUT /data/input/{place}/{variety}/{year}` endpoint updates the previously mentioned dataframe, requesting data to external services (e.g. weather station data from the VITIGEOSS
  Dashboard APIs). This endpoint should be called periodically, to keep the input data updated during the year, having more precise inferences.
* `PUT /inference/{place}/{variety}/{year}` endpoint performs the actual inference, loading the model and its weights (which are stored in [this](model_weights) folder) and using
  the input dataframe uniquely identified by the place, variety and year parameters. The output is a json file with the phenological phases and their predicted corresponding starting
  dates

### Running tests ###

Tests are contained in the `test/` folder. In order to run tests, just type `sh manage.sh test`. A testing container is built and then pytest is run inside it.
Tests need to be maintained and new tests need to be added. The current version has just a couple of them.