# CorrClim

A **toolbox** for facilitating climatic correction of timeseries and the implementation of new models.

WIP ..
<!-- 
## Table of Contents

1. **Installation**
   - [How to Install CorrClim from GitHub](#installation)
   - [New features of the latest release](#see-whats-new)
2. **Architecture**
   - [Overview and Link to Detailed Architecture](#architecture)
3. **Global workflow**
   - [Global workflow using the CorrClim objects](#global-objects-workflow)
4. **Usage Guide**
   - [Key Concepts and Object Descriptions](#how-to-use-it)
6. **Examples**
   - [Accessing Example Notebooks](#examples)
7. **Contribution**
   - [How to Contribute to CorrClim](#contribution)
   - [Guidelines and Code of Conduct](#contribution)
8. **Authors**
   - [Contact Information](#authors)
  

## Installation

```bash 
git clone https://github.com/lucarammel/pycorrclim.git
``` -->
<!-- 
## See what's new 

[Here](/CHANGELOG.md) find the new features of the latest release !

## Architecture

Find [here](/docs/architecture.md) the architecture of the repo to make it clear for you what's happening here !  -->
<!-- 
## How to use it

#### Concepts

The **CorrClim** package is built on multiple objects. It enables to keep the same data format, standardizes operations and objects used.

- [`TimeseriesDT`](corrclim/R/timeseries_dt.py) : This class is designed to handle time series data, which is fundamental in climate analysis. It would encapsulate methods for importing, processing, and managing time series data, likely making extensive use of R's data.table for efficient data manipulation.
- [`Operator`](corrclim/R/operator.py) : This class would define operators used in the climate correction process. Operators could be mathematical (e.g., addition, multiplication) or more complex analytical operations
- [`Smoother`](corrclim/R/smoother.py) : The Smoother class would specialize in data smoothing techniques, which are crucial for preparing climate data for analysis and correction.
- [`TimeseriesModel`](corrclim/R/timeseries_model.py) : This class would be dedicated to modeling time series data, particularly for forecasting or identifying trends within climate data. 
- [`ClimaticCorrector`](corrclim/R/climatic_corrector.py) : Finally, the ClimaticCorrector class would act as the orchestrator of the climate correction process, utilizing the other classes to perform comprehensive climate data correction. -->
<!-- 
## Global objects workflow 

<p align="center">
  <img src="/docs/objects_workflow.png" alt="objects_worfklos" title="objects_worklow" width="500"/>
</p>

## Examples

* Find **model** and **smoothers** usage example [here](/notebooks/models_smoothers.Rmd).
* Find **TimeseriesDT** object usage [here](/notebooks/timeseriesDT.Rmd)
* Find **metrics** usage examples [here](/notebooks/metrics.Rmd)
* Find **API** usage examples [here](/notebooks/api.Rmd) -->


## Contribution

Anyone can contribute to **CorrClim**. Add your own model, smoother or metrics following the [guidelines](/CONTRIBUTING.md) and the [recommandations](/CODE_OF_CONDUCT.md) to maintain the code.
<!-- 
## Authors

* [Lucas PEREIRA](mailto:lucas.pereira@artelys.com) - *Main developper* -  **Artelys**
* [Arthur Bossavy](mailto:arthur.bossavy@artelys.com) - *Developper & superviser* -  **Artelys**
* [Guillaume HOFMANN](mailto:guillaume.hofmann@edf.fr) - *Modelling* - **EDF**
 -->
