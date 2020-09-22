if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, reticulate, tidyverse)
source(here("scripts", "venv_setup", "conda_setup.R"))

spacy_install()

setwd(here("scripts", "py_scripts"))

source_python(here("scripts", "py_scripts", "agent.py"))
