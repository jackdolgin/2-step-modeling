if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, reticulate, tidyverse)
source(here("scripts", "venv_setup", "conda_setup.R"))

spacy_install()

source_python("agent.py")

