if (!require(pacman)) install.packages("pacman")
pacman::p_load(arrow, here, reticulate, tidyverse, bit64)
source(here("scripts", "venv_setup", "conda_setup.R"))

spacy_install()

setwd(here("scripts", "py_scripts"))

source_python(here("scripts", "py_scripts", "agent_my_attempt_v2.py"))

df <- read_parquet(here("Data", "softmax_experiment.parquet"))

donuts <- df %>%
  filter(step == 0, trial < 150) %>%
  mutate(john = paste0(past_states, choice)) %>%
  mutate(across(where(is.character), as.factor),
         across(where(is.integer64), as.numeric),
         across(john, as.numeric))

donuts %>%
  ggplot(aes(x = trial, y = john)) +
  geom_line() +
  geom_rug(sides = 'top', aes(colour = imm_reward))
