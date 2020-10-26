if (!require(pacman)) install.packages("pacman")
pacman::p_load(arrow, here, reticulate, tidyverse, bit64)
source(here("scripts", "venv_setup", "conda_setup.R"))

spacy_install()

setwd(here("scripts", "py_scripts"))

source_python(here("scripts", "py_scripts", "agent.py"))

df <- read_parquet(here("Data", "softmax_experiment.parquet"))

df_cleaned <- mutate(df,
                     lead_reward = lead(imm_reward),
                     john = paste0(past_states, choice),
                     across(where(is.character), as.factor),
                     across(where(is.integer64), as.numeric),
                     across(john, as.numeric))

df_cleaned %>%
  filter(step == 1,
         trial < 300) %>%
  pivot_longer(cols=ends_with("1)"),
               names_to = c("State1", "State2"),
               # names_pattern = "((?<=').+?(?=', '))((?<=', ').+?(?='))",
               names_pattern = "((?<=').+)', '(.+?(?='))",
               values_to = "Reward_Probability") %>%
  ggplot(aes(trial, Reward_Probability, colour = State1, linetype = State2)) +
  geom_line(key_glyph = "timeseries") +
  scale_color_manual(values=c("#F1AE86", "#667682")) +
  theme_minimal() +
  theme(text = element_text(size=14, family = "Verdana", color = "#F1AE86")) +
  labs(title = "Reward Probability Random Walk at Final States",
       caption = 'Based on draws from\a Gaussian distibution\nwith mean=0 and sd=.025, y="Reward Probability", x="Trial") +
  theme(axis.text = element_text(colour = "#667682")) + 
  theme(plot.title = element_text(size=22, family = "Verdana", face="bold")) + 
  theme(strip.background = element_rect(color="#7A7676", fill="#FDF7C0", size=0.5, linetype="solid")) +
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5),"cm")) +
  guides(col = guide_legend(order = 1, title="Transition 1"),
         linetype = guide_legend(order = 2, title="Transition 2")) +
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic")) + 
  theme(plot.background = element_rect(fill = "azure1"))

  ggplot(aes(x = trial, y = john)) +
  geom_line() +
  geom_rug(sides = 'top', aes(colour = lead_reward))




df %>%
  filter(step == 1) %>%
  mutate(across(where(is.integer64), as.numeric)) %>%
  select(ends_with("1)"))
  group_by(across(c(updated_state, ends_with("1)")))) %>%
  summarise(mean(imm_reward)) %>%
  View
  # `Q((((),), 'right'), left)`
  
  
  