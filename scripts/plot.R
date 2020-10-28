if (!require(pacman)) install.packages("pacman")
pacman::p_load(arrow, here, reticulate, tidyverse, bit64, tsibble, slider)
pacman::p_load_gh("cttobin/ggthemr")
source(here("scripts", "venv_setup", "conda_setup.R"))

spacy_install()

setwd(here("scripts", "py_scripts"))

source_python(here("scripts", "py_scripts", "agent.py"))

df <- read_parquet(here("Data", "softmax_experiment.parquet"))

df_cleaned <- mutate(df,
                     lead_reward = lead(imm_reward),
                     john = paste0(states_so_far, choice),
                     across(where(is.character), as.factor),
                     across(where(is.integer64), as.numeric),
                     across(john, as.numeric),
                     across(trial, ~ . + 1))

ggthemr("dust", layout = "clean")

# ggthemr_reset()

df_cleaned %>%
  filter(step == 1) %>%
  mutate(across(updated_state, ~ str_remove_all(., "[\\(]|[\\)]|,|'"))) %>%
  separate(updated_state, c(NA, "State_1", "State_2"), sep=" ") %>%
  pivot_longer(cols=ends_with("1)"),
               names_to = c("Imagined_States_1", "Imagined_States_2"),
               # names_pattern = "((?<=').+?(?=', '))((?<=', ').+?(?='))",
               names_pattern = "((?<=').+)', '(.+?(?='))",
               values_to = "Reward_Probability") %>%
  mutate(Q_value = paste0("Qtd((((),), '",
                          Imagined_States_1, "'), ",
                          Imagined_States_2,
                          ")")) %>%
  mutate(Q_value = pmap_chr(
    .l = .,
    .f = function(...){
      row <- c(...)
      row %>%
        pluck("Q_value") %>%
        pluck(row, .)
    }
  )) %>%
  mutate(across(Q_value, as.numeric)) %>%
  as_tsibble(trial, key = c(State_1,
                            State_2,
                            Imagined_States_1,
                            Imagined_States_2)) %>%
  fill_gaps(outcome_chance = 0,
            .full = TRUE) %>%
  update_tsibble(key = c(Imagined_States_1, Imagined_States_2),
                 validate = FALSE) %>%
  group_by_key() %>% # review whether this line should be commented out; waldo::compare indicates it makes some difference but not sure if for better or worse
  fill(c(Reward_Probability, Q_value), .direction = "downup") %>%
  group_by_key() %>%
  update_tsibble(key = c(State_1, State_2), validate = FALSE) %>%
  mutate(across(imm_reward, ~replace_na(., 1)),
         Percent_of_Choices = slide_dbl(outcome_chance,
                                        mean,
                                        .before=20,
                                        .after=20),
         across(imm_reward, ~slide_dbl(.,
                                       mean,
                                       .before=10,
                                       .after=10))) %>%
  filter(State_1 == Imagined_States_1,
         State_2 == Imagined_States_2) %>%
  pivot_longer(cols=c(Reward_Probability, Q_value, Percent_of_Choices),
               names_to = "v_names",
               values_to = "v_values",
               names_transform = list(v_names = ~str_replace_all(.x,
                                                                 "_",
                                                                 " "))) %>%
  rename_with(~str_replace_all(.x, "_", " "), starts_with("State_")) %>%
  ggplot(aes(trial, v_values, colour = v_names, fill = imm_reward)) +
  scale_x_continuous(limits=c(0, 1200)) +
  geom_tile(aes(y = .4, width=3),
            stat='identity',
            color = "#FBF7F3",
            show.legend = FALSE) +
  scale_fill_gradient(low="#09dbdb", high="#FBF7F3") +
  geom_line(key_glyph = "timeseries") +
  geom_line() +
  theme(text = element_text(size=14, family = "Verdana", color = "#F1AE86")) +
  facet_grid(vars(`State 1`), vars(`State 2`), labeller = label_both) +
  labs(title = "Reward Probability Random Walk at Final States",
       caption = "Based on draws from a Gaussian\ndistibution with mean=0 and sd=.025", y="", x="Trial") +
  theme(plot.title = element_text(size=22, family = "Verdana", face="bold")) +
  theme(strip.background = element_rect(color="#7A7676", fill="#FBF7F3", size=0.5, linetype="solid")) +
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5),"cm")) +
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic"))# +
  theme(plot.background = element_rect(fill = "azure1"))
  
  
  
  
  ggplot(aes(trial, v_values, colour = v_names, fill = imm_reward)) +
  scale_x_continuous(limits=c(0, 1200)) +
  geom_tile(aes(y = .35, width=3),
            stat='identity',
            # color = "#FBF7F3",
            show.legend = FALSE) +
  # scale_fill_gradient(low="#7A6752", high="#FBF7F3") +
  geom_line(key_glyph = "timeseries") +
  geom_line() +
  # theme_minimal() +
  theme(text = element_text(size=14, family = "Verdana")) +
  facet_grid(vars(`State 1`), vars(`State 2`), labeller = label_both) +
  labs(title = "Reward Probability Random Walk at Final States",
       caption = "Based on draws from a Gaussian\ndistibution with mean=0 and sd=.025", y="", x="Trial") +
  theme(strip.background = element_rect(color="#7A7676", fill="#FDF7C0", size=0.5, linetype="solid")) +
  theme(plot.title = element_text(size=22, family = "Verdana", face="bold")) +
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5),"cm")) +
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic"))








# generate the random walk graph

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
       caption = "Based on draws from a Gaussian\ndistibution with mean=0 and sd=.025", y="Reward Probability", x="Trial") +
  theme(axis.text = element_text(colour = "#667682")) + 
  theme(plot.title = element_text(size=22, family = "Verdana", face="bold")) + 
  theme(strip.background = element_rect(color="#7A7676", fill="#FDF7C0", size=0.5, linetype="solid")) +
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5),"cm")) +
  guides(col = guide_legend(order = 1, title="Transition 1"),
         linetype = guide_legend(order = 2, title="Transition 2")) +
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic")) + 
  theme(plot.background = element_rect(fill = "azure1"))



