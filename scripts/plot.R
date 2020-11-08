if (!require(pacman)) install.packages("pacman")
pacman::p_load(arrow, here, reticulate, tidyverse, bit64, tsibble, slider, patchwork, ggalluvial)
pacman::p_load_gh("cttobin/ggthemr")
source(here("scripts", "venv_setup", "conda_setup.R"))

spacy_install()

setwd(here("scripts", "py_scripts"))

source_python(here("scripts", "py_scripts", "agent.py"))

df <- read_parquet(here("Data", "softmax_experiment.parquet"))

df_cleaned <- mutate(df,
                     # john = paste0(states_so_far, choice),
                     across(where(is.character), as.factor),
                     across(where(is.integer64), as.numeric),
                     # across(john, as.numeric),
                     across(c(trial, step), ~ . + 1))

ggthemr("dust", layout = "clean")

df_cleaned %>%
  mutate(across(imm_reward, lag)) %>%
  filter(step == 1) %>%
  mutate(repetition = ifelse(choice == lag(choice), 1, 0)) %>%
  filter(trial > 1) %>%
  mutate(across(outcome_freq, ~ifelse(. == .7, "Common", "Rare")),
         across(imm_reward, ~ifelse(. == 1, "Rewarded", "Unrewarded"))) %>%
  group_by(outcome_freq, imm_reward) %>%
  summarise(across(repetition, mean)) %>%
  ggplot(aes(imm_reward, repetition, fill = outcome_freq)) +
  geom_bar(stat='identity', position = "dodge")

# confirmed that the Q is updating like it should (at least when w = 0)
df_condensed <- df_cleaned %>%
  select(
    trial,
    states_so_far,
    choice,
    step,
    outcome_freq,
    imm_reward,
    `Q(((),), left)`,
    `Qtd(((),), left)`,
    `Qmb(((),), left)`,
    `Q(((),), right)`,
    `Qtd(((),), right)`,
    `Qmb(((),), right)`,
    `Q((((),), 'left'), left)`,
    `Qtd((((),), 'left'), left)`,
    `Qmb((((),), 'left'), left)`,
    `Q((((),), 'left'), right)`,
    `Qtd((((),), 'left'), right)`,
    `Qmb((((),), 'left'), right)`,
    `Q((((),), 'right'), right)`,
    `Qtd((((),), 'right'), right)`,
    `Qmb((((),), 'right'), right)`,
    `Q((((),), 'right'), left)`,
    `Qtd((((),), 'right'), left)`,
    `Qmb((((),), 'right'), left)`,
    `P((((),), 'left', 'left'), 1)`,
    `P((((),), 'left', 'right'), 1)`,
    `P((((),), 'right', 'left'), 1)`,
    `P((((),), 'right', 'right'), 1)`) %>%
  mutate(Q_left = ifelse(states_so_far == "((),)",
                         `Q(((),), left)`,
                         ifelse(states_so_far == "((), 'left')",
                                `Q((((),), 'left'), left)`,
                                `Q((((),), 'right'), left)`)),
         Q_right =ifelse(states_so_far == "((),)",
                        `Q(((),), right)`,
                        ifelse(states_so_far == "((), 'right')",
                               `Q((((),), 'left'), right)`,
                               `Q((((),), 'right'), right)`)),
         Q_bigger = ifelse(Q_left > Q_right, 'left', 'right'),
         Q_diff = abs(Q_right - Q_left),
         Bigger_selected = ifelse(Q_bigger == choice, TRUE, FALSE),
         subsequent_reward = lead(imm_reward)) %>%
  group_by(step) %>%
  mutate(repetition = ifelse(choice == lag(choice), 1, 0)) %>%
  ungroup()

  # mutate(repetition = ifelse(choice == lag(choice), 1, 0)) %>%
  #   filter(trial > 1) %>%
  #   mutate(across(outcome_chance, ~ifelse(. == .7, "Common", "Rare")),
  #          across(imm_reward, ~ifelse(. == 1, "Rewarded", "Unrewarded"))) %>%
  #   group_by(outcome_chance, imm_reward) %>%
  #   summarise(across(repetition, mean))

# Ppl pick the higher value option more option, about 69% of the time in step 1 and 63% in step 2
df_condensed %>%
  group_by(step) %>%
  summarise(mean(Bigger_selected))

# Ppl get rewarded only 52-53%? So they're performing chance? Need to investigate more
df_cleaned %>%
  filter(step == 2) %>%
  summarise(mean(imm_reward))


# Confirmed so far that at least the transitions are working as they should- 70% of time ppl are arriving at the more likely next location
df_cleaned %>%
    filter(step == 1) %>%
    count(outcome_freq)

# average reward probability for any given outcome is weirdly .49, not .5
df_cleaned %>%
  filter(step == 2) %>%
  select(ends_with("1)")) %>%
  summarise_all(mean)


# How well does reward probability translate to reward? spot on
toasty %>%
  filter(Reward_dots > 0) %>%
  mutate(across(Reward_dots, ~round(., 2))) %>%
  group_by(Reward_dots) %>%
  summarise(y_val = mean(imm_reward)) %>%
  ggplot(aes(Reward_dots, y_val)) +
  geom_point()


# We established above that the average reward per trial is right around 50%; presumably, then, half the trials have an associated reward_prob less than .5? it's a little more- .54
toasty %>%
  filter(Reward_dots > 0) %>%
  summarise(mean(Reward_Probability))





custom_graphing <- list(
  geom_line(key_glyph = "timeseries"),
  scale_x_continuous(limits=c(0, 500)),
  theme_minimal(),
  theme(text = element_text(size=14, family = "Verdana", color = "#F1AE86")),
  theme(plot.title = element_text(size=22, family = "Verdana", face="bold")), 
  theme(strip.background = element_rect(color="#7A7676", fill="#FDF7C0", size=0.5, linetype="solid")),
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5),"cm")),
  theme(axis.text = element_text(colour = "#667682")),
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic")),
  theme(plot.background = element_rect(fill = "azure1"))
)


df_condensed %>%
  filter(step == 1) %>%
  pivot_longer(cols=c(`Q(((),), left)`,
                      `Q(((),), right)`),
               values_to = "Q_Val") %>%
  ggplot(aes(trial, Q_Val, colour = name, fill = subsequent_reward)) +
  geom_tile(aes(y = .4, width=.1),
            stat='identity',
            color = "azure1",
            show.legend = FALSE) +
  scale_fill_gradient(low="#09dbdb", high="azure1") +
  scale_color_manual(values=c("#F1AE86", "#667682")) +
  custom_graphing +
  scale_x_continuous(limits=c(1900, 2000)) +
  labs(title = "Q Value and No Rewards during Stage 1") +
  guides(col = guide_legend(order = 1, title="Transition 1"))



toasty <- df_cleaned %>%
  filter(step == 2) %>%
  mutate(across(updated_state, ~ str_remove_all(., "[\\(]|[\\)]|,|'")),
         across(outcome_freq, ~ifelse(. == .7, "Common", "Rare")),
         imm_reward_discrete = ifelse(imm_reward == 1, "Rewarded", "Unrewarded"),
         across(picks, as.character),
         across(picks, as.factor)) %>%
  separate(updated_state, c(NA, "State_1", "State_2"), sep=" ") %>%
  pivot_longer(cols=ends_with("1)"),
               names_to = c("Imagined_States_1", "Imagined_States_2"),
               names_pattern = "((?<=').+)', '(.+?(?='))",
               values_to = "Reward_Probability") %>%
  mutate(Q_value = paste0("Q((((),), '",
                          Imagined_States_1, "'), ",
                          Imagined_States_2,
                          ")")) %>%
  mutate(
    Q_value = pmap_dbl(
      .l = .,
      .f = function(...){
        row <- c(...)
        row %>%
          pluck("Q_value") %>%
          pluck(row, .) %>%
          as.double
      }
    ),
    Reward_dots = ifelse(State_1 == Imagined_States_1 & State_2 == Imagined_States_2,
                    Reward_Probability, 0),
    Q_dots = ifelse(State_1 == Imagined_States_1 & State_2 == Imagined_States_2,
                         Q_value, 0))

ggthemr("grape", layout = "clean")


some_graphing_additions = list(
  theme_minimal(),
  theme(text = element_text(size=14, family = "Verdana", color = "#F1AE86")),
  theme(plot.title = element_text(size=22, family = "Verdana", face="bold")),
  theme(strip.background = element_rect(color="#7A7676", fill="#FDF7C0", size=0.5, linetype="solid")),
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5),"cm")),
  theme(axis.text = element_text(colour = "#667682")),
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic")),
  scale_fill_manual(values=c("Unrewarded"="red","Rewarded" = "green")),
  scale_x_continuous(limits=c(2800, 2900))
)

reward_plot <- toasty %>%
  ggplot(aes(trial, Reward_Probability)) +
  geom_line(key_glyph = "timeseries", aes(colour = Imagined_States_1, linetype = Imagined_States_2)) +
  some_graphing_additions +
  scale_y_continuous(limits=c(.25, .75)) +
  geom_point(aes(trial, Reward_dots, fill = imm_reward_discrete, colour = picks), size=1, shape=21, stroke=1) +
  labs(title = "Reward Probability Random Walk")


q_plot <- toasty %>%
  ggplot(aes(trial, Q_value)) +
  geom_line(key_glyph = "timeseries", aes(colour = Imagined_States_1, linetype = Imagined_States_2)) +
  some_graphing_additions +
  scale_y_continuous(limits=c(.001, 1)) +
  geom_point(aes(trial, Q_dots, fill = imm_reward_discrete, colour = picks), size=1, shape=21, stroke=1) +
  labs(title = "Second-Stage Q Values")

reward_plot + q_plot + plot_layout(guides = "collect") & theme(legend.position = 'bottom')




# why does updating the same q values on different trials by the same amount result in different values?
# I guess because it's not just about the final reward you get (and the update after step 2), but also there's an update just after the first step; so might make more sense to look at two values that were equal after step 1 and then
# see whether the update is the same; I'll do this in the block below
df_cleaned %>%
  filter(step == 1) %>%
  pivot_longer(cols=c(`Qtd(((),), left)`,
                      `Qtd(((),), right)`),
               values_to = "Q_Val") %>%
  ggplot(aes(trial, Q_Val, colour = name)) +
  geom_line(key_glyph = "timeseries") +
  scale_color_manual(values=c("#F1AE86", "#667682")) +
  scale_x_continuous(limits=c(0, 50), breaks = scales::pretty_breaks(n = 50)) +
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


# okay i guess it's not even that the changes are the same if you look at changes within a tial/ comparing changes after stage 1; the changes depend not as much re. whether there was a reward but rather how the prediction error changed, which depends on the stage 2 q value; but looking at stage 2 values only will be different because it won't depend on the prediction error from another stage; i'll do this in the subsequent code chunk
df_cleaned %>%
  mutate(trial = ifelse(step == 1, trial, trial + .5)) %>%
  # filter(`Qtd(((),), left)` < .4 & `Qtd(((),), left)` > .38) %>%
  pivot_longer(cols=c(`Qtd(((),), left)`,
                      `Qtd(((),), right)`),
               values_to = "Q_Val") %>%
  ggplot(aes(trial, Q_Val, colour = name)) +
  geom_line(key_glyph = "timeseries") +
  scale_color_manual(values=c("#F1AE86", "#667682")) +
  scale_x_continuous(limits=c(0, 50), breaks = scales::pretty_breaks(n = 50)) +
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


# looks like second stage q values update to the same value when they were both previously equal; also note, both from stage 2 and stage 1 (see above graph), q values are only changing when they're supposed to, which is a good thing (if their acton was not taken, they don't get updated, hence only one value per level can get updated per trial); so q values i guess are changing both when they should and by how much they should
df_cleaned %>%
  mutate(trial = ifelse(step == 1, trial, trial + .5)) %>%
  # arrange(`Qtd((((),), 'right'), left)`) %>% View
  # filter(`Qtd((((),), 'right'), left)` < .4 & `Qtd((((),), 'right'), left)` > .39) %>% View
  pivot_longer(cols=c(`Qtd((((),), 'right'), left)`,
                      `Qtd((((),), 'right'), right)`),
               values_to = "Q_Val") %>%
  ggplot(aes(trial, Q_Val, colour = name)) +
  geom_line(key_glyph = "timeseries") +
  scale_color_manual(values=c("#F1AE86", "#667682")) +
  # scale_x_continuous(limits=c(12280, 12462), breaks = scales::pretty_breaks(n = 95)) +
  scale_x_continuous(limits=c(0, 1000)) +
  theme_minimal() +
  theme(text = element_text(size=14, family = "Verdana", color = "#F1AE86")) +
  labs(title = "Q Values during Stage 2, Right State",
       caption = "Based on draws from a Gaussian\ndistibution with mean=0 and sd=.025", y="Reward Probability", x="Trial") +
  theme(axis.text = element_text(colour = "#667682")) + 
  theme(plot.title = element_text(size=22, family = "Verdana", face="bold")) + 
  theme(strip.background = element_rect(color="#7A7676", fill="#FDF7C0", size=0.5, linetype="solid")) +
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5),"cm")) +
  guides(col = guide_legend(order = 1, title="Transition 1"),
         linetype = guide_legend(order = 2, title="Transition 2")) +
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic")) + 
  theme(plot.background = element_rect(fill = "azure1"))





# ggthemr_reset()

df_cleaned %>%
  filter(step == 2) %>%
  filter(trial < 1300) %>%
  mutate(across(updated_state, ~ str_remove_all(., "[\\(]|[\\)]|,|'"))) %>%
  separate(updated_state, c(NA, "State_1", "State_2"), sep=" ") %>%
  pivot_longer(cols=ends_with("1)"),
               names_to = c("Imagined_States_1", "Imagined_States_2"),
               # names_pattern = "((?<=').+?(?=', '))((?<=', ').+?(?='))",
               names_pattern = "((?<=').+)', '(.+?(?='))",
               values_to = "Reward_Probability") %>%
  mutate(Q_value = paste0("Q((((),), '",
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
  fill_gaps(outcome_freq = 0,
            .full = TRUE) %>%
  update_tsibble(key = c(Imagined_States_1, Imagined_States_2),
                 validate = FALSE) %>%
  group_by_key() %>% # review whether this line should be commented out; waldo::compare indicates it makes some difference but not sure if for better or worse
  fill(c(Reward_Probability, Q_value), .direction = "downup") %>%
  group_by_key() %>%
  update_tsibble(key = c(State_1, State_2), validate = FALSE) %>%
  mutate(across(imm_reward, ~replace_na(., 1)),
         Percent_of_Choices = slide_dbl(outcome_freq,
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








# generate the random walk graph; this is outdated now, though, so I can delete it

df_cleaned %>%
  filter(step == 2,
         trial > 800,
         trial < 900) %>%
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





