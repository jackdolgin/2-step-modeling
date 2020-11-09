if (!require(pacman)) install.packages("pacman")
pacman::p_load(arrow, here, reticulate, tidyverse, bit64, tsibble, slider,
               ggalluvial)
pacman::p_load_gh("cttobin/ggthemr")
source(here("scripts", "venv_setup", "conda_setup.R"))

spacy_install()

setwd(here("scripts", "py_scripts"))

source_python(here("scripts", "py_scripts", "agent.py"))

df <- read_parquet(here("Data", "softmax_experiment.parquet"))

df_cleaned <- mutate(df,
                     across(where(is.character), as.factor),
                     across(where(is.integer64), as.numeric),
                     across(c(trial, step), ~ . + 1))

ggthemr("dust", layout = "clean")


df_cleaned %>%
  group_by(trial) %>%
  mutate(across(imm_reward, max)) %>%
  group_by(step) %>%
  mutate(repetition = ifelse(choice == lead(choice), 1, 0)) %>%
  ungroup() %>%
  filter(row_number() < nrow(df_cleaned) - 1,
         step == 1) %>%
  mutate(across(outcome_freq, ~ifelse(round(., 1) == .7, "Common", "Rare")),
         across(imm_reward, ~ifelse(. == 1, "Rewarded", "Unrewarded"))) %>%
  group_by(outcome_freq, imm_reward) %>%
  summarise(across(repetition, mean)) %>%
  ggplot(aes(imm_reward, repetition, fill = outcome_freq)) +
  geom_bar(stat='identity', position = "dodge") +
  coord_cartesian(ylim = c(.1, .9))



first_qs <- colnames(df_cleaned) %>%
  str_subset(., "Q\\(\\(\\(\\),\\), \\w*\\)")
second_qs <- colnames(df_cleaned) %>%
  str_subset(., "Q\\(*\\),\\), '\\w*'\\), \\w*\\)")
rs <- select(df_cleaned, ends_with("1)")) %>% colnames

toasty1 <- df_cleaned %>%
  rowwise() %>%
  mutate(
    max_q_s_1 = all_of(first_qs)[which.max(c_across(all_of(first_qs)))],
    max_q_s_2 = all_of(second_qs)[which.max(c_across(all_of(second_qs)))],
    max_r = all_of(rs)[which.max(c_across(all_of(rs)))]
  ) %>%
  filter(step == 2) %>%
  mutate(across(updated_state, ~ str_remove_all(., "[\\(]|[\\)]|,|'")),
         across(max_q_s_1, ~ str_remove_all(., "Q|[\\(]|[\\)]|,|'| ")),
         across(max_q_s_2, ~ str_remove_all(., "Q|[\\(]|[\\)]|,|'" )),
         across(max_r, ~ str_remove_all(., "P|[\\(]|[\\)]|,|'|1")),
         across(picks, as.character),
         across(picks, as.factor)) %>%
  separate(updated_state, c(NA, "State_1", "State_2"), sep=" ") %>%
  separate(max_q_s_2, c(NA, "Q_Biggest_State_1", "Q_Biggest_State_2"), sep=" ") %>%
  separate(max_r, c(NA, "R_Biggest_State_1", "R_Biggest_State_2"), sep=" ") %>%
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
    ))



toasty1 %>%
  mutate(high_q_2_pt_1 = ifelse(picks == Q_Biggest_State_1, "Aha", "nah"),
         high_q_2_pt_2 = case_when(State_1 != Q_Biggest_State_1 ~ "Other",
                                   State_2 != Q_Biggest_State_2 ~ "Close",
                                   TRUE ~ "Biggest"),
         high_q_1 = ifelse(State_1 == max_q_s_1, TRUE, FALSE)) %>%
  count(high_q_2_pt_1, high_q_2_pt_2, high_q_1) %>%
  ggplot(aes(y = n, axis1 = high_q_2_pt_1, axis2 = high_q_2_pt_2)) +
  # stat_stratum(reverse = FALSE) +
  geom_alluvium(aes(fill=high_q_1)) +
  geom_stratum(width = 1/12, fill="black", color = "grey") +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  # scale_x_discrete(limits = c("Gender", "Dept"), expand = c(.05, .05)) +
  
  scale_fill_brewer(type = "qual", palette = "Set1") +
  theme_minimal()



toasty1 %>%
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
         Percent_of_Choices = slide_dbl(outcome_freq, mean,
                                        .before=20, .after=20),
         across(imm_reward, ~slide_dbl(., mean,
                                       .before=10, .after=10))) %>%
  filter(State_1 == Imagined_States_1,
         State_2 == Imagined_States_2) %>%
  pivot_longer(cols=c(Reward_Probability, Q_value, Percent_of_Choices),
               names_to = "v_names",
               values_to = "v_values",
               names_transform = list(
                 v_names = ~str_replace_all(.x, "_", " "))) %>%
  rename_with(~str_replace_all(.x, "_", " "), starts_with("State_")) %>%
  ggplot(aes(trial, v_values, colour = v_names, fill = imm_reward)) +
  scale_x_continuous(limits=c(0, 400)) +
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
  theme(plot.margin=unit(c(0.5,1.5,0.5,0.5), "cm")) +
  theme(plot.subtitle=element_text(size=16, family = "Verdana", face="italic"))
