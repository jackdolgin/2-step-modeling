df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "left", TRUE, FALSE)) %>% filter(tong | lag(tong) | lag(tong, 2) | lag(tong, 3) | lag(tong, 4)) %>% View()








df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))



df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))



df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))



df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))














df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))



df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))



df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))



df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))









df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward, repetition) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))


df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward, repetition) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))


df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward, repetition) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))


df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward, repetition) %>%
  summarise(across(subsequent_Q, ~mean(., na.rm = TRUE)))








df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(c(subsequent_Q, repetition), ~mean(., na.rm = TRUE)))







df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(c(subsequent_Q, repetition), ~mean(., na.rm = TRUE)))

df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .3 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(c(subsequent_Q, repetition), ~mean(., na.rm = TRUE)))

df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "left", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), left)`, 2) - `Q(((),), left)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(c(subsequent_Q, repetition), ~mean(., na.rm = TRUE)))

df_condensed %>% mutate(tong = ifelse(round(outcome_freq, 2) == .7 & choice == "right", TRUE, FALSE),
                        subsequent_reward = lead(imm_reward),
                        subsequent_Q = lead(`Q(((),), right)`, 2) - `Q(((),), right)`,
                        repetition = ifelse(choice == lead(choice, 2), 1, 0)) %>%
  filter(tong) %>%
  group_by(subsequent_reward) %>%
  summarise(across(c(subsequent_Q, repetition), ~mean(., na.rm = TRUE)))

