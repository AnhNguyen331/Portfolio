
library(readr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(knitr)
library(scales)
inventory <- read.csv("data/national_bridge_inventory_ne.csv")

douglas_bridges <- inventory %>%
  select(
    COUNTY_CODE_003,
    YEAR_BUILT_027,
    YEAR_RECONSTRUCTED_106,
    ADT_029,
    FUNCTIONAL_CLASS_026,
    DECK_COND_058,
    SUPERSTRUCTURE_COND_059,
    SUBSTRUCTURE_COND_060,
    LOWEST_RATING,
    BRIDGE_CONDITION,
    STRUCTURE_LEN_MT_049,
    DECK_AREA
  ) %>%
  filter(COUNTY_CODE_003 == 55) %>%
  mutate(
    YEAR_BUILT_027 = as.numeric(YEAR_BUILT_027),
    ADT_029 = as.numeric(ADT_029),
    DECK_COND_058 = as.numeric(DECK_COND_058),
    SUPERSTRUCTURE_COND_059 = as.numeric(SUPERSTRUCTURE_COND_059),
    SUBSTRUCTURE_COND_060 = as.numeric(SUBSTRUCTURE_COND_060),
    bridge_age = 2026 - YEAR_BUILT_027
  ) %>%
  drop_na(ADT_029, bridge_age, DECK_COND_058, SUPERSTRUCTURE_COND_059, SUBSTRUCTURE_COND_060)

douglas_bridges <- douglas_bridges %>%
  mutate(
    risk_group = ifelse(
      DECK_COND_058 <= 4 |
        SUPERSTRUCTURE_COND_059 <= 4 |
        SUBSTRUCTURE_COND_060 <= 4,
      "High Risk",
      "Lower Risk"
    )
  )

n_bridges <- nrow(douglas_bridges)

adt_min <- min(douglas_bridges$ADT_029, na.rm = TRUE)
adt_max <- max(douglas_bridges$ADT_029, na.rm = TRUE)
adt_median <- median(douglas_bridges$ADT_029, na.rm = TRUE)
adt_mean <- mean(douglas_bridges$ADT_029, na.rm = TRUE)

age_min <- min(douglas_bridges$bridge_age, na.rm = TRUE)
age_max <- max(douglas_bridges$bridge_age, na.rm = TRUE)
age_median <- median(douglas_bridges$bridge_age, na.rm = TRUE)
age_mean <- mean(douglas_bridges$bridge_age, na.rm = TRUE)

deck_mean <- mean(douglas_bridges$DECK_COND_058, na.rm = TRUE)
super_mean <- mean(douglas_bridges$SUPERSTRUCTURE_COND_059, na.rm = TRUE)
sub_mean <- mean(douglas_bridges$SUBSTRUCTURE_COND_060, na.rm = TRUE)


## Study Sample Characteristics

#| tbl-cap: "Descriptive statistics for traffic volume and bridge age among analyzed Douglas County bridges"

summary_table <- tibble(
  Variable = c("Daily Traffic Mean", "Bridge Age (years)"),
  Min = c(adt_min, age_min),
  Median = c(comma(adt_median),comma(age_median)),
  Mean = c(comma(adt_mean),comma(age_mean)),
  Max = c(comma(adt_max),comma(age_max))
)
kable(summary_table, booktabs = TRUE, digits = 1,
      format.args = list(big.mark = ","))


## Structural Condition  



#| fig-cap: "Distribution of Bridge Deck Condition Ratings in Douglas County, Nebraska."
#| label: fig-dist-deck-condition

cor_age_deck <- cor.test(
  douglas_bridges$bridge_age,
  douglas_bridges$DECK_COND_058,
  method = "spearman"
)

rho_age_deck <- unname(cor_age_deck$estimate)
p_age_deck <- cor_age_deck$p.value

douglas_bridges %>%
  mutate(DECK_COND_058 = factor(DECK_COND_058)) %>%
  ggplot(aes(x = DECK_COND_058,
             fill = DECK_COND_058 %in% as.character(0:4))) +
  geom_bar() +
  scale_fill_manual(values = c("FALSE" = "steelblue", "TRUE" = "darkorange"),
                    labels = c("FALSE" = "Rating ≥ 5", "TRUE" = "Rating ≤ 4 (Higher Risk)"),
                    name   = NULL) +
  labs(
    x = "Deck Condition Rating",
    y = "Number of Bridges"
  ) +
  theme_minimal()

## Traffic Exposure

#| label: fig-adt-distribution
#| fig-cap: "Distribution of Average Daily Traffic (ADT) for bridges in Douglas County, plotted on a logarithmic scale to accommodate the wide range of values. Most bridges carry between 1,000 and 100,000 vehicles per day."

cor_deck_adt <- cor.test(
  douglas_bridges$DECK_COND_058,
  douglas_bridges$ADT_029,
  method = "spearman"
)

rho_deck_adt <- unname(cor_deck_adt$estimate)
p_deck_adt <- cor_deck_adt$p.value

ggplot(douglas_bridges, aes(x = ADT_029)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "white") +
  scale_x_log10(labels = comma) +
  labs(
    x = "Average Daily Traffic (log scale)",
    y = "Number of Bridges"
  ) +
  theme_minimal()

risk_summary <- douglas_bridges %>%
  group_by(risk_group) %>%
  summarise(
    n = n(),
    mean_adt = mean(ADT_029, na.rm = TRUE),
    median_adt = median(ADT_029, na.rm = TRUE),
    mean_age = mean(bridge_age, na.rm = TRUE),
    .groups = "drop"
  )

mean_adt_high <- risk_summary %>%
  filter(risk_group == "High Risk") %>%
  pull(mean_adt)

mean_adt_low <- risk_summary %>%
  filter(risk_group == "Lower Risk") %>%
  pull(mean_adt)

mean_age_high <- risk_summary %>% 
  filter(risk_group == "High Risk")  %>% 
  pull(mean_age)

mean_age_low  <- risk_summary %>% 
  filter(risk_group == "Lower Risk") %>% 
  pull(mean_age)

n_high <- risk_summary %>%
  filter(risk_group == "High Risk") %>%
  pull(n)

n_low <- risk_summary %>%
  filter(risk_group == "Lower Risk") %>%
  pull(n)

## Traffic by Bridge Condition

#| echo: false
#| label: fig-condition-traffic
#| fig-cap: "Relationship between bridge structural condition and traffic exposure in Douglas County, Nebraska."

ggplot(douglas_bridges,
       aes(x = factor(DECK_COND_058), y = ADT_029,
           fill = DECK_COND_058 <= 4)) +
  geom_boxplot() +
  scale_y_log10(labels = comma) +
  scale_fill_manual(values = c("FALSE" = "steelblue", "TRUE" = "darkorange"),
                    labels = c("FALSE" = "Rating ≥ 5", "TRUE"  = "Rating ≤ 4 (Higher Risk)"),
                    name   = NULL) +
  labs(
    x = "Deck Condition Rating (0 = Failed, 9 = Excellent)",
    y = "Average Daily Traffic (log scale)"
  ) +
  theme_minimal()

