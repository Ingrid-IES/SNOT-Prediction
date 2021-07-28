# Load packages -----------------------------------------------------------
library(table1) #table1
library(gtsummary) #tbl_summary
library(caret) #train
library(tidyverse)
library(magrittr) #%>%
library(missForest) #missForest
library(flextable)
library(ggthemes)



# Read in data ------------------------------------------------------------
fulldat <- read.csv("2R01 dataset for outcomes azure SF6D.csv", na.strings=c("", " "))
dim(fulldat)
# [1] 791  50

#Apply exclusion criteria (TREATMENT == "Sinus surgery")
dat <- subset(fulldat, TREATMENT == "Sinus surgery") 
dim(dat)
# [1] 604  50

# Modify and create variables  --------------------------------------------
#Outcome variables 
dat$SNOT22Change_cts <- dat$SNOT22_FU_TOTAL_6MONTH - dat$SNOT22_BLN_TOTAL
dat$SNOT22Change_bin <- ifelse(dat$SNOT22Change_cts <= -9, "Yes", "No")
dat$SF6DChange_cts <- dat$SF6D_SCORE_6MONTH - dat$SF6D_HUV_BLN
dat$SF6DChange_bin <- ifelse(dat$SF6DChange_cts >= 0.03, "Yes", "No")

#Education
# Fix issue with "22" and Missing values
# Recode as 0-12 up to highschool, 13-16 college, >=17 some post graduate degree
dat$EDUCATION_rec <- ifelse(dat$EDUCATION%in%c("20+","22"), "20", dat$EDUCATION)
dat$EDUCATION_rec <- as.numeric(dat$EDUCATION)
dat$EDUCATION_rec <- cut(dat$EDUCATION_rec, breaks=c(0, 12, 17, Inf), include.lowest=TRUE, right=TRUE)

pdat <- dat %>% 
  mutate(
    RACE = factor(RACE),
    RACE_rec = fct_recode(RACE, "Other" = "American Indian/Alaska Native", "Other" = "Native Hawaiian/Pacific Islander"), 
    HOUSEHOLD_INCOME = factor(HOUSEHOLD_INCOME, levels=c("0-25000","26000-50000","51000-75000","76000-100000","100000+")),
    PREVIOUS_SURGERY_rec = fct_recode(PREVIOUS_SURGERY,
                                      "1-2" = "1",
                                      "1-2" = "2",
                                      "3-5" = "3",
                                      "3-5" = "4",
                                      "3-5" = "5",
                                      "6+" = "6+"),
    SMOKER_rec= ifelse(SMOKER==0,0,1),
    ALCOHOL_rec=ifelse(ALCOHOL==0,0,1),
    BSIT_rec = factor(BSIT_DIAGNOSIS_string,levels=c("Normosmia","Abnormal olfaction"),labels=c("No","Yes")),
    IMMUNODEF_rec=factor(ifelse(!is.na(IMMUNODEF)&IMMUNODEF!=0,"Yes","No")),
    CILIARY_rec=factor(ifelse(CILIARY=="Cystic Fibrosis"|CILIARY=="Other","Yes","No")),
    AUTOIMMUNE_rec=factor(ifelse(AUTOIMMUNE!="None","Yes","No")),
    STEROID_rec=factor(ifelse(STEROID!="None","Yes","No")),
    DIABETES_rec=factor(ifelse(DIABETES!="None","Yes","No")),
    INSURANCE_rec = fct_recode(INSURANCE,
                               "VA Benefits" = "VA Benefits / Tricare",
                               "Medicaid/State Assisted/Uninsured" = "Medicaid",
                               "Medicaid/State Assisted/Uninsured" = "State Assisted",
                               "Medicaid/State Assisted/Uninsured" = "None",
                               "Employer/Private" = "Employer provided",
                               "Employer/Private" = "Private"),
    ALLERGY_rec = ifelse(ALLERGY_HISTORY==1|ALLERGY_TESTING==1, 1, 0), 
    OSA_rec = ifelse(OSA_HISTORY==1|OSA_TESTING==1, 1, 0), 
    IMMUNODEF_rec = ifelse(is.na(IMMUNODEF)|IMMUNODEF==0, 0, 1)
  )

# binary 0/1 vars - 1 always equals yes
bin01vars=c("SEX","AFS","SEPT_DEV","CRS_POLYPS","RAS","HYPER_TURB","MUCOCELE","ASTHMA","ASA_INTOLERANCE",
            "COPD","DEPRESSION","FIBROMYALGIA",
            "SMOKER_rec","ALCOHOL_rec","GERD",
            "ALLERGY_rec","OSA_rec","IMMUNODEF_rec")

pdat %<>%  mutate_each_(funs(factor(.)), bin01vars)


qvars=c("Age","BLN_CT_TOTAL","BLN_ENDOSCOPY_TOTAL")


# ml predictors
mlpredictors=c("Age", "SEX", "RACE_rec", "ETHNICITY", "EDUCATION_rec", "HOUSEHOLD_INCOME", 
               "PREVIOUS_SURGERY_rec", "INSURANCE_rec", "AFS", "SEPT_DEV", "CRS_POLYPS", "RAS",
               "HYPER_TURB", "MUCOCELE", "ASTHMA", "ASA_INTOLERANCE", "ALLERGY_rec", "OSA_rec", 
               "IMMUNODEF_rec", "SMOKER_rec", "ALCOHOL_rec", "CILIARY_rec", "AUTOIMMUNE_rec", "STEROID_rec", 
               "DIABETES_rec", "GERD", "BLN_CT_TOTAL", "BLN_ENDOSCOPY_TOTAL", "BSIT_rec", "PREOP_STEROID_USE", "SNOT22_BLN_TOTAL") 

outcomes=c("SNOT22Change_cts","SNOT22Change_bin") #, "SF6DChange_cts", "SF6DChange_bin")

# don't use qol vars for now
qol_bln_surveys=c( #"SNOT22_BLN_TOTAL",
  "RSDI_PHYS_BLN_TOTAL","RSDI_FUNCT_BLN_TOTAL", "RSDI_EMOT_BLN_TOTAL" , "RSDI_BLN_TOTAL" ,"SF6D_HUV_BLN", "PHQ2_BLN_TOTAL")

# Table 1  ----------------------------------------------------------------
table1(~ . | SNOT22Change_bin, data=pdat[,]) 

table1(~ . | SNOT22Change_bin, data=pdat[,c(outcomes, mlpredictors)]) 

pdatcc <- subset(pdat, !is.na(pdat$SNOT22Change_bin))[,c(outcomes, mlpredictors)]

# tab1 <- pdatcc %>% 
#   tbl_summary(by = 'SNOT22Change_bin', missing="ifany",
#               type = list(all_dichotomous() ~ "categorical",all_of(qvars) ~ 'continuous'),
#               digits = list(all_continuous() ~ c(1, 1))) %>% 
#   add_n() %>%
#   add_p(pvalue_fun = function(x) ifelse(x<0.001, "<0.001",format(round(x,3),nsmall=3)), test = list(all_categorical() ~ "fisher.test")) %>%
#   add_overall() %>% 
#   bold_labels() %>% 
#   bold_p(t = 0.05, q = FALSE) %>% 
#   sort_p() %>% 
#   gtsummary::as_flex_table()