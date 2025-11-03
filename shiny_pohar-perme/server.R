library(shiny)
library(popEpi)
library(Epi)
library(survival)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(plotly)
library(readr)
library(DT)

server <- function(input, output, session) {

  logText <- reactiveVal("")
  addLog <- function(msg){
    old <- logText()
    logText(paste(old, paste0(Sys.time(), " - ", msg), sep="\n"))
  }
  output$log <- renderText({ logText() })

  # --- Chargement population hazard ---
  pop_haz <- reactive({
    if(!file.exists("pop_haz.csv")){
      addLog("Création du fichier pop_haz.csv...")
      mlt <- read.table("mltper_1x1.txt", header=TRUE, skip=2)
      flt <- read.table("fltper_1x1.txt", header=TRUE, skip=2)
      
      pop_haz_men <- expand.grid(age=0:110, per=1990:2023)
      pop_haz_men$haz <- mapply(function(a,y){ val <- mlt$mx[mlt$Age==a & mlt$Year==y]; ifelse(length(val)==0, NA, val)}, pop_haz_men$age, pop_haz_men$per)
      pop_haz_men$sex <- 1

      pop_haz_women <- expand.grid(age=0:110, per=1990:2023)
      pop_haz_women$haz <- mapply(function(a,y){ val <- flt$mx[flt$Age==a & flt$Year==y]; ifelse(length(val)==0, NA, val)}, pop_haz_women$age, pop_haz_women$per)
      pop_haz_women$sex <- 2

      pop_haz_df <- rbind(pop_haz_men, pop_haz_women)
      write.csv(pop_haz_df, "pop_haz.csv", row.names=FALSE)
      addLog("pop_haz.csv créé")
      pop_haz_df
    } else {
      addLog("pop_haz.csv trouvé et chargé")
      read.csv("pop_haz.csv")
    }
  })

  # --- Chargement fichier patients ---
  dataInput <- reactive({
    if(input$source=="auto"){
      path <- "/srv/uploads/hm.csv"
      if(!file.exists(path)){
        addLog("❌ Fichier hm.csv non trouvé")
        showModal(modalDialog(title="Fichier manquant",
                              "Le fichier hm.csv n’a pas été trouvé dans /srv/uploads.",
                              easyClose=TRUE,
                              footer = modalButton("Fermer")))
        return(NULL)
      }
      addLog("✅ Fichier hm.csv trouvé dans /srv/uploads")
      read.csv(path, sep=",")
    } else {
      req(input$hmfile)
      addLog("📁 Fichier hm.csv chargé manuellement")
      read.csv(input$hmfile$datapath, sep=",")
    }
  })

  # --- Préparation commune ---
  prepareData <- reactive({
    data <- dataInput()
    req(data)
    names(data)[names(data)=="age"] <- "age_diag"
    data$DateDuDiag <- as.Date(data$DateDuDiag)
    data$DateDernieresNouvelles <- as.Date(data$DateDernieresNouvelles)
    data$time <- as.numeric(data$DateDernieresNouvelles - data$DateDuDiag)
    data$status <- data$event
    data$age_days <- data$age_diag*365.25
    data$year_frac <- as.numeric(format(data$DateDuDiag,"%Y")) + (as.numeric(format(data$DateDuDiag,"%m"))-0.5)/12
    data$sex_char <- ifelse(data$sex==1,"male","female")
    
    quartiles <- quantile(data$age_diag, probs=seq(0,1,0.25), na.rm=TRUE)
    data_prep <- data %>%
      mutate(dg_date = DateDuDiag,
             ex_date = DateDernieresNouvelles,
             dg_age = age_diag,
             status_num = status,
             age_quartile = cut(age_diag, breaks=quartiles, include.lowest=TRUE,
                                labels=paste0(round(head(quartiles,-1)),"–",round(tail(quartiles,-1))," ans")))
    data_prep
  })

  # --- Fonction de lexpand ---
  runLex <- function(data){
    pop <- pop_haz() %>% rename(agegroup=age, year=per)
    lex <- lexpand(data, birth=data$DateDuDiag-(data$age_diag*365.25),
                   entry=data$DateDuDiag, exit=data$DateDernieresNouvelles,
                   status=data$status, pophaz=pop, pp=TRUE, fot=0:30)
    lex
  }

  # --- Analyse Global ---
  results_global <- eventReactive(input$run_global, {
    data <- prepareData()
    req(data)
    lex <- runLex(data)
    addLog("Calcul survie net Global...")
    fit <- survtab(Surv(time=fot, event=lex.Xst==1)~1, data=lex,
                   pophaz=pop_haz() %>% rename(agegroup=age, year=per),
                   surv.type="surv.rel", relsurv.method="pp", breaks=list(fot=seq(0,15,1)))
    df <- as.data.frame(summary(fit))
    df$surv.exp <- df$surv.obs/df$r.pp
    df$TSM <- (df$surv.exp/df$surv.obs-1)*100
    data_excess <- lex %>% group_by(fot) %>% summarise(py=sum(lex.dur), d=sum(lex.Xst==1),
                                                        haz_obs=d/py,
                                                        haz_exp=sum(pop.haz*lex.dur)/py,
                                                        excess=haz_obs-haz_exp)
    addLog("Global terminé")
    list(global=df, data_lex=lex, data_raw=data, excess=data_excess)
  })

  # --- Analyse par Sexe ---
  results_sex <- eventReactive(input$run_sex, {
    data <- prepareData()
    req(data)
    lex <- runLex(data)
    addLog("Calcul survie net par Sexe...")
    fit <- survtab(Surv(time=fot, event=lex.Xst==1)~sex, data=lex,
                   pophaz=pop_haz() %>% rename(agegroup=age, year=per),
                   surv.type="surv.rel", relsurv.method="pp", breaks=list(fot=seq(0,15,1)))
    df <- as.data.frame(summary(fit))
    df$surv.exp <- df$surv.obs/df$r.pp
    df$TSM <- (df$surv.exp/df$surv.obs-1)*100
    data_excess <- lex %>% group_by(fot, sex) %>% summarise(py=sum(lex.dur), d=sum(lex.Xst==1),
                                                           haz_obs=d/py,
                                                           haz_exp=sum(pop.haz*lex.dur)/py,
                                                           excess=haz_obs-haz_exp)
    addLog("Analyse par Sexe terminée")
    list(sex=df, data_excess=data_excess)
  })

  # --- Analyse par Âge+Sexe ---
  results_age_sex <- eventReactive(input$run_age_sex, {
    data <- prepareData()
    req(data)
    lex <- runLex(data)
    addLog("Calcul survie net par Âge+Sexe...")
    fit <- survtab(Surv(time=fot, event=lex.Xst==1)~sex+age_quartile, data=lex,
                   pophaz=pop_haz() %>% rename(agegroup=age, year=per),
                   surv.type="surv.rel", relsurv.method="pp", breaks=list(fot=seq(0,15,1)))
    df <- as.data.frame(summary(fit))
    df$surv.exp <- df$surv.obs/df$r.pp
    df$TSM <- (df$surv.exp/df$surv.obs-1)*100
    data_excess <- lex %>% group_by(fot, sex, age_quartile) %>% summarise(py=sum(lex.dur), d=sum(lex.Xst==1),
                                                                          haz_obs=d/py,
                                                                          haz_exp=sum(pop.haz*lex.dur)/py,
                                                                          excess=haz_obs-haz_exp)
    df_quartile_all <- df %>% filter(Tstop %in% c(1,5,10)) %>% group_by(age_quartile,Tstop) %>% summarise(surv.exp=mean(surv.exp, na.rm=TRUE), .groups="drop") %>% mutate(horizon=paste0(Tstop," ans"))
    df_quartile_male <- df %>% filter(sex==1, Tstop %in% c(1,5,10)) %>% select(age_quartile,Tstop,surv.exp) %>% mutate(horizon=paste0(Tstop," ans"))
    df_quartile_female <- df %>% filter(sex==2, Tstop %in% c(1,5,10)) %>% select(age_quartile,Tstop,surv.exp) %>% mutate(horizon=paste0(Tstop," ans"))
    addLog("Analyse Âge+Sexe terminée")
    list(age_sex=df, data_excess=data_excess,
         df_quartile_all=df_quartile_all,
         df_quartile_male=df_quartile_male,
         df_quartile_female=df_quartile_female)
  })

  # --- Table et graphique outputs ---
  output$summary_table <- DT::renderDataTable({
    req(results_global())
    data <- results_global()$data_raw
    df <- data.frame(
      Variable=c("Nombre total","Hommes","Femmes","Âge médian","Âge min","Âge max"),
      Valeur=c(nrow(data), sum(data$sex==1), sum(data$sex==2), round(median(data$age_diag),1),
               min(data$age_diag), max(data$age_diag))
    )
    datatable(df, options=list(dom="t"))
  })

  output$table_global <- DT::renderDataTable({ req(results_global()); datatable(results_global()$global) })
  output$table_sex <- DT::renderDataTable({ req(results_sex()); datatable(results_sex()$sex) })
  output$table_age_sex <- DT::renderDataTable({ req(results_age_sex()); datatable(results_age_sex()$age_sex) })
  output$lex_table <- DT::renderDataTable({ req(results_global()); datatable(results_global()$data_lex) })

  # --- Graphiques ---
  plot_surv <- function(df) ggplot(df,aes(x=Tstop)) +
    geom_line(aes(y=surv.obs,color="Observée"),linewidth=1.2) +
    geom_line(aes(y=r.pp,color="Relative"),linewidth=1.2) +
    geom_line(aes(y=surv.exp,color="Attendue"),linetype="dashed") +
    scale_color_manual(values=c("Observée"="red","Relative"="green","Attendue"="black")) +
    labs(x="Temps (années)",y="Survie") + theme_minimal()

  plot_TSM <- function(df) ggplot(df,aes(x=Tstop,y=TSM)) + geom_line(color="blue") + geom_hline(yintercept=0, linetype="dashed", color="red") + labs(y="TSM (%)", x="Temps") + theme_minimal()

  plot_excess <- function(df) ggplot(df,aes(x=fot,y=excess)) + geom_line() + labs(y="Excess mortality", x="Temps") + theme_minimal()

  plot_quartile <- function(df) ggplot(df,aes(x=age_quartile,y=surv.exp,color=horizon,group=horizon)) + geom_line() + geom_point() + labs(y="Survie nette", x="Quartile") + theme_minimal()

  output$plot_global <- renderPlotly({ req(results_global()); ggplotly(plot_surv(results_global()$global)) })
  output$plot_sex <- renderPlotly({ req(results_sex()); ggplotly(plot_surv(results_sex()$sex)) })
  output$plot_age_sex <- renderPlotly({ req(results_age_sex()); ggplotly(plot_surv(results_age_sex()$age_sex)) })

  output$plot_TSM_global <- renderPlotly({ req(results_global()); ggplotly(plot_TSM(results_global()$global)) })
  output$plot_TSM_sex <- renderPlotly({ req(results_sex()); ggplotly(plot_TSM(results_sex()$sex)) })
  output$plot_TSM_age_sex <- renderPlotly({ req(results_age_sex()); ggplotly(plot_TSM(results_age_sex()$age_sex)) })

  output$plot_excess <- renderPlotly({ req(results_global()); ggplotly(plot_excess(results_global()$excess)) })
  output$plot_excess_sex <- renderPlotly({ req(results_sex()); ggplotly(plot_excess(results_sex()$data_excess)) })
  output$plot_excess_age_sex <- renderPlotly({ req(results_age_sex()); ggplotly(plot_excess(results_age_sex()$data_excess)) })

  output$plot_net_quartile_all <- renderPlotly({ req(results_age_sex()); ggplotly(plot_quartile(results_age_sex()$df_quartile_all)) })
  output$plot_net_quartile_male <- renderPlotly({ req(results_age_sex()); ggplotly(plot_quartile(results_age_sex()$df_quartile_male)) })
  output$plot_net_quartile_female <- renderPlotly({ req(results_age_sex()); ggplotly(plot_quartile(results_age_sex()$df_quartile_female)) })

  # --- Auto-run Global au démarrage ---
  observe({
    invalidateLater(1500, session)
    isolate({
      if(is.null(session$userData$autoRunDone) || !session$userData$autoRunDone){
        session$userData$autoRunDone <- TRUE
        addLog("🟢 Lancement automatique Global")
        session$sendCustomMessage("triggerRunGlobal", list())
      }
    })
  })

}
