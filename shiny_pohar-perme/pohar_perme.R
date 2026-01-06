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

ui <- fluidPage(
  titlePanel("Analyse de survie relative (Pohar-Perme)"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Chargement du fichier des patients"),
      
      radioButtons(
        "source",
        "Choix du mode de chargement :",
        choices = c(
          "Chargement manuel" = "manuel",
          "Chargement automatique (/srv/uploads)" = "auto"
        ),
        selected = "auto"
      ),
      
      conditionalPanel(
        condition = "input.source == 'manuel'",
        fileInput("hmfile", "Importer hm.csv :", accept = ".csv")
      ),
      
      h4("Calcul des analyses"),
      actionButton("run_global", "Calculer Global", class = "btn-primary"),
      actionButton("run_sex", "Calculer par Sexe", class = "btn-primary"),
      actionButton("run_age_sex", "Calculer par Âge + Sexe", class = "btn-primary")
    ),
    
    mainPanel(
      tabsetPanel(
        id = "tabs",
        tabPanel("Global", plotlyOutput("plot_global")),
        tabPanel("Par sexe", plotlyOutput("plot_sex")),
        tabPanel("Par âge et sexe", plotlyOutput("plot_age_sex")),
        
        tabPanel("TSM Global", plotlyOutput("plot_TSM_global")),
        tabPanel("TSM par Sexe", plotlyOutput("plot_TSM_sex")),
        tabPanel("TSM par Âge et Sexe", plotlyOutput("plot_TSM_age_sex")),
        
        tabPanel("Excès Global", plotlyOutput("plot_excess")),
        tabPanel("Excès par Sexe", plotlyOutput("plot_excess_sex")),
        tabPanel("Excès Âge+Sexe", plotlyOutput("plot_excess_age_sex")),
        
        tabPanel("Survie nette quartiles (tous)", plotlyOutput("plot_net_quartile_all")),
        tabPanel("Survie nette quartiles Hommes", plotlyOutput("plot_net_quartile_male")),
        tabPanel("Survie nette quartiles Femmes", plotlyOutput("plot_net_quartile_female")),
        
        tabPanel("Résumé des données", DT::dataTableOutput("summary_table")),
        tabPanel("Table survie globale", DT::dataTableOutput("table_global")),
        tabPanel("Table survie par sexe", DT::dataTableOutput("table_sex")),
        tabPanel("Table survie âge + sexe", DT::dataTableOutput("table_age_sex")),
        tabPanel("Table lexpand", DT::dataTableOutput("lex_table")),
        tabPanel("Messages", verbatimTextOutput("log"))
      )
    )
  ),
  
  tags$script(HTML("
    Shiny.addCustomMessageHandler('triggerRunGlobal', function(message) {
      setTimeout(function() {
        $('#run_global').click();
      }, 500);
    });
  "))
)

server <- function(input, output, session) {

  logText <- reactiveVal("")
  addLog <- function(msg){
    old <- logText()
    logText(paste(old, paste0(Sys.time(), " - ", msg), sep="\n"))
  }
  output$log <- renderText({ logText() })

  pop_haz_reactive <- reactive({
    path_csv <- "shiny_pohar-perme/pop_haz.csv"
    if(!file.exists(path_csv)){
      addLog("Création du fichier pop_haz.csv...")
      # Assurez-vous que ces fichiers existent dans le dossier
      mlt <- read.table("shiny_pohar-perme/mltper_1x1.txt", header=TRUE, skip=2)
      flt <- read.table("shiny_pohar-perme/fltper_1x1.txt", header=TRUE, skip=2)
      
      pop_haz_men <- expand.grid(age=0:110, per=1990:2023)
      pop_haz_men$haz <- mapply(function(a,y){ val <- mlt$mx[mlt$Age==a & mlt$Year==y]; ifelse(length(val)==0, NA, val)}, pop_haz_men$age, pop_haz_men$per)
      pop_haz_men$sex <- 1

      pop_haz_women <- expand.grid(age=0:110, per=1990:2023)
      pop_haz_women$haz <- mapply(function(a,y){ val <- flt$mx[flt$Age==a & flt$Year==y]; ifelse(length(val)==0, NA, val)}, pop_haz_women$age, pop_haz_women$per)
      pop_haz_women$sex <- 2

      pop_haz_df <- rbind(pop_haz_men, pop_haz_women)
      if(!dir.exists("shiny_pohar-perme")) dir.create("shiny_pohar-perme")
      write.csv(pop_haz_df, path_csv, row.names=FALSE)
      pop_haz_df
    } else {
      read.csv(path_csv)
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

  prepareData <- reactive({
    data <- dataInput()
    req(data)
    
    # Nettoyage des dates
    data$DateDuDiag <- as.Date(data$DateDuDiag)
    data$DateDernieresNouvelles <- as.Date(data$DateDernieresNouvelles)
    
    # Suppression des cas où durée de survie = 0 (évite l'erreur 'entry == exit')
    data <- data %>% filter(DateDernieresNouvelles > DateDuDiag)
    
    # 1. RENOMMER 'age' pour éviter le conflit avec lexpand
    if("age" %in% names(data)) {
      data <- data %>% rename(age_diag = age)
    }
    
    data$time <- as.numeric(data$DateDernieresNouvelles - data$DateDuDiag)
    data$status <- data$event
    
    # Calcul des quartiles sur le nouvel age_diag
    quartiles <- quantile(data$age_diag, probs=seq(0,1,0.25), na.rm=TRUE)
    
    data %>%
      mutate(
        age_quartile = cut(age_diag, breaks=quartiles, include.lowest=TRUE,
                           labels=paste0(round(head(quartiles,-1)),"–",round(tail(quartiles,-1))," ans")),
        # 2. GARDER 'sex' en INTEGER pour la jointure avec pop_haz
        sex = as.integer(sex) 
      )
  })

  runLex <- function(data){
    # On s'assure que pop_haz est aussi en integer pour le sexe
    pop <- pop_haz_reactive() %>% 
      rename(agegroup=age, year=per) %>%
      mutate(sex = as.integer(sex))
    
    data$agegroup <- floor(data$age_diag)
    data$year <- floor(as.numeric(format(data$DateDuDiag,"%Y")))
    
    # Filtrage des années disponibles dans la table de mortalité
    data <- data %>% dplyr::filter(year >= 1990, year <= 2023)
    
    # Utilisation de age_diag pour définir la date de naissance
    lex <- lexpand(data, 
                   birth = data$DateDuDiag - (data$age_diag * 365.25),
                   entry = data$DateDuDiag, 
                   exit = data$DateDernieresNouvelles,
                   status = data$status, 
                   pophaz = pop, 
                   pp = TRUE, 
                   fot = seq(0, 15, 1))
    lex
  }

  # --- Fonctions de Plot corrigées ---
  
  plot_surv_grouped <- function(df, group_var = NULL) {
    p <- ggplot(df, aes(x = Tstop))
    if (is.null(group_var)) {
      p <- p + 
        geom_line(aes(y = surv.obs, color = "Observée"), size = 1) +
        geom_line(aes(y = r.pp, color = "Relative"), size = 1) +
        geom_line(aes(y = surv.exp, color = "Attendue"), linetype = "dashed")
    } else {
      # Utilisation de interaction pour séparer les lignes proprement
      p <- p + 
        geom_line(aes(y = surv.obs, color = get(group_var), linetype = "Observée")) +
        geom_line(aes(y = r.pp, color = get(group_var), linetype = "Relative"))
    }
    p + scale_y_continuous(limits = c(0, 1)) + labs(y="Survie", color=group_var) + theme_minimal()
  }

  plot_excess_grouped <- function(df, group_var = NULL) {
    p <- ggplot(df, aes(x = fot, y = excess))
    if (is.null(group_var)) {
      p <- p + geom_line(color = "black")
    } else {
      p <- p + geom_line(aes(color = as.factor(get(group_var))))
    }
    p + labs(y="Excès de mortalité", color=group_var) + theme_minimal()
  }

  # --- Analyse Global ---
  results_global <- eventReactive(input$run_global, {
    data <- prepareData()
    lex <- runLex(data)
    addLog("Calcul survie net Global...")
    fit <- survtab(Surv(time=fot, event=lex.Xst==1)~1, data=lex,
                   pophaz=pop_haz_reactive() %>% rename(agegroup=age, year=per),
                   surv.type="surv.rel", relsurv.method="pp", breaks=list(fot=seq(0,15,1)))
    df <- as.data.frame(summary(fit))
    df$surv.exp <- df$surv.obs/df$r.pp
    df$TSM <- (df$surv.exp/df$surv.obs-1)*100
    
    excess <- lex %>% group_by(fot) %>% 
      summarise(py=sum(lex.dur), d=sum(lex.Xst==1), haz_obs=d/py,
                haz_exp=sum(pop.haz*lex.dur)/py, excess=haz_obs-haz_exp)
    
    addLog("Global terminé")
    list(global=df, data_lex=lex, data_raw=data, excess=excess)
  })

  # --- Analyse par Sexe ---
  results_sex <- eventReactive(input$run_sex, {
    data <- prepareData()
    lex <- runLex(data)
    addLog("Calcul survie net par Sexe...")
    fit <- survtab(Surv(time=fot, event=lex.Xst==1)~sex, data=lex,
                   pophaz=pop_haz_reactive() %>% rename(agegroup=age, year=per),
                   surv.type="surv.rel", relsurv.method="pp", breaks=list(fot=seq(0,15,1)))
    df <- as.data.frame(summary(fit))
    df$surv.exp <- df$surv.obs/df$r.pp
    df$TSM <- (df$surv.exp/df$surv.obs-1)*100
    
    excess <- lex %>% group_by(fot, sex) %>% 
      summarise(py=sum(lex.dur), d=sum(lex.Xst==1), haz_obs=d/py,
                haz_exp=sum(pop.haz*lex.dur)/py, excess=haz_obs-haz_exp, .groups="drop")
    
    addLog("Analyse par Sexe terminée")
    list(sex=df, data_excess=excess)
  })

  # --- Analyse par Âge+Sexe ---
  results_age_sex <- eventReactive(input$run_age_sex, {
    data <- prepareData()
    lex <- runLex(data)
    addLog("Calcul survie net par Âge+Sexe...")
    fit <- survtab(Surv(time=fot, event=lex.Xst==1)~sex+age_quartile, data=lex,
                   pophaz=pop_haz_reactive() %>% rename(agegroup=age, year=per),
                   surv.type="surv.rel", relsurv.method="pp", breaks=list(fot=seq(0,15,1)))
    df <- as.data.frame(summary(fit))
    
    # Conversion FORCÉE en dataframe propre pour éviter les conflits d'objets popEpi
    df_raw <- as.data.frame(fit)
    
    # Calcul TSM sécurisé par ligne
    df <- df_raw %>%
      mutate(
        # On s'assure que r.pp et surv.obs existent bien sous ces noms
        # Parfois popEpi les nomme surv.rel.pp
        rel_surv = if("r.pp" %in% names(df_raw)) r.pp else surv.rel,
        surv.exp = ifelse(rel_surv > 0, surv.obs / rel_surv, NA),
        TSM = ifelse(surv.obs > 0 & !is.na(surv.exp), (surv.exp / surv.obs - 1) * 100, NA)
      ) %>%
      # On retire les lignes où le calcul est impossible (division par zéro)
      filter(!is.na(TSM), is.finite(TSM))
    
    # Calcul de l'excès (en s'assurant que py n'est pas nul)
    excess <- lex %>% 
      group_by(fot, sex, age_quartile) %>% 
      summarise(
        py = sum(lex.dur), 
        d = sum(lex.Xst == 1), 
        haz_obs = ifelse(py > 0.001, d / py, 0),
        haz_exp = ifelse(py > 0.001, sum(pop.haz * lex.dur) / py, 0), 
        excess = haz_obs - haz_exp, 
        .groups = "drop"
      )
    
    df_quartile_all <- df %>% filter(Tstop %in% c(1,5,10)) %>% 
      group_by(age_quartile, Tstop) %>% 
      summarise(surv.exp=mean(surv.exp, na.rm=TRUE), .groups="drop") %>% 
      mutate(horizon=as.factor(Tstop))
    df_quartile_male <- df %>% filter(sex==1, Tstop %in% c(1,5,10)) %>% 
      select(age_quartile,Tstop,surv.exp) %>% 
      mutate(horizon=paste0(Tstop," ans"))
    df_quartile_female <- df %>% filter(sex==2, Tstop %in% c(1,5,10)) %>%
      select(age_quartile,Tstop,surv.exp) %>%
      mutate(horizon=paste0(Tstop," ans"))
    
    addLog("Analyse Âge+Sexe terminée")
    list(age_sex=df, data_excess=excess,
         df_quartile_all=df_quartile_all,
         df_quartile_male=df_quartile_male,
         df_quartile_female=df_quartile_female)
  })

  # --- Renders Plotly ---
  # --- 1. COURBES DE SURVIE ---
  output$plot_global <- renderPlotly({ 
    req(results_global())
    ggplotly(plot_surv_grouped(results_global()$global))  %>% 
      layout(
        xaxis = list(title = "Temps de suivi (années)"), 
        yaxis = list(title = "Survie relative (Pohar-Perme)")
      )
  })

  output$plot_sex <- renderPlotly({ 
    req(results_sex())
    df <- results_sex()$sex
    df$Sexe <- factor(df$sex, levels = c(1, 2), labels = c("Hommes", "Femmes"))
    ggplotly(plot_surv_grouped(df, "Sexe")) %>% 
      layout(
        xaxis = list(title = "Temps de suivi (années)"), 
        yaxis = list(title = "Survie relative (Pohar-Perme)")
      )
  })

  output$plot_age_sex <- renderPlotly({ 
    req(results_age_sex())
    df <- results_age_sex()$age_sex
    df$Sexe_Label <- factor(df$sex, levels = c(1, 2), labels = c("H", "F"))
    df$Groupe <- interaction(df$Sexe_Label, df$age_quartile, sep = " - ")
    ggplotly(plot_surv_grouped(df, "Groupe")) %>% 
      layout(
        xaxis = list(title = "Temps de suivi (années)"), 
        yaxis = list(title = "Survie relative (Pohar-Perme)")
      )
  })

  # --- 2. COURBES TSM (Taux de Survie Marginal) ---
  output$plot_TSM_global <- renderPlotly({ 
    req(results_global())
    df <- results_global()$global %>% filter(is.finite(TSM))
    p <- ggplot(df, aes(x = Tstop, y = TSM)) + 
      geom_line(color = "blue", size = 1) + 
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") + 
      labs(y = "TSM (%)", x = "Années de suivi") + theme_minimal()
    ggplotly(p)
  })

  output$plot_TSM_sex <- renderPlotly({ 
    req(results_sex())
    df <- results_sex()$sex %>% filter(is.finite(TSM))
    df$Sexe <- factor(df$sex, levels = c(1, 2), labels = c("Hommes", "Femmes"))
    p <- ggplot(df, aes(x = Tstop, y = TSM, color = Sexe, group = Sexe)) + 
      geom_line(size = 1) + 
      geom_hline(yintercept = 0, linetype = "dashed", color = "black") + 
      labs(y = "TSM (%)", x = "Années de suivi") + theme_minimal()
    ggplotly(p)
  })

  output$plot_TSM_age_sex <- renderPlotly({ 
    req(results_age_sex())
    df <- results_age_sex()$age_sex %>% filter(is.finite(TSM))
    df$Sexe_Label <- factor(df$sex, levels = c(1, 2), labels = c("H", "F"))
    df$Groupe <- interaction(df$Sexe_Label, df$age_quartile, sep = " - ")
    p <- ggplot(df, aes(x = Tstop, y = TSM, color = Groupe, group = Groupe)) + 
      geom_line() + 
      geom_hline(yintercept = 0, linetype = "dashed") + 
      labs(y = "TSM (%)", x = "Années de suivi") + theme_minimal()
    ggplotly(p)
  })

  # --- 3. EXCÈS DE MORTALITÉ ---
  output$plot_excess <- renderPlotly({ 
    req(results_global())
    ggplotly(plot_excess_grouped(results_global()$excess)) 
  })

  output$plot_excess_sex <- renderPlotly({ 
    req(results_sex())
    df <- results_sex()$data_excess
    df$Sexe <- factor(df$sex, levels = c(1, 2), labels = c("Hommes", "Femmes"))
    ggplotly(plot_excess_grouped(df, "Sexe"))  %>% 
      layout(
        xaxis = list(title = "Temps de suivi (années)")
      )
  })

  output$plot_excess_age_sex <- renderPlotly({ 
    req(results_age_sex())
    df <- results_age_sex()$data_excess
    df$Sexe_Label <- factor(df$sex, levels = c(1, 2), labels = c("H", "F"))
    df$Groupe <- interaction(df$Sexe_Label, df$age_quartile, sep = " - ")
    ggplotly(plot_excess_grouped(df, "Groupe"))  %>% 
      layout(
        xaxis = list(title = "Temps de suivi (années)")
      )
  })

  # --- 4. QUARTILES (Survie Nette) ---
  output$plot_net_quartile_all <- renderPlotly({
    req(results_age_sex())
    # Assurez-vous que l'horizon est un facteur pour éviter le dégradé de couleur
    df <- results_age_sex()$df_quartile_all
    df$Horizon <- factor(df$Tstop, levels = c(1, 5, 10), labels = c("1 an", "5 ans", "10 ans"))
    p <- ggplot(df, aes(x = age_quartile, y = surv.exp, color = Horizon, group = Horizon)) +
      geom_line() + geom_point() + labs(y = "Survie nette", x = "Quartiles d'âge") + theme_minimal()
    ggplotly(p)
  })

  # --- Survie nette par quartiles : HOMMES ---
  output$plot_net_quartile_male <- renderPlotly({
    req(results_age_sex())
    df <- results_age_sex()$df_quartile_male
    req(nrow(df) > 0)
    
    # Préparation des étiquettes
    df$Horizon <- factor(df$Tstop, levels = c(1, 5, 10), labels = c("1 an", "5 ans", "10 ans"))
    
    p <- ggplot(df, aes(x = age_quartile, y = surv.exp, color = Horizon, group = Horizon)) +
      geom_line(size = 1) + 
      geom_point(size = 2) +
      labs(title = "Survie nette par quartile - Hommes",
           y = "Survie nette", x = "Quartiles d'âge") +
      scale_y_continuous(limits = c(0, 1)) +
      theme_minimal()
    
    ggplotly(p)
  })

  # --- Survie nette par quartiles : FEMMES ---
  output$plot_net_quartile_female <- renderPlotly({
    req(results_age_sex())
    df <- results_age_sex()$df_quartile_female
    req(nrow(df) > 0)
    
    # Préparation des étiquettes
    df$Horizon <- factor(df$Tstop, levels = c(1, 5, 10), labels = c("1 an", "5 ans", "10 ans"))
    
    p <- ggplot(df, aes(x = age_quartile, y = surv.exp, color = Horizon, group = Horizon)) +
      geom_line(size = 1) + 
      geom_point(size = 2) +
      labs(title = "Survie nette par quartile - Femmes",
           y = "Survie nette", x = "Quartiles d'âge") +
      scale_y_continuous(limits = c(0, 1)) +
      theme_minimal()
    
    ggplotly(p)
  })

  # --- Tables ---
  output$summary_table <- DT::renderDataTable({
    req(results_global())
    data <- results_global()$data_raw
    df <- data.frame(
      Variable=c("Nombre total","Hommes","Femmes","Âge médian"),
      Valeur=c(nrow(data), sum(data$sex==1), sum(data$sex==2), round(median(data$age_diag),1))
    )
    datatable(df, options=list(dom="t"))
  })

  output$table_global <- DT::renderDataTable({ req(results_global()); datatable(results_global()$global) })
  output$table_sex <- DT::renderDataTable({ req(results_sex()); datatable(results_sex()$sex) })
  output$table_age_sex <- DT::renderDataTable({ req(results_age_sex()); datatable(results_age_sex()$age_sex) })
  output$lex_table <- DT::renderDataTable({ req(results_global()); datatable(results_global()$data_lex) })

  # --- Auto-run ---
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

shinyApp(ui, server)