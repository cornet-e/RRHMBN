# app.R
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
  
  addLog <- function(msg) {
    old <- logText()
    new <- paste(old, paste0(Sys.time(), " - ", msg), sep = "\n")
    logText(new)
  }
  
  output$log <- renderText({ logText() })
  
  # --- Chargement de pop_haz ---
  pop_haz <- reactive({
    if (!file.exists("pop_haz.csv")) {
      addLog("Création du fichier pop_haz.csv à partir des tables mltper_1x1.txt et fltper_1x1.txt...")
      
      mlt <- read.table("mltper_1x1.txt", header = TRUE, skip = 2)
      flt <- read.table("fltper_1x1.txt", header = TRUE, skip = 2)
      
      pop_haz_men <- expand.grid(age = seq(0, 110, 1), per = seq(1990, 2023, 1))
      pop_haz_men$haz <- mapply(function(a, y) {
        val <- mlt$mx[mlt$Age == a & mlt$Year == y]
        ifelse(length(val) > 0, val, NA)
      }, pop_haz_men$age, pop_haz_men$per)
      pop_haz_men$sex <- 1
      
      pop_haz_women <- expand.grid(age = seq(0, 110, 1), per = seq(1990, 2023, 1))
      pop_haz_women$haz <- mapply(function(a, y) {
        val <- flt$mx[flt$Age == a & flt$Year == y]
        ifelse(length(val) > 0, val, NA)
      }, pop_haz_women$age, pop_haz_women$per)
      pop_haz_women$sex <- 2
      
      pop_haz_df <- rbind(pop_haz_men, pop_haz_women)
      write.csv(pop_haz_df, "pop_haz.csv", row.names = FALSE)
      addLog("pop_haz.csv créé avec succès !")
      pop_haz_df
    } else {
      addLog("pop_haz.csv trouvé et chargé.")
      read.csv("pop_haz.csv")
    }
  })
  
  # --- Chargement du fichier hm.csv ---
  dataInput <- reactive({
    if (input$source == "auto") {
      path <- "/srv/uploads/hm.csv"
      if (!file.exists(path)) {
        addLog("❌ Fichier hm.csv non trouvé dans /srv/uploads. Veuillez le charger manuellement.")
        
        # Affiche une boîte de dialogue pour inviter au chargement manuel
        showModal(modalDialog(
          title = "Fichier manquant",
          "Le fichier hm.csv n’a pas été trouvé dans /srv/uploads.",
          "Veuillez le charger manuellement ci-dessous.",
          easyClose = TRUE,
          footer = modalButton("Fermer")
        ))
        
        return(NULL)
      }
      addLog("✅ Fichier hm.csv trouvé dans /srv/uploads.")
      read.csv(path, sep = ",")
    } else {
      req(input$hmfile)
      addLog("📁 Fichier hm.csv chargé manuellement.")
      read.csv(input$hmfile$datapath, sep = ",")
    }
  })
  
  
  # --- Analyse ---
  results <- eventReactive(input$run, {
    data <- dataInput()
    req(data)
    
    pop_haz_df <- pop_haz()
    
    addLog("Préparation des données...")
    
    names(data)[names(data) == "age"] <- "age_diag"
    data$DateDuDiag <- as.Date(data$DateDuDiag)
    data$DateDernieresNouvelles <- as.Date(data$DateDernieresNouvelles)
    data$time <- as.numeric(data$DateDernieresNouvelles - data$DateDuDiag)
    data$status <- data$event
    data$age_days <- data$age_diag * 365.25
    data$year_frac <- as.numeric(format(data$DateDuDiag, "%Y")) +
      (as.numeric(format(data$DateDuDiag, "%m")) - 0.5) / 12
    data$sex_char <- ifelse(data$sex == 1, "male", "female")
    
    quartiles <- quantile(data$age_diag, probs = seq(0, 1, 0.25), na.rm = TRUE)
    data_prep <- data %>%
      mutate(
        dg_date = DateDuDiag,
        ex_date = DateDernieresNouvelles,
        dg_age = age_diag,
        status_num = status,
        age_quartile = cut(
          age_diag,
          breaks = quartiles,
          include.lowest = TRUE,
          labels = paste0(
            round(head(quartiles, -1)), "–", round(tail(quartiles, -1)), " ans"
          )
        )
      )
    
    addLog("Expansion lexicale (lexpand)...")
    
    pop_haz_df <- pop_haz_df %>%
      rename(agegroup = age, year = per)
    
    data_lex <- lexpand(
      data_prep,
      birth = dg_date - (dg_age * 365.25),
      entry = dg_date,
      exit = ex_date,
      status = status_num,
      pophaz = pop_haz_df,
      pp = TRUE,
      fot = 0:30
    )
    
    pop_haz_df <- pop_haz_df %>%
      rename(age = agegroup, per = year)
    
    addLog("Calcul des survies nettes (Pohar-Perme)...")
    
    fit_net_global <- survtab(
      Surv(time = fot, event = lex.Xst == 1) ~ 1,
      data = data_lex,
      pophaz = pop_haz_df,
      surv.type = "surv.rel",
      relsurv.method = "pp",
      breaks = list(fot = seq(0, 15, 1))
    )
    df_global <- as.data.frame(summary(fit_net_global))
    df_global$surv.exp <- df_global$surv.obs / df_global$r.pp
    
    fit_net_sex <- survtab(
      Surv(time = fot, event = lex.Xst == 1) ~ sex,
      data = data_lex,
      pophaz = pop_haz_df,
      surv.type = "surv.rel",
      relsurv.method = "pp",
      breaks = list(fot = seq(0, 15, 1))
    )
    df_sex <- as.data.frame(summary(fit_net_sex)) %>%
      mutate(surv.exp = surv.obs / r.pp)
    
    fit_net_age_sex <- survtab(
      Surv(time = fot, event = lex.Xst == 1) ~ sex + age_quartile,
      data = data_lex,
      pophaz = pop_haz_df,
      surv.type = "surv.rel",
      relsurv.method = "pp",
      breaks = list(fot = seq(0, 15, 1))
    )
    df_age_sex <- as.data.frame(summary(fit_net_age_sex)) %>%
      mutate(surv.exp = surv.obs / r.pp)
        
    # Calcul du TSM pour chaque DataFrame
    df_global$TSM <- (df_global$surv.exp / df_global$surv.obs - 1) * 100
    df_sex$TSM <- (df_sex$surv.exp / df_sex$surv.obs - 1) * 100
    df_age_sex$TSM <- (df_age_sex$surv.exp / df_age_sex$surv.obs - 1) * 100

      # calcul excess mortality rate par année de suivi
    data_excess <- data_lex %>%
      group_by(fot) %>%
      summarise(
        py = sum(lex.dur, na.rm = TRUE),
        d = sum(lex.Xst == 1, na.rm = TRUE),
        haz_obs = d / py,
        haz_exp = sum(pop.haz * lex.dur, na.rm = TRUE) / py,
        excess = haz_obs - haz_exp
      )

    data_excess_sex <- data_lex %>%
      group_by(fot,sex) %>%
      summarise(
        py = sum(lex.dur, na.rm = TRUE),
        d = sum(lex.Xst == 1, na.rm = TRUE),
        haz_obs = d / py,
        haz_exp = sum(pop.haz * lex.dur, na.rm = TRUE) / py,
        excess = haz_obs - haz_exp
      )

    data_excess_age_sex <- data_lex %>%
      group_by(fot, sex, age_quartile) %>%
      summarise(
        py = sum(lex.dur, na.rm = TRUE),
        d = sum(lex.Xst == 1, na.rm = TRUE),
        haz_obs = d / py,
        haz_exp = sum(pop.haz * lex.dur, na.rm = TRUE) / py,
        excess = haz_obs - haz_exp
      )

    addLog("Analyse terminée avec succès.")

    list(
      global = df_global,
      sex = df_sex,
      age_sex = df_age_sex,
      data_lex = data_lex,
      data_raw = data,
      data_excess = data_excess,
      data_excess_sex = data_excess_sex,
      data_excess_age_sex = data_excess_age_sex
    )
  })
  
  # --- TABLEAU RÉCAPITULATIF ---
  output$summary_table <- DT::renderDataTable({
    req(results())
    data <- results()$data_raw
    
    summary_df <- data.frame(
      Variable = c("Nombre total de patients",
                   "Hommes", "Femmes",
                   "Âge médian (diagnostic)", "Âge minimum", "Âge maximum"),
      Valeur = c(
        nrow(data),
        sum(data$sex == 1, na.rm = TRUE),
        sum(data$sex == 2, na.rm = TRUE),
        round(median(data$age_diag, na.rm = TRUE), 1),
        min(data$age_diag, na.rm = TRUE),
        max(data$age_diag, na.rm = TRUE)
      )
    )
    
    datatable(summary_df,
              rownames = FALSE,
              options = list(dom = "t", pageLength = 10))
  })
  # --- TABLES DE SURVIE ---
  
  # Table survie globale
  output$table_global <- DT::renderDataTable({
    req(results())
    df <- results()$global
    
    datatable(df, options = list(pageLength = 10, scrollX = TRUE)) %>%
      formatStyle(
        c("surv.obs", "r.pp", "surv.exp"),
        fontWeight = "bold",
        color = "black"
      )
  })
  
  # Table survie par sexe
  output$table_sex <- DT::renderDataTable({
    req(results())
    df <- results()$sex
    
    datatable(df, options = list(pageLength = 10, scrollX = TRUE)) %>%
      formatStyle(
        c("surv.obs", "r.pp", "surv.exp"),
        fontWeight = "bold",
        color = "black"
      )
  })
  
  # Table survie âge + sexe
  output$table_age_sex <- DT::renderDataTable({
    req(results())
    df <- results()$age_sex
    
    datatable(df, options = list(pageLength = 10, scrollX = TRUE)) %>%
      formatStyle(
        c("surv.obs", "r.pp", "surv.exp"),
        fontWeight = "bold",
        color = "black"
      )
  })
  
  # --- TABLE data_lex (mise en gras des colonnes) ---
  output$lex_table <- DT::renderDataTable({
    req(results())
    df <- results()$data_lex
    
    # on garde les colonnes utiles
    if (all(c("surv.obs", "r.pp", "surv.exp") %in% names(df))) {
      datatable(df, options = list(pageLength = 10, scrollX = TRUE)) %>%
        formatStyle(
          c("surv.obs", "r.pp", "surv.exp"),
          fontWeight = "bold",
          color = "black"
        )
    } else {
      datatable(df, options = list(pageLength = 10, scrollX = TRUE))
    }
  })
  
  # --- Graphiques interactifs ---
  plot_surv <- function(df, title) {
    ggplot(df, aes(x = Tstop)) +
      geom_line(aes(y = surv.obs, color = "Observée"), linewidth = 1.2) +
      geom_line(aes(y = r.pp, color = "Relative"), linewidth = 1.2) +
      geom_line(aes(y = surv.exp, color = "Attendue"), linetype = "dashed", linewidth = 1) +
      scale_color_manual(
        name = "Type de survie",
        values = c("Observée" = "red", "Relative" = "green", "Attendue" = "black")
      ) +
      labs(title = title, x = "Temps (années)", y = "Probabilité de survie") +
      theme_minimal(base_size = 13)
  }
  
  plot_surv_by_sex <- function(df, title) {
    ggplot(df, aes(x = Tstop)) +
      geom_line(aes(y = surv.obs, color = "Observée"), linewidth = 1.2) +
      geom_line(aes(y = r.pp, color = "Relative"), linewidth = 1.2) +
      geom_line(aes(y = surv.exp, color = "Attendue"), linetype = "dashed", linewidth = 1) +
      facet_wrap(~ sex, labeller = labeller(sex = c(`1` = "Hommes", `2` = "Femmes"))) +
      scale_color_manual(name = "Type de survie",
                         values = c("Observée" = "red", "Relative" = "green", "Attendue" = "black")) +
      labs(title = title, x = "Temps (années)", y = "Probabilité de survie") +
      theme_minimal(base_size = 13)
  }
  
  plot_surv_by_age_sex <- function(df, title) {
    ggplot(df, aes(x = Tstop)) +
      geom_line(aes(y = surv.obs, color = "Observée"), linewidth = 1.2) +
      geom_line(aes(y = r.pp, color = "Relative"), linewidth = 1.2) +
      geom_line(aes(y = surv.exp, color = "Attendue"), linetype = "dashed", linewidth = 1) +
      facet_grid(age_quartile ~ sex,
                 labeller = labeller(sex = c(`1` = "Hommes", `2` = "Femmes"))) +
      scale_color_manual(name = "Type de survie",
                         values = c("Observée" = "red", "Relative" = "green", "Attendue" = "black")) +
      labs(title = title, x = "Temps (années)", y = "Probabilité de survie") +
      theme_minimal(base_size = 13)
  }

  plot_TSM <- function(df, title) {
    ggplot(df, aes(x = Tstop, y = TSM)) +
      geom_line(linewidth = 1.2, color = "blue") +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      labs(title = title, x = "Temps (années)", y = "Taux de Surmortalité (%)") +
      theme_minimal(base_size = 13)
  }

  plot_TSM_by_sex <- function(df, title) {
    ggplot(df, aes(x = Tstop, y = TSM)) +
      geom_line(linewidth = 1.2, color = "blue") +
      facet_wrap(~ sex, labeller = labeller(sex = c(`1` = "Hommes", `2` = "Femmes"))) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      labs(title = title, x = "Temps (années)", y = "Taux de Surmortalité (%)") +
      theme_minimal(base_size = 13)
  }

  plot_TSM_by_age_sex <- function(df, title) {
    ggplot(df, aes(x = Tstop, y = TSM)) +
      geom_line(linewidth = 1.2, color = "blue") +
      facet_grid(age_quartile ~ sex,
                 labeller = labeller(sex = c(`1` = "Hommes", `2` = "Femmes"))) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      labs(title = title, x = "Temps (années)", y = "Taux de Surmortalité (%)") +
      theme_minimal(base_size = 13)
  }
    
  output$plot_global <- renderPlotly({
    req(results())
    ggplotly(plot_surv(results()$global, "Survie brute, nette et attendue - Global"),
             tooltip = c("x", "y", "colour"))
  })
  
  output$plot_sex <- renderPlotly({
    req(results())
    ggplotly(plot_surv_by_sex(results()$sex, "Survie brute, nette et attendue par sexe"),
             tooltip = c("x", "y", "colour"))
  })
  
  output$plot_age_sex <- renderPlotly({
    req(results())
    ggplotly(plot_surv_by_age_sex(results()$age_sex,
                                  "Survie brute, nette et attendue par sexe et quartile d’âge"),
             tooltip = c("x", "y", "colour"))
  })
  
  output$plot_TSM_global <- renderPlotly({
    req(results())
    ggplotly(plot_TSM(results()$global, "Taux de Surmortalité (TSM) - Global"),
             tooltip = c("x", "y"))
  })

  output$plot_TSM_sex <- renderPlotly({
    req(results())
    ggplotly(plot_TSM_by_sex(results()$sex, "Taux de Surmortalité (TSM) par sexe"),
            tooltip = c("x", "y"))
  })

  output$plot_TSM_age_sex <- renderPlotly({
    req(results())
    ggplotly(plot_TSM_by_age_sex(results()$age_sex, "Taux de Surmortalité (TSM) par sexe et quartile d’âge"),
             tooltip = c("x", "y"))
  })

  output$plot_excess <- renderPlotly({
    req(results())
    df <- results()$data_excess

    p <- ggplot(df, aes(x = fot, y = excess)) +
      geom_line(linewidth = 1.2) +
      labs(title = "Taux de mortalité en excès",
          x = "Temps depuis diagnostic (années)",
          y = "Décès par personne-année (excess)") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = c("x","y"))
  })

  output$plot_excess_sex <- renderPlotly({
    req(results())
    df <- results()$data_excess_sex

    p <- ggplot(df, aes(x = fot, y = excess, color=sex)) +
      geom_line(linewidth = 1.2) +
      labs(title = "Taux de mortalité en excès",
          x = "Temps depuis diagnostic (années)",
          y = "Décès par personne-année (excess)") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = c("x","y"))
  })

  output$plot_excess_age_sex <- renderPlotly({
    req(results())
    df <- results()$data_excess_age_sex

    p <- ggplot(data_excess_age_sex, aes(x = fot, y = excess, color = sex)) +
      geom_line(linewidth = 1.1) +
      facet_grid(age_quartile ~ sex,
                labeller = labeller(
                  sex = c(`1` = "Hommes", `2` = "Femmes")
                )) +
      labs(title = "Taux de mortalité en excès selon quartiles d’âge + sexe",
          x = "Temps depuis diagnostic (années)",
          y = "Décès par personne-année (excess)") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = c("x","y"))
  })

  # --- Déclenche automatiquement le clic sur "Lancer l'analyse" à l'ouverture ---
  observe({
    # on attend un tout petit délai pour laisser l'UI se charger
    invalidateLater(1500, session)
    isolate({
      # Déclenche le bouton seulement une fois
      if (is.null(session$userData$autoRunDone) || !session$userData$autoRunDone) {
        session$userData$autoRunDone <- TRUE
        addLog("🟢 Lancement automatique de l'analyse au démarrage...")
        session$sendCustomMessage("triggerRun", list())
      }
    })
  })
  
}
