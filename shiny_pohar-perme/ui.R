library(shiny)
library(plotly)
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
        # Courbes de survie
        tabPanel("Global", plotlyOutput("plot_global")),
        tabPanel("Par sexe", plotlyOutput("plot_sex")),
        tabPanel("Par âge et sexe", plotlyOutput("plot_age_sex")),
        
        # TSM
        tabPanel("TSM Global", plotlyOutput("plot_TSM_global")),
        tabPanel("TSM par Sexe", plotlyOutput("plot_TSM_sex")),
        tabPanel("TSM par Âge et Sexe", plotlyOutput("plot_TSM_age_sex")),
        
        # Excès de mortalité
        tabPanel("Excès Global", plotlyOutput("plot_excess")),
        tabPanel("Excès par Sexe", plotlyOutput("plot_excess_sex")),
        tabPanel("Excès Âge+Sexe", plotlyOutput("plot_excess_age_sex")),
        
        # Survie nette par quartiles
        tabPanel("Survie nette quartiles (tous)", plotlyOutput("plot_net_quartile_all")),
        tabPanel("Survie nette quartiles Hommes", plotlyOutput("plot_net_quartile_male")),
        tabPanel("Survie nette quartiles Femmes", plotlyOutput("plot_net_quartile_female")),
        
        # Tableaux de données
        tabPanel("Résumé des données", DT::dataTableOutput("summary_table")),
        tabPanel("Table survie globale", DT::dataTableOutput("table_global")),
        tabPanel("Table survie par sexe", DT::dataTableOutput("table_sex")),
        tabPanel("Table survie âge + sexe", DT::dataTableOutput("table_age_sex")),
        tabPanel("Table lexpand", DT::dataTableOutput("lex_table")),
        tabPanel("Messages", verbatimTextOutput("log"))
      )
    )
  ),
  
  # JS pour auto-click Global au démarrage
  tags$script(HTML("
    Shiny.addCustomMessageHandler('triggerRunGlobal', function(message) {
      setTimeout(function() {
        $('#run_global').click();
      }, 500);
    });
  "))
)
