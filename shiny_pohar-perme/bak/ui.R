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
        choices = c("Chargement manuel" = "manuel",
                    "Chargement automatique (/srv/uploads)" = "auto"),
        selected = "auto"
      ),
      conditionalPanel(
        condition = "input.source == 'manuel'",
        fileInput("hmfile", "Importer hm.csv :", accept = ".csv")
      ),
      actionButton("run", "Lancer l'analyse", class = "btn-primary")
    ),
    
    mainPanel(
      tabsetPanel(id = "tabs",
        # Onglets opur courbes de survie
        tabPanel("Global", plotlyOutput("plot_global")),
        tabPanel("Par sexe", plotlyOutput("plot_sex")),
        tabPanel("Par sexe et âge", plotlyOutput("plot_age_sex")),

        # Onglets pour les TSM
        tabPanel("TSM Global", plotlyOutput("plot_TSM_global")),
        tabPanel("TSM par Sexe", plotlyOutput("plot_TSM_sex")),
        tabPanel("TSM par Âge et Sexe", plotlyOutput("plot_TSM_age_sex")),

        # Onglet pour excès de mortalité
        tabPanel("Excès de mortalité pour 100 000 hab", plotlyOutput("plot_excess")),
        tabPanel("Excès de mortalité pour 100 000 hab, par sexe", plotlyOutput("plot_excess_sex")),
        tabPanel("Excès de mortalité pour 100 000 hab, par quartile d'âge et sexe", plotlyOutput("plot_excess_age_sex")),

        tabPanel("Survie nette 1/5/10 ans par quartile d’âge", plotlyOutput("plot_net_quartile_all")),
        tabPanel("Survie nette 1/5/10 ans par quartile d’âge (hommes)", plotlyOutput("plot_net_quartile_male")),
        tabPanel("Survie nette 1/5/10 ans par quartile d’âge (femmes)", plotlyOutput("plot_net_quartile_female")),

        # Autres onglets de données
        tabPanel("Résumé des données", DT::dataTableOutput("summary_table")),
        tabPanel("Table survie globale", DT::dataTableOutput("table_global")),
        tabPanel("Table survie par sexe", DT::dataTableOutput("table_sex")),
        tabPanel("Table survie âge + sexe", DT::dataTableOutput("table_age_sex")),
        tabPanel("Table de survie (data_lex)", DT::dataTableOutput("lex_table")),
        tabPanel("Messages", verbatimTextOutput("log"))
      )
      
      
    )
  ),
  tags$script(HTML("
  Shiny.addCustomMessageHandler('triggerRun', function(message) {
    setTimeout(function() {
      $('#run').click();
    }, 500);
  });
"))
  
)