# Live interactive demo of the Concave Process prior for the JSM presentation.
# Run from the repo root:
#   shiny::runApp("presentation/jsm/concave_prior_app.R")

library(shiny)
library(ggplot2)
library(dplyr)

# shiny::runApp() sets the working directory to this app's directory
source("../gradslam/concave_prior_sampler.R")

# UCSB brand palette (https://brand.ucsb.edu/visual-identity/color)
pal <- c(blue = "#047C91", lightblue = "#DAE6E6", orange = "#EF5645",
         darkblue = "#003660")

ui <- fluidPage(
  titlePanel("Concave Process prior — live sampler"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      sliderInput("lengthscale", "Spectral bandwidth ℓ",
                  min = 0.05, max = 3, value = 1, step = 0.05),
      sliderInput("variance", "Variance σ²",
                  min = 0.005, max = 0.2, value = 0.05, step = 0.005),
      sliderInput("peak_age", "Peak age mean t*",
                  min = 20, max = 36, value = 27, step = 0.5),
      sliderInput("peak_age_sd", "Peak age sd",
                  min = 0, max = 4, value = 1.5, step = 0.25),
      sliderInput("peak_value", "Peak value mean f*",
                  min = -2, max = 4, value = 1, step = 0.25),
      sliderInput("peak_value_sd", "Peak value sd",
                  min = 0, max = 1.5, value = 0.4, step = 0.05),
      sliderInput("n_samples", "# prior draws",
                  min = 5, max = 400, value = 60, step = 5),
      sliderInput("num_basis", "# HSGP basis functions J",
                  min = 3, max = 20, value = 10, step = 1),
      checkboxInput("show_band", "Show 95% band + peak markers", TRUE),
      actionButton("resample", "↻ Resample", class = "btn-primary")
    ),
    mainPanel(
      width = 9,
      plotOutput("prior_plot", height = "640px"),
      helpText(paste(
        "Every draw is concave with its peak exactly at (t*, f*).",
        "Note the variance funnel: pinched at the peak, growing quartically away from it."
      ))
    )
  )
)

server <- function(input, output, session) {
  draws <- reactive({
    input$resample
    sample_concave_prior(
      lengthscale         = input$lengthscale,
      variance            = input$variance,
      peak_age            = input$peak_age,
      peak_value          = input$peak_value,
      peak_age_variance   = max(input$peak_age_sd^2, 1e-12),
      peak_value_variance = max(input$peak_value_sd^2, 1e-12),
      n_samples           = input$n_samples,
      num_basis           = input$num_basis,
      age_grid            = seq(18, 38, by = 0.25)
    )
  })

  output$prior_plot <- renderPlot({
    d <- draws()

    band <- d |>
      group_by(age) |>
      summarize(lo = quantile(value, 0.025), hi = quantile(value, 0.975),
                .groups = "drop")

    peaks <- d |>
      group_by(sample_id) |>
      slice_max(value, n = 1, with_ties = FALSE) |>
      ungroup()

    shown <- d |> filter(sample_id <= min(60, max(d$sample_id)))

    p <- ggplot()
    if (isTRUE(input$show_band)) {
      p <- p + geom_ribbon(data = band, aes(age, ymin = lo, ymax = hi),
                           fill = pal["lightblue"], alpha = 0.5)
    }
    p <- p +
      geom_line(data = shown, aes(age, value, group = sample_id),
                color = pal["blue"], alpha = 0.35, linewidth = 0.5)
    if (isTRUE(input$show_band)) {
      p <- p + geom_point(data = peaks |> semi_join(shown, by = "sample_id"),
                          aes(age, value), color = pal["orange"],
                          size = 1.8, alpha = 0.8)
    }
    p +
      labs(x = "age", y = "latent performance f(t)",
           title = "Draws from the Concave Process prior",
           subtitle = sprintf(
             "ℓ = %.2f · σ² = %.3f · t* ~ N(%.1f, %.2f²) · f* ~ N(%.2f, %.2f²) · J = %d",
             input$lengthscale, input$variance, input$peak_age,
             input$peak_age_sd, input$peak_value, input$peak_value_sd,
             input$num_basis)) +
      theme_bw(base_size = 16) +
      theme(panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", color = pal["darkblue"]))
  })
}

shinyApp(ui, server)
