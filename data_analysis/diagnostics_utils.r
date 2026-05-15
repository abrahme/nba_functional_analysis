# Shared utilities sourced by both model_diagnostics.r and latent_space.r.
# Keep this file side-effect-free (no I/O, no global state beyond these definitions).

plot_name <- function(x) if_else(x == "No Name", "Average", x)

posterior_plot_names <- c(
  "Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", "Dwight Howard",
  "Nikola Jokic", "Kevin Garnett", "Steve Nash",
  "Chris Paul", "Shaquille O'Neal", "Anthony Edwards", "Jamal Murray",
  "Donovan Mitchell", "Ray Allen", "Klay Thompson",
  "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki",
  "Jason Kidd", "Marcus Camby", "Rudy Gobert", "Tim Duncan",
  "Manu Ginobili", "James Harden", "Russell Westbrook", "Luka Doncic",
  "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton",
  "LaMelo Ball", "Carmelo Anthony", "Dwyane Wade", "Derrick Rose",
  "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis",
  "Giannis Antetokounmpo", "Jrue Holiday", "No Name"
)

read_parquet_if_exists <- function(path) if (file.exists(path)) read_parquet(path) else NULL
