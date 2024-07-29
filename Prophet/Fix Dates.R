library(dplyr)
library(lubridate)

# Function to standardize date
standardize_date <- function(date) {
  formats <- c("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y.%m.%d", "%Y/%m/%d")
  parsed_date <- NA
  
  for (fmt in formats) {
    parsed_date <- tryCatch(as.Date(date, format = fmt), error = function(e) NA)
    if (!is.na(parsed_date)) break
  }
  
  if (is.na(parsed_date)) {
    return(NA)  # Return NA if no format matched
  }
  
  # Explicit check for years 0000 to 0023 and correct them
  year_part <- year(parsed_date)
  if (year_part >= 0 & year_part <= 23) {
    year(parsed_date) <- 2000 + year_part
  }
  
  # Convert to mm/dd/yyyy format
  formatted_date <- format(parsed_date, "%m/%d/%Y")
  return(formatted_date)
}

# Read the CSV file
data <- read.csv("livestock_commodity.csv", stringsAsFactors = FALSE)

# Standardize the date column
data$Date <- sapply(data$Date, standardize_date)

# Check for any NA values in the Date column after parsing
if (any(is.na(data$Date))) {
  warning("Some dates could not be parsed correctly.")
}

# Save the updated data to a new CSV file
write.csv(data, "fixed_commodity.csv", row.names = FALSE)

# Print the updated data
print(data)


