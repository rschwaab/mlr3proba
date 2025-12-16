test_that("PipeOpRandomEffect train and predict match on same data", {
  # required packages
  library(mlr3)
  library(mlr3fda)
  library(mlr3proba)
  library(mlr3pipelines)
  library(data.table)
  library(pencal)
  library(tf)

  # actual task creation
  data(pbc2data)
  baseline <- pbc2data$baselineInfo
  long <- pbc2data$longitudinalInfo


  # dt for the backend (serChol does not work (maybe because of NAs))
  pbc2_backend = data.table(
    subject_id = baseline$id,
    time = baseline$time,
    event = baseline$event,
    baselineAge = baseline$baselineAge,
    sex = baseline$sex,
    treatment = baseline$treatment
  )

  b <- as_data_backend(pbc2_backend)
  task <- TaskSurv$new("pbc2", b, time = "time", event = "event")

  # make subject_id a group variable (not a feature)
  task$col_roles$group = "subject_id"
  task$col_roles$feature = setdiff(task$col_roles$feature, "subject_id")

  data <- task$data()

  # make it smaller for debugging
  size <- 10
  task$filter(1:size)
  long_small <- long[long$id %in% 1:size, ]
  long_small <- subset(long_small, select = -age)

  # Train and predict with PipeOpRandomEffect on the SAME task

  po_re <- po(
    "random_effect_long",
    long_id_col   = "id",
    long_time_col = "fuptime"
  )

  class(long_small) <- unique(c("Long", class(long_small)))

  task_train <- po_re$train(list(task, long_small))[[1L]]
  task_pred  <- po_re$predict(list(task, long_small))[[1L]]

  dt_train <- task_train$data()
  dt_pred  <- task_pred$data()

  # Should match
  expect_identical(names(dt_train), names(dt_pred))
  expect_equal(dt_train, dt_pred, tolerance = 1e-6)


})
